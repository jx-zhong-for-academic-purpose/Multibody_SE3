import os
import os.path as osp
import tqdm
import yaml
import argparse
import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from EPN_PointCloud.vgtk.vgtk import batched_select_anchor
from EPN_PointCloud.vgtk.vgtk.functional import so3_mean, compute_rotation_matrix_from_quaternion, \
    compute_rotation_matrix_from_ortho6d
from eq_test_2head_oa_icp import object_aware_icp_with_Rt
from losses.seg_loss_unsup import DynamicLoss, SmoothLoss, InvarianceLoss, EntropyLoss, RankLoss, UnsupervisedOGCLoss, \
    interpolate_mask_by_flow, match_mask_by_iou
from metrics.seg_metric import accumulate_eval_results, calculate_PQ_F1
from pointnet2.pointnet2 import grouping_operation, knn
from utils.data_util import batch_segm_to_mask
from utils.pytorch_util import BNMomentumScheduler, save_checkpoint, checkpoint_state, AverageMeter, RunningAverageMeter

BATCH_FACTOR = 2
EPOCH_UPDATING_FLOW = 25
EPOCH_SUPERVISED_BY_R = 20
EPOCH_NOT_JUST_SEG = 15

def mask_weighting_based_on_R_all(pc1, pc2, flow, mask1, mask2, icp_iter=1, temperature=0.01, R_all=None,
                                  exp_factor=2.0, k_removal=10, res_ignore=1e-5):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :param mask1: (B, N, K) torch.Tensor.
    :param mask2: (B, N, K) torch.Tensor.
    :return:
        flow_update: (B, N, 3) torch.Tensor.
    """
    # Aligh the object ordering in two frames
    mask2_interpolated = interpolate_mask_by_flow(pc1, pc2, mask1, flow)
    perm = match_mask_by_iou(mask2_interpolated, mask2)
    mask2 = torch.einsum('bij,bnj->bni', perm, mask2)

    # Compute object consistency scores
    consistency12 = torch.einsum('bmk,bnk->bmn', mask1, mask2)

    n_batch, n_point, n_object = mask1.size()
    mask1, mask2 = mask1.transpose(1, 2), mask2.transpose(1, 2)
    mask1_rep = mask1.reshape(n_batch * n_object, n_point)
    mask2_rep = mask2.reshape(n_batch * n_object, n_point)
    pc1_rep = pc1.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
    pc2_rep = pc2.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

    for iter in range(icp_iter):
        # Compute soft correspondence scores from nearest-neighbor distances
        dist12 = -torch.cdist(pc1 + flow, pc2) / temperature
        corr12 = dist12.softmax(-1)

        # Filter correspondence scores by object consistency scores
        corr12 = corr12 * consistency12
        row_sum = corr12.sum(-1, keepdim=True).clamp(1e-10)
        corr12 = corr12 / row_sum

        # Update scene flow from object-aware soft correspondences
        flow = torch.einsum('bmn,bnj->bmj', corr12, pc2) - pc1

        flow_rep = flow.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
        # Estimate the rigid transformation
        # object_R, object_t = fit_motion_svd_batch(pc1_rep, pc1_rep + flow_rep, mask1_rep)
        if iter == 0:
            # input_pcs, mask_all: torch.Size([12, 512, 3]) torch.Size([12, 512, 8])
            return point_weighting_batch_with_R_all(pc1_rep, pc1_rep + flow_rep, mask1_rep,
                                                    pc2_rep, mask2_rep, R_all,
                                                    exp_factor, k_removal, res_ignore)

def mask_batchwise_chamfer(pc1, pc2, mask1, mask2, loss_norm=2):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    #print(pc1.shape, pc2.shape, mask1.shape, mask2.shape) torch.Size([2880, 512, 3]) torch.Size([2880, 512, 3]) torch.Size([2880, 512]) torch.Size([2880, 512])
    pc2 = pc2.contiguous()
    pc2_t = pc2.transpose(1, 2).contiguous()
    pc1 = pc1.contiguous() #pc1 = (pc1 + flow).contiguous()
    pc1_t = pc1.transpose(1, 2).contiguous()
    _, idx = knn(1, pc1, pc2)
    nn1 = grouping_operation(pc2_t, idx.detach()).squeeze(-1)
    w_nn1 = grouping_operation(mask2.unsqueeze(1).contiguous(), idx.detach()).squeeze(-1).squeeze(1)
    dist1 = mask1 * (pc1_t - nn1).norm(p=loss_norm, dim=1) #* w_nn1
    _, idx = knn(1, pc2, pc1)
    nn2 = grouping_operation(pc1_t, idx.detach()).squeeze(-1)
    w_nn2 = grouping_operation(mask1.unsqueeze(1).contiguous(), idx.detach()).squeeze(-1).squeeze(1)
    dist2 = mask2 * (pc2_t - nn2).norm(p=loss_norm, dim=1) #* w_nn2
    loss = (dist1.mean(-1) + dist2.mean(-1)) / 2
    #print(loss)
    return loss, nn1, w_nn1, nn2, w_nn2

def point_weighting_batch_with_R_all(pc1, pc2, mask, raw_pc2, raw_mask2, init_R_all,
                                    exp_factor, k_removal, res_ignore):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask: (B, N) torch.Tensor.
    :return:
        R_base: (B, 3, 3) torch.Tensor.
        t_base: (B, 3) torch.Tensor.
    """
    n_batch, n_point, _ = pc1.size()

    if init_R_all is not None:
        assert(mask is not None)
        pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)  # (B, 3)
        pc1_mean.unsqueeze_(1)
        pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
        pc2_mean.unsqueeze_(1)
        raw_pc2_mean = torch.einsum('bnd,bn->bd', raw_pc2, raw_mask2) / torch.sum(raw_mask2, dim=1, keepdim=True)
        raw_pc2_mean.unsqueeze_(1)
        n_candidates = init_R_all.shape[1]
        init_R_all = init_R_all.view(init_R_all.shape[0] * n_candidates, init_R_all.shape[2], init_R_all.shape[3])
        pc1_mean_all = pc1_mean.repeat(n_candidates, 1, 1)
        raw_pc2_mean_all = raw_pc2_mean.repeat(n_candidates, 1, 1)
        pc1_all = pc1.repeat(n_candidates, 1, 1)
        raw_pc2_all = raw_pc2.repeat(n_candidates, 1, 1)
        raw_init_t_all = raw_pc2_mean_all.squeeze(1) - torch.bmm(init_R_all, pc1_mean_all.transpose(1, 2)).squeeze(2)
        pc1_transformed_all = torch.matmul(init_R_all, pc1_all.transpose(1, 2)).transpose(1, 2) + raw_init_t_all.unsqueeze(1)
        mask_all = mask.repeat(n_candidates, 1)
        mask2_all = raw_mask2.repeat(n_candidates, 1)
        residual_after_R_all, nn1, w_nn1, nn2, w_nn2 = mask_batchwise_chamfer(pc1_transformed_all, raw_pc2_all, mask_all, mask2_all)
        residual_after_R_all = residual_after_R_all.view(n_batch, n_candidates)
        residual_after_R_all = torch.nan_to_num(residual_after_R_all, float('inf'))
        residual_min_sum, residual_min_idx = residual_after_R_all.min(dim=1, keepdim=True)

        pc1_transformed_all = pc1_transformed_all.view(n_batch, n_candidates, n_point, 3)
        pc1_transformed = torch.take_along_dim(pc1_transformed_all, residual_min_idx.view(n_batch ,1, 1, 1), dim=1).squeeze(1)
        residual_pointwise = (pc2 - pc1_transformed) #* mask.unsqueeze(-1)
        residual_pointwise = torch.square(residual_pointwise).sum(-1)
        weighting = torch.ones_like(mask)
        weighting[(mask * residual_pointwise).sum(-1) < res_ignore] *= torch.exp(-exp_factor * residual_pointwise)[(mask * residual_pointwise).sum(-1) < res_ignore]
        return weighting

class Trainer(object):
    def __init__(self,
                 segnet,
                 criterion,
                 optimizer,
                 aug_transform_epoch,
                 ignore_npoint_thresh,
                 exp_base,
                 lr_scheduler=None,
                 bnm_scheduler=None):
        self.segnet = segnet
        self.criterion = criterion
        self.optimizer = optimizer
        self.aug_transform_epoch = aug_transform_epoch
        self.ignore_npoint_thresh = ignore_npoint_thresh
        self.lr_scheduler = lr_scheduler
        self.bnm_scheduler = bnm_scheduler

        self.exp_base = exp_base
        os.makedirs(exp_base, exist_ok=True)
        self.checkpoint_name, self.best_name = "current", "best"
        self.cur_epoch = 0
        self.training_best, self.eval_best = {}, {}
        self.viz = SummaryWriter(osp.join(exp_base, 'log'))
        self.flow1_dict = {}
        self.flow2_dict = {}
        self.R1_dict = {}
        self.R2_dict = {}

    def _train_it(self, it, batch, aug_transform=False):
        # Forward
        with torch.set_grad_enabled(True):
            pcs, segms, flows, ids = batch
            batch_sid = ids[:, 0]
            b, t, n = segms.size()

            training_seg = False if it * b > EPOCH_NOT_JUST_SEG * len(train_set) and (it // BATCH_FACTOR) % 2 == 0 else True
            seg_supervised_by_R = True if it * b > EPOCH_SUPERVISED_BY_R * len(train_set) else False
            temporal_ensemble_update_flow = True if it * b > EPOCH_UPDATING_FLOW * len(train_set) else False

            pcs = pcs.view(b * t, n, -1).contiguous().cuda()
            if not training_seg:
                with torch.no_grad():
                    masks, _, _, _, _ = self.segnet(pcs)
            else:
                masks, _, _, _, _ = self.segnet(pcs)
            input_pcs, input_masks = pcs.detach(), masks.detach()

            pcs = pcs.view(b, t, n, -1).contiguous()
            masks = masks.view(b, t, n, -1).contiguous()

            pcs = [pcs[:, tt].contiguous() for tt in range(t)]
            masks = [masks[:, tt].contiguous() for tt in range(t)]
            flows = [flows[:, tt].contiguous().cuda() for tt in range(t)]

            temporal_ensemble_flow1 = []
            temporal_ensemble_flow2 = []
            for idx_s, sid in enumerate(batch_sid):
                sid = int(sid)
                if sid not in self.flow1_dict:
                    self.flow1_dict[sid] = flows[0][idx_s]
                if sid not in self.flow2_dict:
                    self.flow2_dict[sid] = flows[1][idx_s]
                temporal_ensemble_flow1.append(self.flow1_dict[sid])
                temporal_ensemble_flow2.append(self.flow2_dict[sid])
            temporal_ensemble_flow1 = torch.stack(temporal_ensemble_flow1, 0).detach().clone()
            temporal_ensemble_flow2 = torch.stack(temporal_ensemble_flow2, 0).detach().clone()
            temporal_ensemble_flows = (temporal_ensemble_flow1, temporal_ensemble_flow2)

            if (not training_seg) or temporal_ensemble_update_flow:
                with torch.no_grad():
                    assert(len(flows) == len(masks) == len(pcs) == 2)
                    flow1, flow2 = flows[0], flows[1]
                    input_segm1, input_segm2 = masks[0], masks[1]
                    x1, x2 = pcs[0], pcs[1]
                    flow1_update, R1, t1 = object_aware_icp_with_Rt(x1, x2, temporal_ensemble_flow1,
                                                                    input_segm1, input_segm2,
                                                                    icp_iter = 20, temperature = 0.01)
                    flow2_update, R2, t2 = object_aware_icp_with_Rt(x2, x1, temporal_ensemble_flow2,
                                                                    input_segm2, input_segm1,
                                                                    icp_iter = 20, temperature = 0.01)
                    R1 = R1.detach()
                    R2 = R2.detach()

                    temporal_ensemble_R1 = []
                    temporal_ensemble_R2 = []
                    for idx_s, sid in enumerate(batch_sid):
                        sid = int(sid)
                        if sid not in self.R1_dict:
                            self.R1_dict[sid] = R1[idx_s].detach()
                        if sid not in self.R2_dict:
                            self.R2_dict[sid] = R2[idx_s].detach()
                        temporal_ensemble_R1.append(self.R1_dict[sid])
                        temporal_ensemble_R2.append(self.R2_dict[sid])
                        old_R_weight = 0.9
                        self.R1_dict[sid] = old_R_weight * self.R1_dict[sid] + (1 - old_R_weight) * R1[idx_s]
                        self.R2_dict[sid] = old_R_weight * self.R2_dict[sid] + (1 - old_R_weight) * R2[idx_s]
                        old_flow_weight = 0.9
                        self.flow1_dict[sid] = old_flow_weight * self.flow1_dict[sid] + (1 - old_flow_weight) * flow1_update[idx_s]
                        self.flow2_dict[sid] = old_flow_weight * self.flow2_dict[sid] + (1 - old_flow_weight) * flow2_update[idx_s]

                    temporal_ensemble_R1 = torch.stack(temporal_ensemble_R1, 0).detach()
                    temporal_ensemble_R2 = torch.stack(temporal_ensemble_R2, 0).detach()
                    temporal_ensemble_R = (temporal_ensemble_R1 + temporal_ensemble_R2.transpose(-1, -2)) / 2

            if not training_seg:
                with torch.no_grad():
                    loss, loss_dict = self.criterion(pcs, masks, temporal_ensemble_flows,
                                                     step_w=True, it=(it * b), aug_transform=aug_transform)
                _, _, _, _, rot_loss_list = self.segnet(input_pcs, input_masks, temporal_ensemble_R)
                loss = sum([l_it for l_it in rot_loss_list[0]]) / len(rot_loss_list[0]) / 10#2.50
                loss_dict['R_rot_loss'] = sum([l_it.item() for l_it in rot_loss_list[0]]) / len(rot_loss_list[0])
                loss_dict['R_l2_loss'] = sum([l_it.item() for l_it in rot_loss_list[2]]) / len(rot_loss_list[2])
            else:
                #torch.cuda.empty_cache()
                if seg_supervised_by_R:
                    with torch.no_grad():
                        _, _, confidence, quats, rot_loss_list = self.segnet(input_pcs, input_masks)
                        b, parts_cnt, nr, na, na = quats.shape
                        confidence = confidence.view(b * parts_cnt, na, na)
                        quats = quats.view(b * parts_cnt, nr, na, na)
                        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d
                        # print(pred_RAnchor.shape) torch.Size([48, 60, 3, 3])
                        conf_selected_all, preds_all = confidence.topk(na, dim=1)
                        quats_all = quats.repeat(na, 1, 1, 1)
                        preds_all = preds_all.view(b * parts_cnt * na, na)
                        conf_selected_all = conf_selected_all.view(b * parts_cnt * na, na)
                        pred_RAnchor_all = batched_select_anchor(preds_all, quats_all, rotation_mapping)
                        anchors_src_all = segnet.module.eq_enc.get_anchor()[None].expand(b * parts_cnt * na, -1, -1,
                                                                                         -1).contiguous()
                        pred_Rs_all = torch.einsum('baij, bajk, balk -> bail', \
                                                   anchors_src_all.cuda(), pred_RAnchor_all.cuda(),
                                                   segnet.module.eq_enc.get_anchor()[preds_all].cuda())
                        conf_selected_all = conf_selected_all / (1e-6 + torch.sum(conf_selected_all, 1, keepdim=True))
                        pred_R_all = so3_mean(pred_Rs_all, conf_selected_all)
                        pred_R_all = pred_R_all.view(b * parts_cnt, na, 3, 3)

                        mask1_weight = mask_weighting_based_on_R_all(x1.detach(), x2.detach(),
                                                                     temporal_ensemble_flow1.detach(),
                                                                     input_segm1.detach(), input_segm2.detach(),
                                                                     R_all=pred_R_all.detach(),#)
                                                                     exp_factor=0.5 if args.dataset == 'ogcdr' else 2.0,
                                                                     res_ignore = 5 if args.dataset == 'ogcdr' else 1e-5)

                        mask2_weight = mask_weighting_based_on_R_all(x2.detach(), x1.detach(),
                                                                     temporal_ensemble_flow2.detach(),
                                                                     input_segm2.detach(), input_segm1.detach(),
                                                                     R_all=pred_R_all.transpose(-1, -2).detach(), #)
                                                                     exp_factor=0.5 if args.dataset == 'ogcdr' else 2.0,
                                                                     res_ignore=5 if args.dataset == 'ogcdr' else 1e-5)

                        mask1_weight = mask1_weight.view(b, parts_cnt, n).transpose(-1, -2)
                        mask2_weight = mask2_weight.view(b, parts_cnt, n).transpose(-1, -2)
                    masks_weight = (mask1_weight, mask2_weight)
                    loss, loss_dict = self.criterion(pcs, masks, temporal_ensemble_flows,
                                                     step_w=True, it=(it * b), aug_transform=aug_transform,
                                                     masks_weight=masks_weight)
                else:
                    loss, loss_dict = self.criterion(pcs, masks, temporal_ensemble_flows,
                                                     step_w=True, it=(it * b), aug_transform=aug_transform)



        segm = segms[:, 0]
        mask = masks[0].detach().cpu()

        # Backward
        try:
            loss /= BATCH_FACTOR
            loss.backward()
        except RuntimeError as runtime_err:
            print('RuntimeError of loss.backward():', runtime_err)
            return loss_dict, segm, mask

        for param in self.segnet.parameters():
            if param.grad is not None and torch.any(torch.isnan(param.grad)):
                return loss_dict, segm, mask

        return loss_dict, segm, mask

    def eval_epoch(self, d_loader):
        if self.segnet is not None:
            self.segnet.eval()

        eval_meter = AverageMeter()
        total_loss = 0.0
        count = 1.0

        ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
        with tqdm.tqdm(enumerate(d_loader, 0), total=len(d_loader), leave=False, desc='val') as tbar:
            for i, batch in tbar:
                with torch.set_grad_enabled(False):
                    pcs, segms, flows, _ = batch

                    b, t, n = segms.size()
                    pcs = pcs.view(b * t, n, -1).contiguous().cuda()
                    masks, _, _, _, _ = self.segnet(pcs)
                    input_pcs, input_masks = pcs.detach(), masks.detach()

                    pcs = pcs.view(b, t, n, -1).contiguous()
                    masks = masks.view(b, t, n, -1).contiguous()

                    pcs = [pcs[:, tt].contiguous() for tt in range(t)]
                    masks = [masks[:, tt].contiguous() for tt in range(t)]
                    flows = [flows[:, tt].contiguous().cuda() for tt in range(t)]
                    loss, loss_dict = self.criterion(pcs, masks, flows, step_w=False)

                    assert (len(flows) == len(pcs) == 2)
                    flow1, flow2 = flows[0], flows[1]
                    assert (t == 2)
                    input_segm1, input_segm2 = segms.split(1, 1) #masks[0], masks[1]
                    input_segm1, _ = batch_segm_to_mask(input_segm1.squeeze(1).detach().cpu().numpy(),
                                                        args.segnet['n_slot'], ignore_npoint_thresh=0)
                    input_segm2, _ = batch_segm_to_mask(input_segm2.squeeze(1).detach().cpu().numpy(),
                                                        args.segnet['n_slot'], ignore_npoint_thresh=0)
                    input_segm1 = torch.from_numpy(input_segm1).cuda()
                    input_segm2 = torch.from_numpy(input_segm2).cuda()
                    x1, x2 = pcs[0], pcs[1]
                    flow_update1, R, t = object_aware_icp_with_Rt(x1, x2, flow1, input_segm1, input_segm2,
                                                                  icp_iter=20, temperature=0.01)
                    R = R.detach()
                    _, _, _, _, rot_loss_list = self.segnet(input_pcs, input_masks, R)
                    loss_dict['R_rot_loss'] = sum([l_it.item() for l_it in rot_loss_list[0]]) / len(rot_loss_list[0])
                    loss_dict['R_l2_loss'] = sum([l_it.item() for l_it in rot_loss_list[2]]) / len(rot_loss_list[2])

                total_loss += loss.item()
                count += 1
                eval_meter.append_loss(loss_dict)
                tbar.set_postfix(eval_meter.get_mean_loss_dict())

                segm = segms[:, 0]
                mask = masks[0].detach().cpu()

                Pred_IoU, Pred_Matched, _, N_GT_Inst = accumulate_eval_results(segm, mask, self.ignore_npoint_thresh)
                ap_eval_meter['Pred_IoU'].append(Pred_IoU)
                ap_eval_meter['Pred_Matched'].append(Pred_Matched)
                ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

        return total_loss / count, eval_meter.get_mean_loss_dict(), ap_eval_meter


    def train(self, n_epochs, train_set, train_loader, test_loader=None):
        it = 0
        best_loss = 1e10
        aug_transform = False
        # Save init model.
        save_checkpoint(
            checkpoint_state(self.segnet), True,
            filename=osp.join(self.exp_base, self.checkpoint_name),
            bestname=osp.join(self.exp_base, self.best_name))

        with tqdm.trange(1, n_epochs + 1, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            for epoch in tbar:
                train_meter = AverageMeter()
                train_running_meter = RunningAverageMeter(alpha=0.3)
                self.cur_epoch = epoch

                # Induce augmented transformation (for the invariance loss) at the specified epoch
                if self.cur_epoch == (self.aug_transform_epoch + 1):
                    aug_transform = True
                    train_set.aug_transform = True
                    best_loss = 1e10

                ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
                for batch in train_loader:
                    self.segnet.train()
                    if it % BATCH_FACTOR == 0:
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step(it)
                        if self.bnm_scheduler is not None:
                            self.bnm_scheduler.step(it)
                        self.optimizer.zero_grad()
                    loss_dict, segm, mask = self._train_it(it, batch, aug_transform=aug_transform)
                    if it % BATCH_FACTOR == 0:
                        self.optimizer.step()
                    it += 1
                    pbar.update()
                    train_running_meter.append_loss(loss_dict)
                    pbar.set_postfix(train_running_meter.get_loss_dict())

                    # Monitor loss
                    tbar.refresh()
                    for loss_name, loss_val in loss_dict.items():
                        self.viz.add_scalar('train/'+loss_name, loss_val, global_step=it)
                    train_meter.append_loss(loss_dict)

                    # Monitor by quantitative evaluation metrics
                    Pred_IoU, Pred_Matched, _, N_GT_Inst = accumulate_eval_results(segm, mask, self.ignore_npoint_thresh)
                    ap_eval_meter['Pred_IoU'].append(Pred_IoU)
                    ap_eval_meter['Pred_Matched'].append(Pred_Matched)
                    ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

                    if (it % len(train_loader)) == 0:
                        pbar.close()

                        # Accumulate train loss and metrics in the epoch
                        train_avg = train_meter.get_mean_loss_dict()
                        for meter_key, meter_val in train_avg.items():
                            self.viz.add_scalar('epoch_sum_train/' + meter_key, meter_val, global_step=epoch)
                        Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
                        Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
                        N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
                        PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
                        self.viz.add_scalar('epoch_sum_train/PQ@50:', PQ, global_step=epoch)
                        self.viz.add_scalar('epoch_sum_train/F1@50:', F1, global_step=epoch)
                        self.viz.add_scalar('epoch_sum_train/Pre@50', Pre, global_step=epoch)
                        self.viz.add_scalar('epoch_sum_train/Rec@50', Rec, global_step=epoch)

                        # Test on the validation set
                        if test_loader is not None:
                            val_loss, val_avg, ap_eval_meter = self.eval_epoch(test_loader)
                            for meter_key, meter_val in val_avg.items():
                                self.viz.add_scalar('epoch_sum_val/'+meter_key, np.mean(val_avg[meter_key]), global_step=epoch)
                            Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
                            Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
                            N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
                            PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
                            self.viz.add_scalar('epoch_sum_val/PQ@50:', PQ, global_step=epoch)
                            self.viz.add_scalar('epoch_sum_val/F1@50:', F1, global_step=epoch)
                            self.viz.add_scalar('epoch_sum_val/Pre@50', Pre, global_step=epoch)
                            self.viz.add_scalar('epoch_sum_val/Rec@50', Rec, global_step=epoch)

                            is_best = val_loss < best_loss
                            best_loss = min(best_loss, val_loss)
                            save_checkpoint(
                                checkpoint_state(self.segnet),
                                is_best,
                                filename=osp.join(self.exp_base, self.checkpoint_name),
                                bestname=osp.join(self.exp_base, self.best_name))

                            # Also save intermediate epochs
                            save_checkpoint(
                                checkpoint_state(self.segnet),
                                is_best,
                                filename=osp.join(self.exp_base, 'epoch_%03d'%(self.cur_epoch)),
                                bestname=osp.join(self.exp_base, self.best_name))

                        pbar = tqdm.tqdm(
                            total=len(train_loader), leave=False, desc='train')
                        pbar.set_postfix(dict(total_it=it))

                    self.viz.flush()

        return best_loss


def lr_curve(it):
    return max(
        args.lr_decay ** (int(it * args.batch_size / args.decay_step)),
        args.lr_clip / args.lr,
    )


def bn_curve(it):
    if args.decay_step == -1:
        return args.bn_momentum
    else:
        return max(
            args.bn_momentum
            * args.bn_decay ** (int(it * args.batch_size / args.decay_step)),
            1e-2,
        )

from matplotlib import pyplot as plt
def show_video_point_clouds(video_point_cloud, continous=True, save_fig=None, color=None):
    video_length = len(video_point_cloud)
    x_min = y_min = 20000
    x_max = y_max = -20000
    for i in range(video_length):
        pts = video_point_cloud[i]
        y_min = min(pts[:, 0].min(), y_min)
        y_max = max(pts[:, 0].max(), y_max)
        x_min = min(pts[:, 1].min(), x_min)
        x_max = max(pts[:, 1].max(), x_max)
        length = max(y_max - y_min, x_max - x_min)
    if continous:
        plt.ion()
        fig = plt.figure()
        for i in range(video_length):
            show_frame_point_clouds(video_point_cloud[i], fig, (x_min, x_min + length),
                                    (y_min, y_min + length), show_img=False,
                                    color=None if color is None else color[i])
            plt.pause(5) #0.5)
            plt.clf()
        plt.ioff()
        plt.close(fig)
    else:
        for i in range(video_length):
            if save_fig is not None:
                fig = plt.figure()
                show_frame_point_clouds(video_point_cloud[i], fig, (x_min, x_min + length),
                                        (y_min, y_min + length), show_img=False,
                                        color=None if color is None else color[i])
                plt.savefig(save_fig + str(i).zfill(3) + ".jpg")
                plt.close(fig)
            else:
                fig = plt.figure()
                show_frame_point_clouds(video_point_cloud[i], fig, (x_min, x_min + length), (y_min, y_min + length))

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
def show_frame_point_clouds(point_cloud, fig, x_minmax, y_minmax, show_img=True, color=None):
    ax = fig.add_subplot(111, projection='3d')
    ys = point_cloud[:, 0]
    xs = x_minmax[1] - point_cloud[:, 1] + x_minmax[0]
    zs = point_cloud[:, 2]
    ax.scatter(xs, ys, zs, c=zs if color is None else color,
               cmap=plt.get_cmap("jet"),
               alpha=1, s=15,
               vmin=100 if color is None else color.min(),
               vmax=500 if color is None else color.max())
    ax.set_ylim(y_minmax[0], y_minmax[1])
    ax.set_xlim(x_minmax[0], x_minmax[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=87, azim=90)
    if show_img:
        plt.show()
    return ax

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--round', type=int, default=0, help='Which round of iterative optimization')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Fix the random seed
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Configuration for different dataset
    data_root = args.data['root']
    if args.dataset == 'sapien':
        from models.eq_2head_sapien import MaskFormer3D
        from datasets.dataset_sapien import SapienDataset as TrainDataset
    else:
        raise KeyError('Unrecognized dataset!')

    # Setup the network
    segnet = MaskFormer3D(n_slot=args.segnet['n_slot'],
                          n_point=args.segnet['n_point'],
                          use_xyz=args.segnet['use_xyz'],
                          n_transformer_layer=args.segnet['n_transformer_layer'],
                          transformer_embed_dim=args.segnet['transformer_embed_dim'],
                          transformer_input_pos_enc=args.segnet['transformer_input_pos_enc']).cuda()

    weight_path = osp.join(args.save_path + '_R%d' % (args.round), 'best.pth.tar')
    segnet = torch.nn.DataParallel(segnet)

    # Setup the scene flow source
    if args.round > 1:
        predflow_path = args.predflow_path + '_R%d'%(args.round - 1)
    else:
        predflow_path = args.predflow_path

    # Setup the dataset
    if args.dataset in ['sapien', 'ogcdr']:
        view_sels = [[0, 1], [1, 2], [2, 3]]
        if args.dataset == 'sapien':
            data_root = osp.join(data_root, 'mbs-shapepart')
        train_set = TrainDataset(data_root=data_root,
                                 split='train',
                                 view_sels=view_sels,
                                 predflow_path=predflow_path,
                                 aug_transform_args=args.data['aug_transform_args'],
                                 decentralize=args.data['decentralize'], need_id=True)
        val_set = TrainDataset(data_root=data_root,
                               split='val',
                               view_sels=view_sels,
                               predflow_path=predflow_path,
                               decentralize=args.data['decentralize'], need_id=True)
    else:       # KITTI-SF
        view_sels = [[0, 1]]
        train_set = TrainDataset(data_root=data_root,
                                 mapping_path=args.data['train_mapping'],
                                 downsampled=True,
                                 view_sels=view_sels,
                                 predflow_path=predflow_path,
                                 aug_transform_args=args.data['aug_transform_args'],
                                 decentralize=args.data['decentralize'], need_id=True)
        val_set = TrainDataset(data_root=data_root,
                               mapping_path=args.data['val_mapping'],
                               downsampled=True,
                               view_sels=view_sels,
                               predflow_path=predflow_path,
                               decentralize=args.data['decentralize'])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Setup the optimizer
    optimizer = optim.Adam(segnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_curve)
    bnm_scheduler = BNMomentumScheduler(segnet, bn_lambda=bn_curve)

    # Setup the loss
    dynamic_loss = DynamicLoss(**args.loss['dynamic_loss_params'])
    smooth_loss = SmoothLoss(**args.loss['smooth_loss_params'])
    invariance_loss = InvarianceLoss(**args.loss['invariance_loss_params'])
    entropy_loss = EntropyLoss()
    rank_loss = RankLoss()
    criterion = UnsupervisedOGCLoss(dynamic_loss, smooth_loss, invariance_loss, entropy_loss, rank_loss,
                                    weights=args.loss['weights'], start_steps=args.loss['start_steps'])

    # Setup the trainer
    trainer = Trainer(segnet=segnet,
                      criterion=criterion,
                      optimizer=optimizer,
                      aug_transform_epoch=args.aug_transform_epoch,
                      ignore_npoint_thresh=args.ignore_npoint_thresh,
                      exp_base=args.save_path + '_R%d'%(args.round),
                      lr_scheduler=lr_scheduler,
                      bnm_scheduler=bnm_scheduler)

    # Train
    trainer.train(args.epochs, train_set, train_loader, val_loader)
