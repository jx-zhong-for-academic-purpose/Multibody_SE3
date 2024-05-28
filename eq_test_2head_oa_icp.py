import os
import os.path as osp
import tqdm
import yaml
import argparse
import json
import torch
from torch.utils.data import DataLoader

from losses.seg_loss_unsup import fit_motion_svd_batch, interpolate_mask_by_flow, match_mask_by_iou
from metrics.flow_metric import eval_flow
from pointnet2.pointnet2 import knn, grouping_operation
from utils.pytorch_util import AverageMeter


def object_aware_icp_with_Rt(pc1, pc2, flow, mask1, mask2, icp_iter=10, temperature=0.01):
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
    #print(pc1.shape, pc2.shape, flow.shape, mask1.shape, mask2.shape)
    mask2_interpolated = interpolate_mask_by_flow(pc1, pc2, mask1, flow)
    perm = match_mask_by_iou(mask2_interpolated, mask2)
    mask2 = torch.einsum('bij,bnj->bni', perm, mask2)

    # Compute object consistency scores
    consistency12 = torch.einsum('bmk,bnk->bmn', mask1, mask2)

    n_batch, n_point, n_object = mask1.size()
    mask1, mask2 = mask1.transpose(1, 2), mask2.transpose(1, 2)
    mask1_rep = mask1.reshape(n_batch * n_object, n_point)
    pc1_rep = pc1.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

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
        object_R, object_t = fit_motion_svd_batch(pc1_rep, pc1_rep + flow_rep, mask1_rep)
        # Apply the estimated rigid transformation onto point cloud
        pc1_transformed = torch.einsum('bij,bnj->bni', object_R, pc1_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc1_transformed = pc1_transformed.reshape(n_batch, n_object, n_point, 3)
        flow = torch.einsum('bkn,bkni->bni', mask1, pc1_transformed) - pc1
    return flow, object_R.reshape(n_batch, n_object, object_R.shape[1], object_R.shape[2]),\
           object_t.reshape(n_batch, n_object, object_t.shape[1])

def fit_motion_svd_batch_with_R(pc1, pc2, mask=None, init_R=None):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param mask: (B, N) torch.Tensor.
    :return:
        R_base: (B, 3, 3) torch.Tensor.
        t_base: (B, 3) torch.Tensor.
    """
    n_batch, n_point, _ = pc1.size()

    if init_R is not None:
        if mask is None:
            pc1_mean = torch.mean(pc1, dim=1, keepdim=True)  # (B, 1, 3)
            pc2_mean = torch.mean(pc2, dim=1, keepdim=True)  # (B, 1, 3)
        else:
            pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)  # (B, 3)
            pc1_mean.unsqueeze_(1)
            pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
            pc2_mean.unsqueeze_(1)
        init_t = pc2_mean.squeeze(1) - torch.bmm(init_R, pc1_mean.transpose(1, 2)).squeeze(2)
        pc1_transformed = torch.matmul(init_R, pc1.transpose(1, 2)).transpose(1, 2) + init_t.unsqueeze(1)
        residual = torch.square(pc2 - pc1_transformed).sum(2)
        assert(mask is not None)
        #'''
        k = 10
        thr = torch.topk(residual, k, dim=1)[0].min(dim=1, keepdim=True)[0] #torch.median(residual, dim=1, keepdim=True)[0]
        mask[residual > thr] = 0
        '''
        mask = residual
        #'''

    if mask is None:
        pc1_mean = torch.mean(pc1, dim=1, keepdim=True)   # (B, 1, 3)
        pc2_mean = torch.mean(pc2, dim=1, keepdim=True)   # (B, 1, 3)
    else:
        pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)   # (B, 3)
        pc1_mean.unsqueeze_(1)
        pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
        pc2_mean.unsqueeze_(1)

    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    if mask is None:
        S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
    else:
        S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask).bmm(pc2_centered))

    # If mask is not well-defined, S will be ill-posed.
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.zeros((n_batch, 3), device=pc1.device)

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        # Correct reflection matrix to rotation matrix
        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
        t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)
        R_base[valid_batches] = R #torch.matmul(R, init_R[valid_batches])
        t_base[valid_batches] = t #t + torch.matmul(R, init_t[valid_batches].unsqueeze(2)).squeeze(2)

    return R_base, t_base

def icp_based_on_R(pc1, pc2, flow, mask1, mask2, icp_iter=10, temperature=0.01, R=None):
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

    object_R = R

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
        if iter == 0:
            object_R, object_t = fit_motion_svd_batch_with_R(pc1_rep, pc1_rep + flow_rep, mask1_rep, object_R)
        else:
            object_R, object_t = fit_motion_svd_batch(pc1_rep, pc1_rep + flow_rep, mask1_rep)
        # Apply the estimated rigid transformation onto point cloud
        pc1_transformed = torch.einsum('bij,bnj->bni', object_R, pc1_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc1_transformed = pc1_transformed.reshape(n_batch, n_object, n_point, 3)
        flow = torch.einsum('bkn,bkni->bni', mask1, pc1_transformed) - pc1
    return flow

def weighted_kabsch(pc, flow, mask):
    """
    :param pc: (B, N, 3) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :param mask: (B, N, K) torch.Tensor.
    :return:
        flow: (B, N, 3) torch.Tensor.
    """
    n_batch, n_point, n_object = mask.size()
    mask = mask.transpose(1, 2)
    mask = mask.reshape(n_batch * n_object, n_point)
    pc_rep = pc.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)
    flow_rep = flow.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

    # Estimate the rigid transformation
    object_R, object_t = fit_motion_svd_batch(pc_rep, pc_rep + flow_rep, mask)
    # Apply the estimated rigid transformation onto point cloud
    pc_transformed = torch.einsum('bij,bnj->bni', object_R, pc_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
    pc_transformed = pc_transformed.reshape(n_batch, n_object, n_point, 3)
    mask = mask.reshape(n_batch, n_object, n_point)
    flow = torch.einsum('bkn,bkni->bni', mask, pc_transformed) - pc
    return flow


def object_aware_icp(pc1, pc2, flow, mask1, mask2, icp_iter=10, temperature=0.01):
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
    pc1_rep = pc1.unsqueeze(1).repeat(1, n_object, 1, 1).reshape(n_batch * n_object, n_point, 3)

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
        object_R, object_t = fit_motion_svd_batch(pc1_rep, pc1_rep + flow_rep, mask1_rep)
        # Apply the estimated rigid transformation onto point cloud
        pc1_transformed = torch.einsum('bij,bnj->bni', object_R, pc1_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc1_transformed = pc1_transformed.reshape(n_batch, n_object, n_point, 3)
        flow = torch.einsum('bkn,bkni->bni', mask1, pc1_transformed) - pc1
    return flow

def mask_batchwise_chamfer(pc1, pc2, mask1, mask2, loss_norm=2):
    """
    :param pc1: (B, N, 3) torch.Tensor.
    :param pc2: (B, N, 3) torch.Tensor.
    :param flow: (B, N, 3) torch.Tensor.
    :return:
        loss: () torch.Tensor.
    """
    pc2 = pc2.contiguous()
    pc2_t = pc2.transpose(1, 2).contiguous()
    pc1 = pc1.contiguous() #pc1 = (pc1 + flow).contiguous()
    pc1_t = pc1.transpose(1, 2).contiguous()
    _, idx = knn(1, pc1, pc2)
    nn1 = grouping_operation(pc2_t, idx.detach()).squeeze(-1)
    w_nn1 = grouping_operation(mask2.unsqueeze(1).contiguous(), idx.detach()).squeeze(-1).squeeze(1)
    dist1 = mask1 * (pc1_t - nn1).norm(p=loss_norm, dim=1) * w_nn1
    _, idx = knn(1, pc2, pc1)
    nn2 = grouping_operation(pc1_t, idx.detach()).squeeze(-1)
    w_nn2 = grouping_operation(mask1.unsqueeze(1).contiguous(), idx.detach()).squeeze(-1).squeeze(1)
    dist2 = mask2 * (pc2_t - nn2).norm(p=loss_norm, dim=1) * w_nn2
    loss = (dist1.mean(-1) + dist2.mean(-1)) / 2
    return loss, nn1, w_nn1, nn2, w_nn2

def fit_motion_svd_batch_with_R_all(pc1, pc2, mask, raw_pc2, raw_mask2, init_R_all,
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
        residual_pointwise = mask.unsqueeze(-1) * (pc2 - pc1_transformed)
        residual_pointwise = torch.square(residual_pointwise).sum(-1)
        mask = mask.clone()
        mask[residual_min_sum.squeeze(-1) * n_point < res_ignore] /= torch.exp(exp_factor * residual_pointwise[residual_min_sum.squeeze(-1) * n_point < res_ignore])
        thr = torch.topk(residual_pointwise, k_removal, dim=1)[0].min(dim=1, keepdim=True)[0] #torch.median(residual, dim=1, keepdim=True)[0]
        mask[torch.logical_and(residual_pointwise > thr, residual_min_sum * n_point < res_ignore)] = 0

    if mask is None:
        pc1_mean = torch.mean(pc1, dim=1, keepdim=True)   # (B, 1, 3)
        pc2_mean = torch.mean(pc2, dim=1, keepdim=True)   # (B, 1, 3)
    else:
        pc1_mean = torch.einsum('bnd,bn->bd', pc1, mask) / torch.sum(mask, dim=1, keepdim=True)   # (B, 3)
        pc1_mean.unsqueeze_(1)
        pc2_mean = torch.einsum('bnd,bn->bd', pc2, mask) / torch.sum(mask, dim=1, keepdim=True)
        pc2_mean.unsqueeze_(1)

    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    if mask is None:
        S = torch.bmm(pc1_centered.transpose(1, 2), pc2_centered)
    else:
        S = pc1_centered.transpose(1, 2).bmm(torch.diag_embed(mask).bmm(pc2_centered))

    # If mask is not well-defined, S will be ill-posed.
    # We just return an identity matrix.
    valid_batches = ~torch.isnan(S).any(dim=1).any(dim=1)
    R_base = torch.eye(3, device=pc1.device).unsqueeze(0).repeat(n_batch, 1, 1)
    t_base = torch.zeros((n_batch, 3), device=pc1.device)

    if valid_batches.any():
        S = S[valid_batches, ...]
        u, s, v = torch.svd(S, some=False, compute_uv=True)
        R = torch.bmm(v, u.transpose(1, 2))
        det = torch.det(R)

        # Correct reflection matrix to rotation matrix
        diag = torch.ones_like(S[..., 0], requires_grad=False)
        diag[:, 2] = det
        R = v.bmm(torch.diag_embed(diag).bmm(u.transpose(1, 2)))

        pc1_mean, pc2_mean = pc1_mean[valid_batches], pc2_mean[valid_batches]
        t = pc2_mean.squeeze(1) - torch.bmm(R, pc1_mean.transpose(1, 2)).squeeze(2)
        R_base[valid_batches] = R #torch.matmul(R, init_R[valid_batches])
        t_base[valid_batches] = t #t + torch.matmul(R, init_t[valid_batches].unsqueeze(2)).squeeze(2)

    return R_base, t_base

def icp_based_on_R_all(pc1, pc2, flow, mask1, mask2, icp_iter=10, temperature=0.01, R_all=None,
                       exp_factor=0.0, k_removal=10, res_ignore=987654321.0):
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
        if iter == 0:
            object_R, object_t = fit_motion_svd_batch_with_R_all(pc1_rep, pc1_rep + flow_rep, mask1_rep,
                                                                 pc2_rep, mask2_rep, R_all,
                                                                 exp_factor=exp_factor,
                                                                 k_removal=k_removal, res_ignore=res_ignore)
        else:
            object_R, object_t = fit_motion_svd_batch(pc1_rep, pc1_rep + flow_rep, mask1_rep)
        # Apply the estimated rigid transformation onto point cloud
        pc1_transformed = torch.einsum('bij,bnj->bni', object_R, pc1_rep) + object_t.unsqueeze(1).repeat(1, n_point, 1)
        pc1_transformed = pc1_transformed.reshape(n_batch, n_object, n_point, 3)
        flow = torch.einsum('bkn,bkni->bni', mask1, pc1_transformed) - pc1
    return flow

def eval():
    # Iterate over samples
    eval_meter = AverageMeter()
    eval_meter_kabsch = AverageMeter()
    eval_meter_oaicp = AverageMeter()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader_predflow = DataLoader(test_set_predflow, batch_size=batch_size, shuffle=False, pin_memory=True,
                                      num_workers=4)
    with tqdm.tqdm(enumerate(zip(test_loader, test_loader_predflow), 0), total=len(test_loader), desc='test') as tbar:
        for i, (batch1, batch2) in tbar:
            pcs, _, flows, _ = batch1
            _, _, flow_preds, _ = batch2
            pc1, pc2 = pcs[:, 0].contiguous(), pcs[:, 1].contiguous()
            flow, flow_pred = flows[:, 0].contiguous(), flow_preds[:, 0].contiguous()

            # Forward inference: segmentation
            pc1, pc2, flow_pred = pc1.cuda(), pc2.cuda(), flow_pred.cuda()
            with torch.no_grad():
                mask1, _, _, _, _ = segnet(pc1)  # .detach()
                mask2, _, _, _, _ = segnet(pc2)  # .detach()
                n_points = pcs.shape[2]
                input_pcs = pcs.view(pcs.shape[0] * pcs.shape[1], pcs.shape[2], 3).cuda()
                mask_all, _, _, _, _ = segnet(input_pcs)


            mask1 = mask1.detach()
            mask2 = mask2.detach()

            # Upadate flow predictions using Weighted Kabsch
            flow_pred_kabsch = weighted_kabsch(pc1, flow_pred, mask1.detach())
            # Upadate flow predictions using OA-ICP
            flow_pred_oaicp = object_aware_icp(pc1, pc2, flow_pred, mask1, mask2, icp_iter=icp_iter)

            # Monitor the change of flow accuracy
            epe, acc_strict, acc_relax, outlier = eval_flow(flow, flow_pred,
                                                            epe_norm_thresh=epe_norm_thresh)
            eval_meter.append_loss(
                {'EPE': epe, 'AccS': acc_strict, 'AccR': acc_relax, 'Outlier': outlier})
            epe_r, acc_strict_r, acc_relax_r, outlier_r = eval_flow(flow, flow_pred_kabsch,
                                                                    epe_norm_thresh=epe_norm_thresh)
            eval_meter_kabsch.append_loss(
                {'EPE': epe_r, 'AccS': acc_strict_r, 'AccR': acc_relax_r, 'Outlier': outlier_r})
            epe_update, acc_strict_update, acc_relax_update, outlier_update = eval_flow(flow, flow_pred_oaicp,
                                                                                        epe_norm_thresh=epe_norm_thresh)
            eval_meter_oaicp.append_loss(
                {'EPE': epe_update, 'AccS': acc_strict_update, 'AccR': acc_relax_update, 'Outlier': outlier_update})

            # Save
            if args.save:
                test_set._save_predflow(flow_pred_oaicp, save_root=SAVE_DIR, batch_size=batch_size, n_frame=n_frame, offset=i)
    eval_avg = eval_meter.get_mean_loss_dict()
    print('Original Flow:', eval_avg)
    eval_avg_kabsch = eval_meter_kabsch.get_mean_loss_dict()
    print('Weighted Kabsch Flow:', eval_avg_kabsch)
    eval_avg_oaicp = eval_meter_oaicp.get_mean_loss_dict()
    print('Multi-body SE(3) Flow:', eval_avg_oaicp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--round', type=int, default=0, help='Which round of iterative optimization')
    parser.add_argument('--test_batch_size', type=int, default=6, help='Batch size in testing')
    parser.add_argument('--save', dest='save', default=False, action='store_true', help='Save flow predictions or not')
    parser.add_argument('--saveflow_path', type=str, default=None, help='Path to save flow predictions')

    # Read parameters
    args = parser.parse_args()
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    for ckey, cvalue in configs.items():
        args.__dict__[ckey] = cvalue

    # Configuration for different dataset
    data_root = args.data['root']
    if args.dataset == 'sapien':
        from models.eq_2head_sapien import MaskFormer3D
        from datasets.dataset_sapien import SapienDataset as TestDataset
        if args.split == 'test':
            data_root = osp.join(data_root, 'mbs-sapien')
        else:
            data_root = osp.join(data_root, 'mbs-shapepart')
        epe_norm_thresh = 0.01
    else:
        raise KeyError('Unrecognized dataset!')

    # Setup the segmentation network
    segnet = MaskFormer3D(n_slot=args.segnet['n_slot'],
                          n_point=args.segnet['n_point'],
                          use_xyz=args.segnet['use_xyz'],
                          n_transformer_layer=args.segnet['n_transformer_layer'],
                          transformer_embed_dim=args.segnet['transformer_embed_dim'],
                          transformer_input_pos_enc=args.segnet['transformer_input_pos_enc']).cuda()

    # Load the trained model weights
    weight_path = osp.join(args.save_path + '_R%d'%(args.round), 'best.pth.tar')
    segnet.load_state_dict(torch.load(weight_path)['model_state'])
    segnet.cuda().eval()
    print('SegNet loaded from', weight_path)
    segnet = torch.nn.DataParallel(segnet)
    # Setup the dataset
    if args.round > 1:
        predflow_path = 'flowstep3d_R%d' % (args.round - 1)
    else:
        predflow_path = 'flowstep3d' if args.dataset != 'sapien' else 'all_frames_unsup_flowstep3d'
    if args.dataset in ['sapien', 'ogcdr']:
        view_sels = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]] if args.dataset != 'sapien' else\
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 0], [2, 1], [3, 1], [3, 2]]
        test_set = TestDataset(data_root=data_root,
                               split=args.split,
                               view_sels=view_sels,
                               decentralize=args.data['decentralize'])
        test_set_predflow = TestDataset(data_root=data_root,
                                        split=args.split,
                                        view_sels=view_sels,
                                        predflow_path=predflow_path,
                                        decentralize=args.data['decentralize'])

    else:
        raise KeyError('Unrecognized dataset!')

    n_frame = len(view_sels)
    batch_size = args.test_batch_size

    # Hyperparam for Object-Aware ICP
    icp_iters = {0: 20, 1: 20, 2: 10, 3: 5, 4: 3}
    icp_iter = icp_iters[args.round]

    # Save updated flow predictions
    if args.save:
        assert batch_size % n_frame == 0, \
            'Frame pairs of one scene should be in the same batch, otherwise very inconvenient for saving!'
        # Path to save flow predictions
        if args.saveflow_path is None:
            args.saveflow_path = 'flowstep3d'
        SAVE_DIR = osp.join(data_root, 'flow_preds', args.saveflow_path + '_R%d'%(args.round))
        os.makedirs(SAVE_DIR, exist_ok=True)
        # Write information about "view_sel" into a meta file
        if args.dataset in ['sapien', 'ogcdr']:
            SAVE_META = SAVE_DIR + '.json'
            with open(SAVE_META, 'w') as f:
                json.dump({'view_sel': view_sels}, f)
    eval()