import os
import os.path as osp
import pickle
import time

import tqdm
import yaml
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from metrics.seg_metric import accumulate_eval_results, calculate_AP, calculate_PQ_F1, ClusteringMetrics
from utils.pytorch_util import AverageMeter

# pc: (N, 3) numpy array
# segm: (N,) numpy array


# Borrowed from PointGroup, with order adjusted
COLOR20 = np.array(
    [[245, 130,  48], [  0, 130, 200], [ 60, 180,  75], [255, 225,  25], [145,  30, 180],
     [250, 190, 190], [230, 190, 255], [210, 245,  60], [240,  50, 230], [ 70, 240, 240],
     [  0, 128, 128], [230,  25,  75], [170, 110,  40], [255, 250, 200], [128,   0,   0],
     [170, 255, 195], [128, 128,   0], [255, 215, 180], [  0,   0, 128], [128, 128, 128]])
COLORGRAY2 = np.array([128, 128, 128])

import open3d as o3d
def pc_segm_to_sphere(pc, segm=None, radius = 0.01, resolution=10,
                      with_background=False, default_color=COLORGRAY2):
    """
    Visualize point cloud as mesh balls. The color denotes hard segmentation.
    :param pc: (N, 3)
    :param segm: (N,)
    """
    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    n_point = pc.shape[0]
    if with_background:
        colors = np.concatenate((COLOR20[-1:], COLOR20[:-1]), axis=0)
    else:
        colors = COLOR20

    meshes = []
    for pid in range(n_point):
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        if segm is not None:
            mesh.paint_uniform_color(colors[segm[pid] % colors.shape[0]] / 255.)
        else:
            mesh.paint_uniform_color(default_color / 255.)
        mesh.translate(pc[pid])
        meshes.append(mesh)

    # Merge
    mesh = meshes[0]
    for i in range(1, len(meshes)):
        mesh += meshes[i]
    return mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config files')
    parser.add_argument('--split', type=str, help='Dataset split')
    parser.add_argument('--round', type=int, default=-1, help='Trained segmentation model of which round')
    parser.add_argument('--visualize', dest='visualize', default=False, action='store_true', help='Qualitative / Quantitative evaluation mode')
    parser.add_argument('--test_batch_size', type=int, default=4, help='Batch size in testing')
    parser.add_argument('--curate_by_object', type=int, default=0, help='Test on a curated dataset where the number of objects per scene is larger than this threshold')
    parser.add_argument('--save', dest='save', default=False, action='store_true', help='Save segmentation predictions or not')

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
    else:
        raise KeyError('Unrecognized dataset!')

    # Setup the network
    segnet = MaskFormer3D(n_slot=args.segnet['n_slot'],
                          n_point=args.segnet['n_point'],
                          use_xyz=args.segnet['use_xyz'],
                          n_transformer_layer=args.segnet['n_transformer_layer'],
                          transformer_embed_dim=args.segnet['transformer_embed_dim'],
                          transformer_input_pos_enc=args.segnet['transformer_input_pos_enc']).cuda()
    '''
    # Load the trained model weights
    if args.round > 0:
        weight_path = osp.join(args.save_path + '_R%d'%(args.round), 'current.pth.tar')
    else:
        weight_path = osp.join(args.save_path, 'current.pth.tar')
    '''
    if args.round >= 0:
        weight_path = osp.join(args.save_path + '_R%d'%(args.round), 'best.pth.tar')
    else:
        weight_path = osp.join(args.save_path, 'best.pth.tar')
    #'''
    segnet.load_state_dict(torch.load(weight_path)['model_state'])
    segnet.cuda().eval()
    print('Loaded weights from', weight_path)

    segnet = torch.nn.DataParallel(segnet)

    # Setup the dataset
    if args.dataset in ['sapien', 'ogcdr', 'dynlab']:
        view_sels = [[0, 1], [1, 2], [2, 3], [3, 2]]
        n_frame = len(view_sels)
        test_set = TestDataset(data_root=data_root,
                               split=args.split,
                               view_sels=view_sels,
                               decentralize=args.data['decentralize'])
        ignore_npoint_thresh = 0
    else:
        raise KeyError('Unrecognized dataset!')

    batch_size = args.test_batch_size

    # If test on curated dataset, ensure samples in a batch come from a single scene
    if args.curate_by_object > 0:
        batch_size = n_frame


    # Qualitative evaluation mode
    if args.visualize:
        import open3d as o3d
        from utils.visual_util import build_pointcloud

        if args.dataset in ['sapien', 'ogcdr', 'dynlab']:
            test_loader = DataLoader(test_set, batch_size=n_frame, shuffle=False, pin_memory=True, num_workers=1)
            h_interval = -1.5
            w_interval = 1.5
            with_background = False
        else:
            test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)
            w_interval = 50
            with_background = True
        mbs_eval = ClusteringMetrics(spec=[ClusteringMetrics.IOU, ClusteringMetrics.RI])
        i_sid2Iou_RI = {}
        with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
            for i, batch in tbar:
                pcs, segms, flows, _ = batch
                pc = pcs[:, 0].contiguous().cuda()
                segm = segms[:, 0].contiguous()     # Ground-truth segmentation

                input_segm = torch.ones_like(segm)  # Dummy
                # Forward inference
                with torch.no_grad():
                   mask, _, _, _, _ = segnet(pc)

                # mIoU & RI metrics
                for sid in range(segm.shape[0] // n_frame):
                    all_mask = mask[(n_frame * sid):(n_frame * (sid + 1))]
                    all_segm = segm[(n_frame * sid):(n_frame * (sid + 1))].long()
                    per_scan_mbs = mbs_eval(all_mask, all_segm, ignore_npoint_thresh=ignore_npoint_thresh)

                mask = mask.detach().cpu().numpy()
                segm_pred = mask.argmax(2)

                # Visualize
                pc = pc.detach().cpu().numpy()
                segm = segm.numpy()
                pcds = []

                if args.dataset in ['sapien', 'ogcdr', 'dynlab']:
                    for t in range(segm.shape[0]):
                        pcds.append(build_pointcloud(pc[t], segm[t], with_background=with_background).translate([t*w_interval, 0.0, 0.0]))
                        pcds.append(build_pointcloud(pc[t], segm_pred[t], with_background=with_background).translate([t*w_interval, h_interval, 0.0]))

                        file_name = './vis/%d_%d_' % (i, t) + str(np.mean(per_scan_mbs['iou'])) + '_' + \
                                    str(np.mean(per_scan_mbs['ri'])) + '.obj'
                        mesh = pc_segm_to_sphere(pc[t], segm_pred[t])
                        o3d.io.write_triangle_mesh(file_name, mesh)
                else:
                    pcds.append(build_pointcloud(pc[0], segm[0], with_background=with_background).translate([0.0, 0.0, 0.0]))
                    pcds.append(build_pointcloud(pc[0], segm_pred[0], with_background=with_background).translate([w_interval, 0.0, 0.0]))

    # Quantitative evaluation mode
    else:
        assert batch_size % n_frame == 0, \
            'Frames of one scene should be in the same batch, otherwise very inconvenient for evaluation!'
        # Save segmentation predictions
        if args.save:
            # Path to save segmentation predictions
            SAVE_DIR = osp.join(data_root, 'segm_preds/OGC' + '_R%d'%(args.round))
            os.makedirs(SAVE_DIR, exist_ok=True)
            print('Save segmentation predictions into', SAVE_DIR, '...')

        # Iterate over the dataset
        mbs_eval = ClusteringMetrics(spec=[ClusteringMetrics.IOU, ClusteringMetrics.RI])
        eval_meter = AverageMeter()
        ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

        with tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), desc='test') as tbar:
            for i, batch in tbar:
                pcs, segms, flows, _ = batch
                pc = pcs[:, 0].contiguous().cuda()
                segm = segms[:, 0].contiguous()     # Groundtruth segmentation

                # Curate the dataset if specified
                n_object = torch.unique(segm[0]).shape[0]
                if n_object <= args.curate_by_object:
                    continue
                # Forward inference
                with torch.no_grad():
                    mask, _, _, _, _ = segnet(pc)

                # Accumulate for AP, PQ, F1, Pre, Rec
                Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = accumulate_eval_results(segm, mask, ignore_npoint_thresh=ignore_npoint_thresh)
                ap_eval_meter['Pred_IoU'].append(Pred_IoU)
                ap_eval_meter['Pred_Matched'].append(Pred_Matched)
                ap_eval_meter['Confidence'].append(Confidence)
                ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)

                # mIoU & RI metrics
                for sid in range(segm.shape[0] // n_frame):
                    all_mask = mask[(n_frame * sid):(n_frame * (sid + 1))]
                    all_segm = segm[(n_frame * sid):(n_frame * (sid + 1))].long()
                    per_scan_mbs = mbs_eval(all_mask, all_segm, ignore_npoint_thresh=ignore_npoint_thresh)
                    eval_meter.append_loss({'Per-scan IoU': np.mean(per_scan_mbs['iou']),
                                            'Per-scan RI': np.mean(per_scan_mbs['ri'])})

                # Save
                if args.save:
                    test_set._save_predsegm(mask, save_root=SAVE_DIR, batch_size=batch_size, n_frame=n_frame, offset=i)

        # Evaluate
        print('Evaluation on %s-%s:'%(args.dataset, args.split))
        Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
        Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
        Confidence = np.concatenate(ap_eval_meter['Confidence'])
        N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
        AP = calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot='True')
        print('AveragePrecision@50:', AP)
        PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
        print('PanopticQuality@50:', PQ,
              '\nF1-score@50:', F1,
              '\nPrec@50:', Pre,
              '\nRecall@50:', Rec)
        eval_avg = eval_meter.get_mean_loss_dict()
        for e in eval_avg:
            print(e + ':', eval_avg[e])