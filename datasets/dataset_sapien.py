import os
import os.path as osp
import json
import numpy as np
from torch.utils.data import Dataset

from utils.sapien_util import Isometry
from utils.data_util import compress_label_id, batch_segm_to_mask, augment_transform


def compute_flow(base_pc, base_segms, base_cam, base_motions, dest_cam, dest_motions):
    n_parts = len(base_motions)
    final_pc = np.empty_like(base_pc)
    for part_id in range(n_parts):
        part_mask = np.where(base_segms == (part_id + 1))[0]
        part_pc = (dest_cam.inv().dot(dest_motions[part_id]).dot(
            base_motions[part_id].inv()).dot(base_cam)) @ base_pc[part_mask]
        final_pc[part_mask] = part_pc
    return final_pc - base_pc


class SapienDataset(Dataset):
    def __init__(self,
                 data_root,
                 split='train',
                 view_sels=[[0, 1]],
                 predflow_path=None,
                 decentralize=False,
                 aug_transform=False,
                 aug_transform_args=None,
                 onehot_label=False,
                 max_n_object=8,
                 gt_flow=False, need_id=False):
        """
        :param data_root: root path containing `data' and `meta.json'.
        :param split: split to be loaded.
        :param view_sels: paired combinations of views to be used.
        :param predflow_path: path to load pre-saved flow predictions, otherwise use GT flows.
        :param decentralize: whether normalize point cloud to be centered at the origin.
        :param aug_transform: whether augment with spatial transformatons.
        :param aug_transform_args: a dict containing hyperparams for sampling spatial augmentations.
        :param onehot_label: whether convert the segmentation to one-hot encoding (only for fully-supervised training).
        :param max_n_object: predefined number of objects per scene that is large enough for the dataset, to be used in one-hot encoding.
        """
        self.data_root = osp.join(data_root, 'data')
        with open(osp.join(data_root, 'meta.json')) as f:
            self.meta = json.load(f)
        self.split = split
        self.data_ids = self.meta[split]
        self.view_sels = view_sels

        if predflow_path is not None:
            self.predflow_path = osp.join(data_root, 'flow_preds', predflow_path)
            pf_meta = self.predflow_path + '.json'
            with open(pf_meta, 'r') as f:
                self.pf_view_sels = json.load(f)['view_sel']
            # Check if flow predictions cover the specified "view_sel"
            if any([sel not in self.pf_view_sels for sel in view_sels]):
                raise ValueError('Flow predictions cannot cover specified view selections!')
            print('Load flow predictions from', self.predflow_path)
        else:
            self.predflow_path = None

        self.decentralize = decentralize
        self.aug_transform = aug_transform
        self.aug_transform_args = aug_transform_args
        self.onehot_label = onehot_label
        self.max_n_object = max_n_object
        self.gt_flow = gt_flow
        self.need_id = need_id

    def __len__(self):
        return len(self.data_ids) * len(self.view_sels)


    def _load_data(self, idx):
        data_path = osp.join(self.data_root, '%06d.npz'%(self.data_ids[idx]))
        data = np.load(data_path, allow_pickle=True)

        pc = data['pc'].astype(np.float32)
        segm = data['segm']
        trans = data['trans'].item()
        return pc, segm, trans


    def _load_predflow(self, idx):
        data_path = osp.join(self.predflow_path, '%06d.npy'%(self.data_ids[idx]))
        flow_pred = np.load(data_path)
        return flow_pred


    def __getitem__(self, sid):
        idx, view_sel_idx = sid // len(self.view_sels), sid % len(self.view_sels)
        pcs, segms, trans_dict = self._load_data(idx)
        n_parts = len(trans_dict) - 1
        view_sel = self.view_sels[view_sel_idx]

        def get_view_motions(view_id):
            return [Isometry.from_matrix(trans_dict[t][view_id]) for t in range(1, n_parts + 1)]

        # Extract two-frame point cloud, segmentation, and flow
        pcs, segms = pcs[view_sel], segms[view_sel]
        flows = []
        view_id1, view_id2 = view_sel
        gt_flows = []
        if self.predflow_path is not None:
            flow_pred = self._load_predflow(idx)
            flows.append(flow_pred[self.pf_view_sels.index([view_id1, view_id2])])
            flows.append(flow_pred[self.pf_view_sels.index([view_id2, view_id1])])
            if self.gt_flow:
                gt_flows.append(compute_flow(pcs[0], segms[0],
                                             Isometry.from_matrix(trans_dict['cam'][view_id1]), get_view_motions(view_id1),
                                             Isometry.from_matrix(trans_dict['cam'][view_id2]), get_view_motions(view_id2)))
                gt_flows.append(compute_flow(pcs[1], segms[1],
                                             Isometry.from_matrix(trans_dict['cam'][view_id2]), get_view_motions(view_id2),
                                             Isometry.from_matrix(trans_dict['cam'][view_id1]), get_view_motions(view_id1)))

        else:
            flows.append(compute_flow(pcs[0], segms[0],
                                      Isometry.from_matrix(trans_dict['cam'][view_id1]), get_view_motions(view_id1),
                                      Isometry.from_matrix(trans_dict['cam'][view_id2]), get_view_motions(view_id2)))
            flows.append(compute_flow(pcs[1], segms[1],
                                      Isometry.from_matrix(trans_dict['cam'][view_id2]), get_view_motions(view_id2),
                                      Isometry.from_matrix(trans_dict['cam'][view_id1]), get_view_motions(view_id1)))
        flows = np.stack(flows, 0)
        if self.gt_flow:
            gt_flows = np.stack(gt_flows, 0)
        # Normalize point cloud to be centered at the origin
        if self.decentralize:
            center = pcs.mean(1).mean(0)
            pcs = pcs - center

        # Compress the object-id in segmentation to consecutive numbers starting from 0
        segms = np.reshape(segms, -1)
        segms = compress_label_id(segms)
        segms = np.reshape(segms, (2, -1))

        # Convert the segmentation to one-hot encoding (only for fully-supervised training)
        if self.onehot_label:
            assert self.max_n_object > 0, 'max_n_object must be above 0!'
            segms, valids = batch_segm_to_mask(segms, self.max_n_object, ignore_npoint_thresh=0)
        else:
            valids = np.ones_like(segms, dtype=np.float32)

        # Augment the point cloud & flow with spatial transformations
        if self.aug_transform:
            pcs, flows = augment_transform(pcs, flows, self.aug_transform_args)
            if self.gt_flow:
                pcs, gt_flows = augment_transform(pcs, gt_flows, self.aug_transform_args)
            segms = np.concatenate((segms, segms), 0)
            valids = np.concatenate((valids, valids), 0)

        if self.gt_flow:
            valids = gt_flows

        if self.need_id:
            valids = np.array((idx, view_id1, view_id2))

        if self.onehot_label:
            return pcs.astype(np.float32), segms.astype(np.float32), flows.astype(np.float32), valids.astype(np.float32)
        else:
            return pcs.astype(np.float32), segms.astype(np.int32), flows.astype(np.float32), valids.astype(np.float32)


    def _save_predflow(self, flow_pred, save_root, batch_size, n_frame=1, offset=0):
        """
        :param flow_pred: (B, N, 3) torch.Tensor.
        """
        flow_pred = flow_pred.detach().cpu().numpy()
        for sid in range(flow_pred.shape[0] // n_frame):
            save_flow = flow_pred[sid * n_frame:(sid + 1) * n_frame]
            idx = offset * batch_size // n_frame + sid
            data_id = self.data_ids[idx]
            save_file = osp.join(save_root, '%06d.npy'%(data_id))
            np.save(save_file, save_flow)


    def _save_predsegm(self, mask, save_root, batch_size, n_frame=1, offset=0):
        """
        :param mask: (B, N, K) torch.Tensor.
        """
        mask = mask.detach().cpu().numpy()
        for sid in range(mask.shape[0]):
            segm_pred = mask[sid].argmax(1)
            idx, view_sel_idx = (offset * batch_size + sid) // n_frame, (offset * batch_size + sid) % n_frame
            data_id = self.data_ids[idx]
            save_path = os.path.join(save_root, '%06d'%(data_id))
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, 'segm_%02d.npy'%(view_sel_idx))
            np.save(save_file, segm_pred)


# Test the dataset loader
if __name__ == '__main__':
    split = 'train'
    data_root = '/home/jiaxing/data/2021/MultiBodySync/'
    if split == 'test':
        data_root = osp.join(data_root, 'mbs-sapien')
    else:
        data_root = osp.join(data_root, 'mbs-shapepart')
    view_sels = [[0, 1], [2, 3]]
    predflow_path = None

    decentralize = True
    aug_transform = True
    aug_transform_args = {
        'scale_low': 0.95,
        'scale_high': 1.05,
        'degree_range': [0, 180, 0],
        'shift_range': [0, 0, 0]
    }
    onehot_label = False
    max_n_object = 8
    dataset = SapienDataset(data_root=data_root,
                            split=split,
                            view_sels=view_sels,
                            predflow_path=predflow_path,
                            decentralize=decentralize,
                            aug_transform=aug_transform,
                            aug_transform_args=aug_transform_args,
                            onehot_label=onehot_label,
                            max_n_object=max_n_object)
    print (len(dataset))


    import open3d as o3d
    from utils.visual_util import build_pointcloud

    interval = 1.5
    segm_list = []
    min_p, max_p = [], []
    for sid in range(len(dataset)):
        pcs, segms, flows, _ = dataset[sid]
        min_p.append(pcs.min(0).min(0))
        max_p.append(pcs.max(0).max(0))
        segm_list.append(segms.min())
        continue
        pc1, pc2 = pcs[0], pcs[1]
        segm1, segm2 = segms[0], segms[1]
        if onehot_label:
            segm1, segm2 = segm1.argmax(1), segm2.argmax(1)

        pcds = []
        pcds.append(build_pointcloud(pc1, segm1))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))
        pcds.append(build_pointcloud(pc2, segm2).translate([interval, 0.0, 0.0]))
        pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[interval, 0, 0]))

        # Check spatial augmentations
        if aug_transform:
            pc3, pc4 = pcs[2], pcs[3]
            segm3, segm4 = segms[2], segms[3]
            if onehot_label:
                segm3, segm4 = segm3.argmax(1), segm4.argmax(1)
            pcds.append(build_pointcloud(pc3, segm3).translate([2 * interval, 0.0, 0.0]))
            pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[2 * interval, 0, 0]))
            pcds.append(build_pointcloud(pc4, segm4).translate([3 * interval, 0.0, 0.0]))
            pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[3 * interval, 0, 0]))

        # Check flows
        flow1, flow2 = flows[0], flows[1]
        pcds.append(build_pointcloud(pc1, np.zeros_like(segm1)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-2 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc1 + flow1, np.zeros_like(segm1)).translate([-1 * interval, 0.0, 0.0]))
        pcds.append(build_pointcloud(pc2, np.ones_like(segm2)).translate([-1 * interval, 0.0, 0.0]))