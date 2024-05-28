import numpy as np
import torch
import torch.nn as nn
import json
from vgtk.pc import group_nd

from EPN_PointCloud.vgtk.vgtk import MultiTaskDetectionLoss
from EPN_PointCloud.vgtk.vgtk.functional import label_relative_rotation_np
from EPN_PointCloud.vgtk.vgtk.spconv import SphericalPointCloud

import EPN_PointCloud.SPConvNets.utils as M
import torch.nn.functional as F

# outblock for relative rotation regression
class PointwiseRelSO3OutBlockR(nn.Module):
    def __init__(self, params, pointnet_anchors):
        super(PointwiseRelSO3OutBlockR, self).__init__()

        c_in = params['dim_in']
        mlp = params['mlp']

        self.pointnet = PointnetSO3PointwiseConv(c_in, c_in, pointnet_anchors)
        c_in = c_in * 2

        self.linear = nn.ModuleList()

        self.temperature = params['temperature']
        rp = params['representation']

        if rp == 'quat':
            self.out_channel = 4
        elif rp == 'ortho6d':
            self.out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%rp)

        self.attention_layer = nn.Conv2d(mlp[-1], 1, (1,1))

        # out channel equals 4 for quaternion representation, 6 for ortho representation
        self.regressor_layer = nn.Conv2d(mlp[-1],self.out_channel,(1,1))

        # ------------------ uniary conv ----------------
        for c in mlp:
            self.linear.append(nn.Conv2d(c_in, c, (1,1)))
            c_in = c


    def forward(self, f1, f2, x1, x2, seg1, seg2):
        # nb, nc, np, na -> nb, nc, na
        sp1 = SphericalPointCloud(x1, f1, None)
        sp2 = SphericalPointCloud(x2, f2, None)

        f1 = self._pooling(sp1, seg1)
        f1 = f1.view(f1.shape[0] * f1.shape[1], f1.shape[2], f1.shape[3])
        f2 = self._pooling(sp2, seg2)
        f2 = f2.view(f2.shape[0] * f2.shape[1], f2.shape[2], f2.shape[3])

        nb = f1.shape[0]
        na = f1.shape[2]

        # expand and concat into metric space (nb, nc*2, na_tgt, na_src)
        f2_expand = f2.unsqueeze(-1).expand(-1,-1,-1,na).contiguous()
        f1_expand = f1.unsqueeze(-2).expand(-1,-1,na,-1).contiguous()
        x_out = torch.cat((f1_expand,f2_expand),1)

        # fc layers with relu
        for linear in self.linear:
            x_out = linear(x_out)
            x_out = F.relu(x_out)

        attention_wts = self.attention_layer(x_out).view(nb, na, na)
        confidence = F.softmax(attention_wts * self.temperature, dim=1)
        y = self.regressor_layer(x_out)
        confidence = confidence.view(seg1.shape[0], seg1.shape[1], confidence.shape[1], confidence.shape[2])
        y = y.view(seg1.shape[0], seg1.shape[1], y.shape[1], y.shape[2], y.shape[3])
        # return: [nb, na, na], [nb, n_out, na, na]
        return confidence, y


    def _pooling(self, x, segm, kmax=1):
        # [nb, nc, na]
        x_out = self.pointnet(x)
        x_out = F.relu(x_out)

        x_out = x_out.unsqueeze(1)
        segm = segm.unsqueeze(2).unsqueeze(-1)
        ret = x_out * segm
        return ret.topk(kmax, 3)[0].sum(3)

class PointnetSO3PointwiseConv(nn.Module):
    '''
    equivariant pointnet architecture for a better aggregation of spatial point features
    f (nb, nc, np, na) x xyz (nb, 3, np, na) -> maxpool(h(nb,nc+3,p0,na),h(nb,nc+3,p1,na),h(nb,nc+3,p2,na),...)
    '''
    def __init__(self, dim_in, dim_out, anchor):
        super(PointnetSO3PointwiseConv, self).__init__()

        # get so3 anchors (60x3x3 rotation matrices)
        anchors = anchor
        self.dim_in = dim_in + 3
        self.dim_out = dim_out

        self.embed = nn.Conv2d(self.dim_in, self.dim_out,1)
        self.register_buffer('anchors', anchors)#torch.from_numpy(anchors))

    def forward(self, x):
        xyz = x.xyz
        feats = x.feats
        nb, nc, np, na = feats.shape

        # normalize xyz
        xyz = xyz - xyz.mean(2,keepdim=True)

        if na == 1:
            feats = torch.cat([x.feats, xyz[...,None]],1)
        else:
            xyzr = torch.einsum('aji,bjn->bina',self.anchors,xyz)
            feats = torch.cat([x.feats, xyzr],1)

        feats = self.embed(feats)
        return feats # nb, nc, np, na

class InvSO3ConvModel(nn.Module):
    def __init__(self, params):
        super(InvSO3ConvModel, self).__init__()

        self.backbone = nn.ModuleList()
        self.attention_layer_list = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))
            last_feat_dim = block_param[-1]['args']['dim_out']
            self.attention_layer_list.append(PointnetSO3PointwiseConv(last_feat_dim, 1, self.get_anchor()))
        self.na_in = params['na']
        self.invariance = True
        self.outblock = PointwiseRelSO3OutBlockR(params['outblock'], self.get_anchor())
        self.rot_loss = MultiTaskDetectionLoss(self.get_anchor(), w=1,
                                               nr=4 if params['outblock']['representation'] == 'quat' else 6)


    def forward(self, x, input_segm, object_rot):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = M.preprocess_input(x, self.na_in, False)
        if input_segm is not None:
            input_segm = input_segm.transpose(1, 2).contiguous()
        x_list, inv_feat_list, att_list = [], [], []
        for block_i, block in enumerate(self.backbone):
            x, sample_idx_list = block(x)
            x_list.append(x)
            for idx in sample_idx_list:
                if input_segm is not None:
                    input_segm = input_segm if idx is None else\
                        group_nd(input_segm.to(device=x.xyz.device).float(), idx)
                #print('input_segm', input_segm.shape)

            attention = self.attention_layer_list[block_i](x).squeeze(1)

            attention = attention.softmax(-1) #self.attention_layer_list[block_i](x.feats)
            att_list.append(attention)
            inv_feat_list.append(torch.sum(x.feats * attention.unsqueeze(1), -1))

        confidence, quats, rot_loss_list = None, None, None
        if input_segm is not None:
            f = x.feats.view(x.feats.shape[0] // 2, 2, x.feats.shape[1], x.feats.shape[2], x.feats.shape[3])
            f1, f2 = f.split(1, 1)
            x_xyz = x.xyz.view(x.xyz.shape[0] // 2, 2, x.xyz.shape[1], x.xyz.shape[2])
            x1, x2 = x_xyz.split(1, 1)
            input_segm1, input_segm2 = input_segm.view(input_segm.shape[0] // 2, 2, input_segm.shape[1],
                                                       input_segm.shape[2]).split(1, 1)

            confidence, quats = self.outblock(f1.squeeze(1), f2.squeeze(1),
                                              x1.squeeze(1), x2.squeeze(1),
                                              input_segm1.squeeze(1), input_segm2.squeeze(1))

            if object_rot is not None:
                R = object_rot.view(object_rot.shape[0] * object_rot.shape[1],
                                    object_rot.shape[2], object_rot.shape[3])
                confidence = confidence.view(confidence.shape[0] * confidence.shape[1], confidence.shape[2],
                                             confidence.shape[3])
                quats = quats.view(quats.shape[0] * quats.shape[1], quats.shape[2], quats.shape[3], quats.shape[4])
                R0_list, R_label_list = [], []
                for r in R:
                    R0, R_label = label_relative_rotation_np(self.get_anchor().cpu().detach().numpy(),
                                                            r.cpu().detach().numpy())
                    R0_list.append(R0)
                    R_label_list.append(R_label)
                R0_list = np.array(R0_list)
                R_label_list = np.array(R_label_list)
                self.rot_loss.anchors = self.rot_loss.anchors.cuda()
                rot_loss, cls_loss, l2_loss, acc, error = self.rot_loss(confidence,
                                                                        torch.from_numpy(R_label_list).long().cuda(),
                                                                        quats, torch.from_numpy(R0_list).cuda(), R)
                rot_loss_list = [rot_loss, cls_loss, l2_loss, acc, error]
        return x_list, inv_feat_list, att_list, confidence, quats, rot_loss_list

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

# Full Version
def build_model(opt,
                mlps=[[32,32], [64,64], [128,128], [128,128]],
                out_mlps=[128, 64],
                strides=[2, 2, 2, 2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8, #0.4, 0.36
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio= 0.5, # 1e-3, 0.68
                xyz_pooling = None, # None, 'no-stride'
                to_file=None):
    print(opt)
    device = opt.device
    input_num= opt.model.input_num
    dropout_rate= opt.model.dropout_rate
    temperature= opt.train_loss.temperature
    so3_pooling =  opt.model.flag
    input_radius = opt.model.search_radius
    kpconv = opt.model.kpconv

    na = 1 if opt.model.kpconv else opt.model.kanchor

    # to accomodate different input_num
    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    print("[MODEL] USING RADIUS AT %f"%input_radius)
    params = {'name': 'Invariant ZPConv Model',
              'backbone': [],
              'na': na
              }
    dim_in = 1

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio]

    # Compute sigma
    # weighted_sigma = [sigma_ratio * radii[i]**2 * stride_multipliers[i] for i in range(n_layer + 1)]

    weighted_sigma = [sigma_ratio * radii[0]**2]
    for idx, s in enumerate(strides):
        weighted_sigma.append(weighted_sigma[idx] * s)

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != 'stride'

            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

            if i == 0 and j == 0:
                neighbor *= int(input_num / 1024)

            kernel_size = 1
            if j == 0:
                inter_stride = strides[i]
                nidx = i if i == 0 else i+1
                if stride_conv:
                    neighbor *= 2
                    kernel_size = 1
            else:
                inter_stride = 1
                nidx = i+1

            print(f"At block {i}, layer {j}!")
            print(f'neighbor: {neighbor}')
            print(f'stride: {inter_stride}')

            sigma_to_print = weighted_sigma[nidx]**2 / 3
            print(f'sigma: {sigma_to_print}')
            print(f'radius ratio: {radius_ratio[nidx]}')
            # import ipdb; ipdb.set_trace()

            # one-inter one-intra policy
            block_type = 'inter_block' if na != 60  else 'separable_block'

            conv_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': inter_stride,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na,
                }
            }
            block_param.append(conv_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    params['outblock'] = {
        'dim_in': dim_in,
        'mlp': out_mlps,
        'pooling': so3_pooling,
        'temperature': temperature,
        'kanchor': na,
    }


    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = InvSO3ConvModel(params).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)

# Ref: https://github.com/stat-ml/hist-loss
from torch import Tensor
def triangular_histogram_with_linear_slope(inputs: Tensor, t: Tensor, delta: float):
    """
    Function that calculates a histogram from an article
    [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)
    Args:
        input (Tensor): tensor that contains the data
        t (Tensor): tensor that contains the nodes of the histogram
        delta (float): step in histogram
    """
    inputs = inputs.view(-1)
    # first condition of the second equation of the paper
    x = inputs.unsqueeze(0) - t.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    a = torch.sum(x * m, dim=1) / (delta * len(inputs))

    # second condition of the second equation of the paper
    x = t.unsqueeze(0) - inputs.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    b = torch.sum(x * m, dim=0) / (delta * len(inputs))

    return torch.add(a, b)


def norm_min_max_distributuions(*distributuions: Tensor):
    max_ = max(torch.max(d.data) for d in distributuions)
    min_ = min(torch.min(d.data) for d in distributuions)

    norm_distributuions = ((d - min_) / (max_ - min_) for d in distributuions)
    return norm_distributuions


from abc import ABC, abstractmethod
class BaseHistLoss(nn.Module, ABC):
    """
    Base class for all Loss with histograms
    Args:
        bins (int, optional): .Default: `128`
        alpha (float, optional): parameter for regularization. Default: `0`
    Shape:
        - pos_input: set of positive points, (N, *)
        - neg_input: set of negative points, (M, *)
        - output: scalar
    """
    def __init__(self, bins: int = 128, alpha: float = 0):
        super(BaseHistLoss, self).__init__()
        self.bins = bins
        self._max_val = 1
        self._min_val = 0
        self.alpha = alpha
        self.delta = (self._max_val - self._min_val) / (bins - 1)
        self.t = torch.arange(self._min_val, self._max_val + self.delta, step=self.delta)

    def compute_histogram(self, inputs: Tensor) -> Tensor:
        return triangular_histogram_with_linear_slope(inputs, self.t, self.delta)

    @abstractmethod
    def forward(self, positive: Tensor, negative: Tensor):
        positive, negative = norm_min_max_distributuions(positive, negative)
        pass

    def std_loss(self, *inputs: Tensor):
        if self.alpha > 0:
            std_loss = self.alpha * sum(i.std() for i in inputs)
        else:
            # In order not to waste time compute unnecessary stds
            std_loss = 0
        return std_loss

class HistogramLoss(BaseHistLoss):
    """
    Histogram Loss
    Args:
        bins (int, optional): .Default: `10`
        min_val (float, optional): Default: `-1`
        max_val (float, optional): Default: `1`
        alpha (float, optional): parameter for regularization. Default: `0`
    Shape:
        - positive: set of positive points, (N, *)
        - negative: set of negative points, (M, *)
        - loss: scalar
    Examples::
        >>> criterion = HistogramLoss()
        >>> positive = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> negative = torch.sigmoid(torch.randn(10, requires_grad=True))
        >>> loss = criterion(positive, negative)
        >>> loss.backward()
    Reference:
        E. Ustinova and V. Lempitsky: Learning Deep Embeddings with Histogram Loss:
        https://arxiv.org/pdf/1611.00822.pdf
    """

    def forward(self, positive: Tensor, negative: Tensor):
        self.t = self.t.to(device=positive.device)
        positive, negative = norm_min_max_distributuions(positive, negative)

        pos_hist = self.compute_histogram(positive)  # h_pos
        neg_hist = self.compute_histogram(negative)  # h_neg
        pos_cum = torch.cumsum(pos_hist, 0)  # phi_pos

        hist_loss = (neg_hist * pos_cum).sum()  # 4 equation of the paper
        # Not in the article, own improvements
        std_loss = self.std_loss(positive, negative)

        loss = hist_loss + std_loss
        return loss


class ContinuousHistogramLoss(BaseHistLoss):
    """
    Histogram Loss
    Args:
        bins (int, optional): .Default: `10`
        alpha (float, optional): parameter for regularization. Default: `0`
    Shape:
        - distance: contain predicted distance from model, (N, *)
        - similarity: contain real distance from data, (N, *)
        - loss: scalar
    Examples::
        >>> criterion = ContinuousHistogramLoss()
        >>> distance = torch.rand(100, requires_grad=True)
        >>> similarity = torch.randint(low=0, high=5, size=(100,)).to(torch.float)
        >>> loss = criterion(distance, similarity)
        >>> loss.backward()
    Reference:
        CONTINUOUS HISTOGRAM LOSS: BEYOND NEURAL SIMILARITY
        https://arxiv.org/pdf/2004.02830v1.pdf
    """
    def __init__(self, bins: int = 128, bins_similarity: int = 3, alpha: float = 0):
        super(ContinuousHistogramLoss, self).__init__(bins=bins, alpha=alpha)

        # similarity
        if bins_similarity < 1:
            raise ValueError(
                f'Number of bins for similarity must be grather than 1: {bins_similarity}'
            )

        self.bins_similarity = bins_similarity
        self.delta_z = 1. / (bins_similarity - 1)
        self.dz = 0.5

    def forward(self, distance: Tensor, similarity: Tensor):
        self.t = self.t.to(device=distance.device)
        distance, = norm_min_max_distributuions(distance)

        hists = []
        std_loss = 0
        for i in range(self.bins_similarity + 1):
            mask = torch.abs(similarity / self.delta_z - i) <= self.dz
            if mask.sum() == 0:
                continue
            else:
                hist_i = self.compute_histogram(distance[mask])
                hists.append(hist_i)
                std_loss += self.std_loss(distance[mask])

        hists = torch.stack(hists)  # h_rz
        phi = self.inv_cumsum_with_shift(hists)  # phi_rz

        continuous_hist_loss = (hists * phi).sum()  # last equation of the paper
        loss = continuous_hist_loss + std_loss

        return loss

    @staticmethod
    def inv_cumsum_with_shift(t):
        """
            phi_{rz} = sum_{q=1}^r sum_{z'=z+1}^{R_z} h_{qz'}
        """
        flip_t = torch.flip(t, [0])
        flip_cumsum = torch.cumsum(torch.cumsum(flip_t, 1), 0)

        cumsum = torch.flip(flip_cumsum, [0])
        zero_raw = torch.zeros_like(cumsum[-1:])
        cumsum_with_shift = torch.cat([cumsum[1:], zero_raw])

        return cumsum_with_shift

if __name__ == '__main__':
    opt = None
    BS = 2
    N  = 1024
    C  = 3
    device = torch.device("cuda:0")
    x = torch.randn(BS, N, 3).to(device)
    opt.mode = 'train'
    opt.model.flag = 'rotation'
    print("Performing a regression task...")
    opt.model.model = 'inv_so3net'
    model = build_model_from(opt, outfile_path=None)
    out = model(x)
    print(out[0].shape, out[1].shape)
    print('Con!')
