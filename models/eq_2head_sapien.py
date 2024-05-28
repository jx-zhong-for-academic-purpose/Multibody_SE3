import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn_util import Seq
from utils.pointnet2_util import PointnetFPModule, PointnetSAModuleMSG, PointnetSAModule
from utils.transformer_util import MaskFormerHead
from EPN_PointCloud.SPConvNets.models import eq_2head_so3net

BN_CONFIG = {"class": "GroupNorm", "num_groups": 4}


class MaskFormer3D(nn.Module):
    """
    A 3D object segmentation network, combing PointNet++ and MaskFormer.
    """
    def __init__(self,
                 n_slot,
                 n_point=512,
                 use_xyz=True,
                 bn=BN_CONFIG,
                 n_transformer_layer=2,
                 transformer_embed_dim=256,
                 transformer_input_pos_enc=False):
        super().__init__()

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 64], bn=bn))
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 64, 256, 128], bn=bn))#[256 + 128, 256, 128], bn=bn))

        # MaskFormer head
        self.MF_head = MaskFormerHead(
            n_slot=n_slot, input_dim=128,#256,
            n_transformer_layer=n_transformer_layer,
            transformer_embed_dim=transformer_embed_dim, transformer_n_head=8,
            transformer_hidden_dim=transformer_embed_dim, input_pos_enc=transformer_input_pos_enc
        )
        self.object_mlp = Seq(transformer_embed_dim).conv1d(transformer_embed_dim, bn=bn).conv1d(64, activation=None)

        neighbor_bound = 48
        na = 60
        xyz_pooling = None
        input_radius = 1.0
        initial_radius_ratio = 0.2
        sigma_ratio = 0.5
        sampling_ratio = 0.8
        sampling_density = 0.5
        dropout_rate = 0
        input_num = 512
        eq_enc_params = {'name': 'Invariant ZPConv Model', 'backbone': [], 'na': na}
        dim_in = 1
        kernel_multiplier = 2
        mlps = [[32, 32, 64], [64, 64, 128]] #[[64, 64, 128], [128, 128, 256]]
        strides = [2, 2]
        # process args
        n_layer = len(mlps)
        stride_current = 1
        stride_multipliers = [stride_current]
        for i in range(n_layer):
            stride_current *= 2
            stride_multipliers += [stride_current]

        num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

        radius_ratio = [initial_radius_ratio * multiplier ** sampling_density for multiplier in stride_multipliers]

        radii = [r * input_radius for r in radius_ratio]

        weighted_sigma = [sigma_ratio * radii[0] ** 2]
        for idx, s in enumerate(strides):
            weighted_sigma.append(weighted_sigma[idx] * s)

        for i, block in enumerate(mlps):
            block_param = []
            for j, dim_out in enumerate(block):
                lazy_sample = i != 0 or j != 0

                stride_conv = i == 0 or xyz_pooling != 'stride'

                neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i] ** (1 / sampling_density))
                neighbor = neighbor_bound if neighbor < neighbor_bound else neighbor

                if i == 0 and j == 0:
                    neighbor *= int(input_num / 512)

                kernel_size = 1
                if j == 0:
                    inter_stride = strides[i]
                    nidx = i if i == 0 else i + 1
                    if stride_conv:
                        neighbor *= 2
                        kernel_size = 1
                else:
                    inter_stride = 1
                    nidx = i + 1

                print(f"At block {i}, layer {j}!")
                print(f'neighbor: {neighbor}')
                print(f'stride: {inter_stride}')

                sigma_to_print = weighted_sigma[nidx] ** 2 / 3
                print(f'sigma: {sigma_to_print}')
                print(f'radius ratio: {radius_ratio[nidx]}')
                print(f'radius: {radii[nidx]}')

                block_type = 'inter_block' if na != 60 else 'separable_block'

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

            eq_enc_params['backbone'].append(block_param)

        out_mlps = [128, 64, 32]
        representation = 'quat'
        temperature = 3
        eq_enc_params['outblock'] = {
            'dim_in': dim_in,
            'mlp': out_mlps,
            'fc': [64],
            'representation': representation,
            'temperature': temperature,
        }
        self.eq_enc = eq_2head_so3net.InvSO3ConvModel(eq_enc_params)

    def forward(self, pc, input_segm=None, object_rot=None):
        """
        :param pc: (B, N, 3) torch.Tensor.
        :param point_feats: (B, N, 3) torch.Tensor.
        :return:
            mask: (B, N, K) torch.Tensor.
        """
        l_pc, l_feats = [pc], [pc.transpose(1, 2).contiguous()]
        x_list, inv_feat_list, att_list, confidence, quats, rot_loss_list = self.eq_enc(pc, input_segm, object_rot)
        for x, inv_feat in zip(x_list, inv_feat_list):
            l_pc.append(x.xyz.transpose(1, 2))
            l_feats.append(inv_feat)
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_feats[i - 1] = self.FP_modules[i](
                l_pc[i - 1], l_pc[i], l_feats[i - 1], l_feats[i]
            )

        slot = self.MF_head(l_feats[-1].transpose(1, 2), l_pc[-1])     # (B, K, D)
        slot = self.object_mlp(slot.transpose(1, 2))      # (B, D, K)

        # Obtain mask by dot-product
        mask = torch.einsum('bdn,bdk->bnk',
                            F.normalize(l_feats[0], dim=1),
                            F.normalize(slot, dim=1)) / 0.05
        mask = mask.softmax(dim=-1)
        return mask, att_list, confidence, quats, rot_loss_list


# Test the network implementation
if __name__ == '__main__':
    segnet = MaskFormer3D(n_slot=8,
                          use_xyz=True,
                          n_transformer_layer=2,
                          transformer_embed_dim=128,
                          transformer_input_pos_enc=False).cuda()
    pc = torch.randn(size=(4, 512, 3)).cuda()
    point_feats = torch.randn(size=(4, 512, 3)).cuda()
    mask = segnet(pc, point_feats)
    print (mask.shape)

    print('Number of parameters:', sum(p.numel() for p in segnet.parameters() if p.requires_grad))
    print('Number of parameters in PointNet++ encoder:', sum(p.numel() for p in segnet.SA_modules.parameters() if p.requires_grad))
    print('Number of parameters in PointNet++ decoder:', sum(p.numel() for p in segnet.FP_modules.parameters() if p.requires_grad))
    print('Number of parameters in MaskFormer head:', sum(p.numel() for p in segnet.MF_head.parameters() if p.requires_grad))

    print(segnet)