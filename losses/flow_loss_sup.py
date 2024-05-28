import torch
from torch.nn import Module, MSELoss, L1Loss

from losses.flow_loss_unsup import SmoothLoss


class SupervisedL1Loss(Module):
    def __init__(self, **kwargs):
        super(SupervisedL1Loss, self).__init__()
        self.l1_loss = L1Loss()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(pred_flow, gt_flow)


class SupervisedL2Loss(Module):
    def __init__(self, **kwargs):
        super(SupervisedL2Loss, self).__init__()
        self.l2_loss = MSELoss()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        return self.l2_loss(pred_flow, gt_flow)


class SupervisedL1RegLoss(Module):
    def __init__(self, w_data, w_smoothness, smoothness_loss_params, **kwargs):
        super(SupervisedL1RegLoss, self).__init__()
        self.data_loss = L1Loss()
        self.smoothness_loss = SmoothLoss(**smoothness_loss_params)
        self.w_data = w_data
        self.w_smoothness = w_smoothness

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor, i=0) -> torch.Tensor:
        if len(self.w_data) == 1:
            w_data = self.w_data[0]
            w_smoothness = self.w_smoothness[0]
        else:
            w_data = self.w_data[i]
            w_smoothness = self.w_smoothness[i]

        loss = (w_data * self.data_loss(pred_flow, gt_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow))
        return loss


class SupervisedFlowStep3DLoss(Module):
    def __init__(self, smooth_loss, weights=[0.75, 0.25], iters_w=[1.0]):
        super().__init__()
        self.data_loss = L1Loss()
        self.smooth_loss = smooth_loss
        self.w_chamfer, self.w_smooth = weights
        self.iters_w = iters_w

    def forward(self, pc1, pc2, flow_preds, flow_gts):
        """
        :param pc1 & pc2: (B, N, 3) torch.Tensor.
        :param flow_preds: [(B, N ,3), ...], list of torch.Tensor.
        """
        assert len(flow_preds) == len(self.iters_w)

        loss_dict = {}
        loss_arr = []
        for i in range(len(flow_preds)):
            flow_pred = flow_preds[i]
            chamfer_loss_i = self.data_loss(flow_pred, flow_gts) #self.chamfer_loss(pc1, pc2, flow_pred)
            loss_dict['data_loss_#%d'%(i)] = chamfer_loss_i.item()
            smooth_loss_i = self.smooth_loss(pc1, flow_pred)
            loss_dict['smooth_loss_#%d'%(i)] = smooth_loss_i.item()
            loss_i = self.w_chamfer * chamfer_loss_i + self.w_smooth * smooth_loss_i
            loss_arr.append(self.iters_w[i] * loss_i)

        loss = sum(loss_arr)
        loss_dict['sum'] = loss.item()
        return loss, loss_dict