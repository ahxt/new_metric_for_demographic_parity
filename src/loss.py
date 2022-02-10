import torch


class gap_reg(torch.nn.Module):
    def __init__(self, mode = "dp"):
        super(gap_reg, self).__init__()
        self.mode = mode

    def forward(self, y_pred, s, y_gt):
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]
        reg_loss = torch.abs(torch.mean(y0) - torch.mean(y1))
        return reg_loss, torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0])