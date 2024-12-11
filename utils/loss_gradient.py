import torch
import torch.nn as nn
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gradient_Loss(nn.Module):
  def __init__(self, device):
    super(Gradient_Loss, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, pred, gt):
    pred_grad_x = F.conv2d(pred, self.weight_x, padding=1)
    pred_grad_y = F.conv2d(pred, self.weight_y, padding=1)
    gt_grad_x = F.conv2d(gt, self.weight_x, padding=1)
    gt_grad_y = F.conv2d(gt, self.weight_y, padding=1)

    loss_x = torch.abs(torch.abs(pred_grad_x) - torch.abs(gt_grad_x)).mean()
    loss_y = torch.abs(torch.abs(pred_grad_y) - torch.abs(gt_grad_y)).mean()

    total_loss = loss_x + loss_y
    # gradient = torch.abs(grad_x) + torch.abs(grad_y)
    # return torch.abs(grad_x), torch.abs(grad_y)
    return total_loss