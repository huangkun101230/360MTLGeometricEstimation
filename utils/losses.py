import torch
import torch.nn as nn
import torchvision
import numpy as np

class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, mask=None):
        loss = (y_pred - y_true)**2
        if mask is not None:
            count = torch.sum(mask)
            masked_loss = loss * mask
            return torch.sum(masked_loss) / count
        return torch.mean(loss)

class quaternion_loss(nn.Module):
    def __init__(self):
        super(quaternion_loss, self).__init__()

    def forward(self, input, target, mask):
        q_pred = -input
        loss_x = target[:, 1, :, :] * q_pred[:, 2, :, :] - target[:, 2, :, :] * q_pred[:, 1, :, :]
        loss_y = target[:, 2, :, :] * q_pred[:, 0, :, :] - target[:, 0, :, :] * q_pred[:, 2, :, :]
        loss_z = target[:, 0, :, :] * q_pred[:, 1, :, :] - target[:, 1, :, :] * q_pred[:, 0, :, :]
        loss_re = -target[:, 0, :, :] * q_pred[:, 0, :, :] - target[:, 1, :, :] * q_pred[:, 1, :, :] - target[:, 2, :, :] * q_pred[:, 2, :, :]
        loss_x = loss_x.unsqueeze(1)
        loss_y = loss_y.unsqueeze(1)
        loss_z = loss_z.unsqueeze(1)
        
        dot = loss_x * loss_x + loss_y * loss_y + loss_z * loss_z
        eps = torch.ones_like(dot) * 1e-8

        vec_diff = torch.sqrt(torch.max(dot, eps))
        real_diff = torch.sign(loss_re) * torch.abs(loss_re)
        real_diff = real_diff.unsqueeze(1)
        
        loss = torch.atan2(vec_diff, real_diff) / (np.pi)
        
        if mask is not None:
            count = torch.sum(mask)
            mask = mask[:, 0, :, :].unsqueeze(1)
            masked_loss = loss * mask
            return torch.sum(masked_loss) / count
        return torch.mean(loss)
    
class NormalDegreeMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        prediction_error = torch.cosine_similarity(pred, gt, dim=1)
        E = 1 - torch.clamp(prediction_error, min=-1.0, max=1.0)
        mse_loss = torch.mean(E ** 2)  # Calculate the mean squared error
        return mse_loss
    
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss