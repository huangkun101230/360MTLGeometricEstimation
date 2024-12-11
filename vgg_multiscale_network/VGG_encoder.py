import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VGG16_Feat(nn.Module):
    def __init__(self, features = None):
        super(VGG16_Feat, self).__init__()
        # Encoder pre-trained features
        self.features = features
        # Rest blocks
        # Encoder
        self.enc_block0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )

        self.enc_block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True)
        )

        self.bottle = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size = 3, padding = 1, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )

        self.init_weights()

    def init_weights(self, init = 'xavier'):
        if init == 'xavier':
            init_func = torch.nn.init.xavier_normal_
        else:
            init_func = torch.nn.init.normal_
        
        if self.features is not None:
            print('vgg16 pretrained weights loaded')
            # pretrained weights initialization
            self.enc_block0._modules['0'].weight = self.features._modules['0'].weight
            self.enc_block0._modules['3'].weight = self.features._modules['2'].weight
            self.enc_block1._modules['1'].weight = self.features._modules['5'].weight
            self.enc_block1._modules['4'].weight = self.features._modules['7'].weight
        else:
            init_func(self.enc_block0._modules['0'].weight)
            init_func(self.enc_block0._modules['3'].weight)
            init_func(self.enc_block1._modules['1'].weight)
            init_func(self.enc_block1._modules['4'].weight)

    
    def forward(self, x):
        x0 = self.enc_block0(x)
        x1 = self.enc_block1(x0)
        b = self.bottle(x1)
        b = b.flatten(2).transpose(1, 2).contiguous()

        return b