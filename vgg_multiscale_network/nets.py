import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .normal_decoder import NormalDecoder
from .depth_decoder import DepthDecoder
from .panoformer import Panoformer as encoder
from .panoformer import BasicPanoformerLayer
from .equisamplingpoint import genSamplingPattern
from utils.switchable_norm import SwitchNorm2d
from .VGG_encoder import VGG16_Feat



class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Downsample, self).__init__()
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out
    
class FusionAndDownsampleModule(nn.Module):
    def __init__(self, num_channels, input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv_normal_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            SwitchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.dowsample_normal = Downsample(num_channels, num_channels * 2, self.input_resolution)

        self.conv_depth_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            SwitchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.dowsample_depth = Downsample(num_channels, num_channels * 2, self.input_resolution)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(num_channels*2, num_channels, kernel_size=1),
            SwitchNorm2d(num_channels),
            nn.ReLU(),
        )
        self.dowsample_fusion = Downsample(num_channels, num_channels * 2, self.input_resolution)

    
    def forward(self, normal, depth):
        f_normal = normal
        f_depth = depth
        f_cat = torch.cat([f_normal, f_depth], dim=2)

        B, L, C = f_cat.shape
        H, W = self.input_resolution
        f_cat = f_cat.transpose(1, 2).contiguous().view(B, C, H, W)

        f_normal = f_normal + self.conv_normal_cat(f_cat).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        f_depth = f_depth + self.conv_depth_cat(f_cat).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        f_fusion = self.conv_cat(f_cat).flatten(2).transpose(1, 2).contiguous()  # B H*W C


        return f_normal, f_depth, f_fusion

class DeepNet(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False, **kwargs):        
        super(DeepNet, self).__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size

        self.ref_point8x16 = genSamplingPattern(8, 16, 3, 3).cuda()#torch.load("network6/Equioffset16x32.pth")
        # enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]


        orig_vgg = torchvision.models.vgg16(pretrained = True)
        features = orig_vgg.features
        self.vgg16_feature_extractor = VGG16_Feat(features)

        self.normal_encoder = encoder()
        self.depth_encoder = encoder()

        self.f0 = FusionAndDownsampleModule(embed_dim, input_resolution=(img_size, img_size * 2))
        self.f1 = FusionAndDownsampleModule(embed_dim * 2, input_resolution=(img_size // 2, img_size))
        self.f2 = FusionAndDownsampleModule(embed_dim * 4, input_resolution=(img_size // 4, img_size // 2))
        self.f3 = FusionAndDownsampleModule(embed_dim * 8, input_resolution=(img_size // 8, img_size // 4))

        self.bottleneck = BasicPanoformerLayer(dim=embed_dim * 16 *2,
                                      output_dim=embed_dim * 16 *2,
                                      input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer,ref_point=self.ref_point8x16, flag = 0)

        self.normal_decoder = NormalDecoder()
        self.depth_decoder = DepthDecoder()


    def forward(self, rgb_inputs):
        vgg_feat = self.vgg16_feature_extractor(rgb_inputs)

        f_normal0 = self.normal_encoder.encoderlayer_0(vgg_feat)
        f_depth0 = self.depth_encoder.encoderlayer_0(vgg_feat)

        f_normal0, f_depth0, fusion0 = self.f0(f_normal0, f_depth0)


        f_normal1 = self.normal_encoder.encoderlayer_1(self.f0.dowsample_normal(f_normal0))

        f_depth1 = self.depth_encoder.encoderlayer_1(self.f0.dowsample_depth(f_depth0))

        f_normal1, f_depth1, fusion1 = self.f1(f_normal1, f_depth1)

        f_normal2 = self.normal_encoder.encoderlayer_2(self.f1.dowsample_normal(f_normal1))
        f_depth2 = self.depth_encoder.encoderlayer_2(self.f1.dowsample_depth(f_depth1))
        f_normal2, f_depth2, fusion2 = self.f2(f_normal2, f_depth2)

        f_normal3 = self.normal_encoder.encoderlayer_3(self.f2.dowsample_normal(f_normal2))
        f_depth3 = self.depth_encoder.encoderlayer_3(self.f2.dowsample_depth(f_depth2))
        f_normal3, f_depth3, fusion3 = self.f3(f_normal3, f_depth3)

        btnec_feature = self.bottleneck(torch.cat([self.f3.dowsample_normal(f_normal3), self.f3.dowsample_depth(f_depth3)], dim=2))

        fusion_feature = [fusion0] + [fusion1] + [fusion2] + [fusion3]
        normal_feature = [f_normal0] + [f_normal1] + [f_normal2] + [f_normal3]
        depth_feature = [f_depth0] + [f_depth1] + [f_depth2] + [f_depth3]

        self.pred_normal_results = self.normal_decoder(btnec_feature, normal_feature, fusion_feature)
        self.pred_depth_results = self.depth_decoder(btnec_feature, depth_feature, fusion_feature)
        # exit()


        outputs = {}
        outputs["pred_normal"] = self.pred_normal_results[0]
        outputs["pred_multiscale_normal"] = self.pred_normal_results
        outputs["pred_depth"] = self.pred_depth_results[0]
        outputs["pred_multiscale_depth"] = self.pred_depth_results

        # exit()
        return outputs