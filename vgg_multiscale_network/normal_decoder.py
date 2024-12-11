import numpy as np
import torch
import torch.nn as nn

from .panoformer import *

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):
    src = nn.functional.interpolate(src,size=tar.shape[2:],mode='bilinear',align_corners=True)
    return src

class NormalDecoder(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, 
                num_output_channels=3, **kwargs):
        super(NormalDecoder, self).__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.ref_point128x256 = genSamplingPattern(128, 256, 3, 3).cuda()
        self.ref_point64x128 = genSamplingPattern(64, 128, 3, 3).cuda()
        self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()
        self.ref_point16x32 = genSamplingPattern(16, 32, 3, 3).cuda()
        self.ref_point8x16 = genSamplingPattern(8, 16, 3, 3).cuda()

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        dec_dpr = enc_dpr[::-1]

        self.output_proj0 = OutputProjNormal(in_channel=32 * embed_dim, out_channel=num_output_channels, kernel_size=3, stride=1,
                                      input_resolution=(img_size//8, img_size//4))
        self.output_proj1 = OutputProjNormal(in_channel=16 * embed_dim, out_channel=num_output_channels, kernel_size=3, stride=1,
                                      input_resolution=(img_size//4, img_size//2))
        self.output_proj2 = OutputProjNormal(in_channel=8 * embed_dim, out_channel=num_output_channels, kernel_size=3, stride=1,
                                      input_resolution=(img_size//2, img_size))
        self.output_proj3 = OutputProjNormal(in_channel=4 * embed_dim, out_channel=num_output_channels, kernel_size=3, stride=1,
                                      input_resolution=(img_size, img_size * 2))

        # Decoder
        self.upsample_0 = upsample(embed_dim * 32, embed_dim * 16,
                                   input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)))
        self.decoderlayer_0 = BasicPanoformerLayer(dim=embed_dim * 16*2,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer,ref_point=self.ref_point16x32, flag = 1)
        self.upsample_1 = upsample(embed_dim * 16*2, embed_dim * 4*2,
                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))
        self.decoderlayer_1 = BasicPanoformerLayer(dim=embed_dim * 8*2,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer,ref_point=self.ref_point32x64,flag = 1)
        self.upsample_2 = upsample(embed_dim * 8*2, embed_dim * 2*2,
                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.decoderlayer_2 = BasicPanoformerLayer(dim=embed_dim * 4*2,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2, img_size * 2 // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer,ref_point=self.ref_point64x128, flag = 1)
        self.upsample_3 = upsample(embed_dim * 4*2, embed_dim*2, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.decoderlayer_3 = BasicPanoformerLayer(dim=embed_dim * 2*2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size, img_size * 2),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection, token_mlp=token_mlp,
                                                se_layer=se_layer, ref_point=self.ref_point128x256, flag = 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, btnec_feat, normal_feat, fusion_feat):
        up0 = self.upsample_0(btnec_feat)
        deconv0 = torch.cat([up0, normal_feat[3], fusion_feat[3]], -1)
        deconv0 = self.decoderlayer_0(deconv0)
        pred_normal0 = self.output_proj0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, normal_feat[2], fusion_feat[2]], -1)
        deconv1 = self.decoderlayer_1(deconv1)
        pred_normal1 = self.output_proj1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, normal_feat[1], fusion_feat[1]], -1)
        deconv2 = self.decoderlayer_2(deconv2)
        pred_normal2 = self.output_proj2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, normal_feat[0], fusion_feat[0]], -1)
        deconv3 = self.decoderlayer_3(deconv3)
        pred_normal3 = self.output_proj3(deconv3)
        pred_normal2 = _upsample_like(pred_normal2,pred_normal3)
        pred_normal1 = _upsample_like(pred_normal1,pred_normal3)
        pred_normal0 = _upsample_like(pred_normal0,pred_normal3)

        pred_outputs = []
        pred_outputs = [pred_normal3]+[pred_normal2]+[pred_normal1]+[pred_normal0]

        return pred_outputs