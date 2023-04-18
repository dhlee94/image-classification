import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .SwinTransforemr_part import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint
import math


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, upsample=None, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.upsample = None
        elif upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            self.downsample = None
        else:
            self.downsample = None
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        elif self.downsample is not None:
            x = self.downsample(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.GELU = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.BatchNorm2d(x)
        x = self.GELU(x)        
        return x
    
class SwinTrans(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None, image_depth=20,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.image_depth = image_depth

        # split image into non-overlapping patches
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                              in_channels=in_chans, embed_dim=embed_dim,
                                              norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embedding.num_patches
        patches_resolution = self.patch_embedding.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.Outconv = OutConv(self.embed_dim*2**(len(depths)-1), self.embed_dim*2**len(depths))
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(self.embed_dim*2**len(depths), self.num_classes)
        #self.Softmax = nn.Softmax(dim=-1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward(self, x):
        # Dencoder and Skip connection
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = self.patch_embedding(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        B, C , _ = x.shape
        size = int(math.sqrt(C))
        x = x.view(B, size, size, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.Outconv(x)        
        x = self.GAP(x)
        x = x.view(B, -1)
        x = self.linear(x)
        #x = self.Softmax(x)     
        return x