import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from random  import random
import os
from torchsummary import summary as summary_
from timm.models.layers import to_2tuple
from einops import rearrange
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, act_layer=nn.GELU):
        super(FeedForward, self).__init__()
        hidden_channels =  hidden_channels if hidden_channels else in_channels//4
        self.feed_layer = nn.Sequential(
                    nn.LayerNorm(in_channels, eps=1e-5),
                    nn.Linear(in_channels, hidden_channels, bias=False),
                    act_layer(),
                    nn.Linear(hidden_channels, in_channels, bias=False)
        )
    def forward(self, x):
        return self.feed_layer(x)
    
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int=0, act_layer=nn.SiLU):
        super(SqueezeExcitation, self).__init__()
        hidden_channels = hidden_channels if hidden_channels else in_channels//16
        self.act_layer = act_layer(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(in_channels=hidden_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.scale_activation = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)
        y = self.fc1(y)
        y = self.act_layer(y)
        y =self.scale_activation(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class StemBlock(nn.Module):
    def __init__(self , in_channels: int=3, hidden_channels: int=0, out_channels: int=64, act_layer=nn.GELU, norm_trigger: bool=False, norm=nn.BatchNorm2d, downsample=False):
        super(StemBlock, self).__init__()
        if hidden_channels==0:
            hidden_channels = out_channels
        self.act_layer = act_layer()
        self.trigger = norm_trigger
        if self.trigger:
            self.norm = norm(hidden_channels)
        stride = 2 if downsample else 1
        self.block1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.block2 = nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
    def forward(self, x):
        x = self.block1(x)
        if self.trigger:
            x = self.norm(x)
            x = self.act_layer(x)
        x = self.block2(x)
        return x

class MBBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, act_layer=nn.SiLU, norm=nn.BatchNorm2d, downsample=False):
        super(MBBlock, self).__init__()
        self.downsample = downsample
        if downsample:
            self.downlayer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block = nn.Sequential(
            self._make_layer(in_channels=in_channels, out_channels=in_channels*4, act_layer=act_layer, norm=norm, de_trigger=False),
            self._make_layer(in_channels=in_channels*4, out_channels=in_channels*4, act_layer=act_layer, norm=norm, de_trigger=True),
            SqueezeExcitation(in_channels=in_channels*4),
            self._make_layer(in_channels=in_channels*4, out_channels=out_channels, act_layer=None, norm=norm, de_trigger=False)
        )
        self.shortcut = nn.Sequential(
            self._make_layer(in_channels=in_channels, out_channels=out_channels, act_layer=None, norm=norm, de_trigger=False)
        )
    def forward(self, x):
        x = self.downlayer(x) if self.downsample else x
        return self.block(x) + self.shortcut(x)
    
    def _make_layer(self, in_channels, out_channels, act_layer, norm, de_trigger):
        layer = nn.ModuleList([])
        layer += [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False) if de_trigger
            else nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
        layer = layer + [norm(out_channels, eps=0.001, momentum=0.1, affine=True)] if norm else layer
        layer = layer + [act_layer(inplace=True)] if act_layer else layer
        return nn.Sequential(*layer)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = to_2tuple(image_size)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)), indexing='ij')
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(in_channels * 4)

        self.ih, self.iw = to_2tuple(img_size)
        self.downsample = downsample

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

        self.attn = Attention(in_channels, out_channels, img_size, heads, dim_head, dropout)
        self.ff = FeedForward(in_channels=out_channels, hidden_channels=hidden_dim)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            self.attn,
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            self.ff,
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.pool(x)
            x = self.proj(x) + self.attn(x)
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x
                
class CoatNet(nn.Module):
    def __init__(self, in_channels: int, out_class: int, img_size: int=224, act_layer=nn.GELU(), norm=nn.BatchNorm2d, 
                 dim=[64, 96, 192, 384, 768], block_size=[2, 2, 3, 5, 2], types=['C', 'C', 'A', 'A'], heads=8, dim_head=32, dropout=0.):
        super(CoatNet, self).__init__()
        self.block_size = block_size
        self.features = nn.ModuleList([])
        self.types = types
        for idx in range(len(dim)):
            if idx==0:
                self.features.append(self._make_Convlayer(block=StemBlock, in_channels=in_channels, out_channels=dim[idx], repeat=block_size[idx]))
            else:
                if types[idx-1]=='C':
                    self.features.append(self._make_Convlayer(block=MBBlock, in_channels=dim[idx-1], out_channels=dim[idx], repeat=block_size[idx]))
                else:
                    self.features.append(self._make_Attenlayer(block=Transformer, in_channels=dim[idx-1], out_channels=dim[idx], repeat=block_size[idx], 
                                                               img_size=img_size//(2**(idx+1)), heads=heads, dim_head=dim_head, dropout=0.))
        self.features.append(nn.AdaptiveAvgPool2d(output_size=1))
        self.features.append(nn.Linear(dim[-1], out_class, bias=False))
    def forward(self, x):
        B, _, _, _ = x.size()
        for layer in self.features[:-2]:
            x = layer(x)
        x = self.features[-2](x)
        x = x.view(B, -1)
        x = self.features[-1](x)
        return x

    def _make_Convlayer(self, block, in_channels, out_channels, repeat, act_layer=nn.SiLU, norm=None, de_trigger=False):
        layers = nn.ModuleList([])
        for num in range(repeat):
            if num==0:
                layers.append(block(in_channels=in_channels, out_channels=out_channels, downsample=True))
            else:
                layers.append(block(in_channels=out_channels, out_channels=out_channels))
        return nn.Sequential(*layers)

    def _make_Attenlayer(self, block, in_channels, out_channels, repeat, img_size, heads=8, dim_head=32, dropout=0.):
        layers = nn.ModuleList([])
        for num in range(repeat):
            if num==0:
                layers.append(block(in_channels=in_channels, out_channels=out_channels, img_size=img_size, heads=heads, dim_head=dim_head, dropout=0., downsample=True))
            else:
                layers.append(block(in_channels=out_channels, out_channels=out_channels, img_size=img_size, heads=heads, dim_head=dim_head, dropout=0.))
        return nn.Sequential(*layers)
    
if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    model = CoatNet(in_channels=3, out_class=1, img_size=128, act_layer=nn.GELU(), norm=nn.BatchNorm2d, 
                 dim=[64, 96, 192, 384, 768], block_size=[2, 2, 3, 5, 2], types=['C', 'C', 'A', 'A'], heads=8, dim_head=32, dropout=0.)
    model.cuda()
    summary_(model, (3, 128, 128), batch_size=3)