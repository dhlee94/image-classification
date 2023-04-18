import torch
import torch.nn as nn
import importlib
class EfficientNet(nn.Module):
    def __init__(self, in_channels, out_channels, weight_type='IMAGENET', name=None, pre_train=True):
        super(EfficientNet, self).__init__()
        #self.layer = efficientnet_v2_s(EfficientNet_V2_S_Weights) if pre_train else efficientnet_v2_s()
        self.layer = self.import_model(name=name, pre_train=pre_train, weight_type=weight_type)
        conv_out_channels = self.layer.features[0][0].out_channels
        linear_in_channels = self.layer.classifier[1].in_features
        self.layer.features[0][0] = nn.Conv2d(in_channels=in_channels, out_channels=conv_out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer.classifier[1] = nn.Linear(in_features=linear_in_channels, out_features=out_channels, bias=True)
        
    def forward(self, x):
        x = self.layer(x)
        return x
    
    def import_model(self, name, pre_train, weight_type='DEFAULT'):
        mod = importlib.import_module('torchvision.models')
        if name=='efficientnet_v2_m':
            if pre_train:
                return mod.efficientnet_v2_m(weights=mod.EfficientNet_V2_M_Weights.IMAGENET1K_V1) if weight_type=='IMAGENET' \
                       else mod.efficientnet_v2_m(weights=mod.EfficientNet_V2_M_Weights.DEFAULT)
            else:
                return mod.efficientnet_v2_m()
        elif name=='efficientnet_v2_l':
            if pre_train:
                return mod.efficientnet_v2_l(weights=mod.EfficientNet_V2_L_Weights.IMAGENET1K_V1) if weight_type=='IMAGENET' \
                       else mod.efficientnet_v2_l(weights=mod.EfficientNet_V2_L_Weights.DEFAULT)
            else:
                return mod.efficientnet_v2_l()
        else:    
            if pre_train:
                return mod.efficientnet_v2_s(weights=mod.EfficientNet_V2_S_Weights.IMAGENET1K_V1) if weight_type=='IMAGENET' \
                       else mod.efficientnet_v2_s(weights=mod.EfficientNet_V2_S_Weights.DEFAULT)
            else:
                return mod.efficientnet_v2_s()

        
        