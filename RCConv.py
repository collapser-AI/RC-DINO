

import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..layers import ResLayer
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


    

class SXGConv(nn.Module):
    def __init__(self, inchannel, outchannel,stride):
        super(SXGConv, self).__init__()

        inplanes = int(inchannel)
        planes = int(outchannel)

        self.k1_1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1)

        self.k1_2 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1)

        self.k1_3 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.k1_4 = nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.stride = stride

    def forward(self, x):
        

        tensor_0_1 = x.clone()
        tensor_0_2 = x.clone()
        tensor_0_3 = x.clone()

        tensor_1_1 = self.k1_1(tensor_0_1)

        tensor_1_2 = nn.functional.interpolate(tensor_0_2, scale_factor=2, mode='bilinear', align_corners=True)
        tensor_1_2 = self.k1_2(tensor_1_2) 
        tensor_1_2 = self.k1_3(tensor_1_2) 

        tensor_1_2 = nn.functional.interpolate(tensor_1_2 , size=(tensor_1_1.size(2),tensor_1_1.size(3)),mode='bilinear', align_corners=True)
  
        out0 = (torch.mul(tensor_1_1, tensor_1_2))

        if self.stride == 1:            
            out = torch.add(tensor_0_3, out0)
        else:
            tensor_0_3 = self.k1_4(tensor_0_3)
            tensor_1_3 = nn.functional.interpolate(tensor_0_3 , size=(tensor_1_1.size(2),tensor_1_1.size(3)),mode='bilinear', align_corners=True)
            out = torch.add(tensor_1_3, out0)
       
        return out



