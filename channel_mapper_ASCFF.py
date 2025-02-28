# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn

from mmengine.model import BaseModule
from torch import Tensor
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
import torch

import torch.nn.functional as F
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch.nn as nn

from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig


import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialChannelAttention, self).__init__()

        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels , 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels , in_channels, 1, bias=False)
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        max_result = self.max_pool(x)
        avg_result = self.avg_pool(x)
        channel_output = torch.sigmoid(self.channel_attention(max_result + avg_result))

        # Spatial attention
        max_result = torch.max(x, dim=1, keepdim=True)[0]
        avg_result = torch.mean(x, dim=1, keepdim=True)
        spatial_output = self.spatial_attention(torch.cat([max_result, avg_result], dim=1))

        # Merge both attentions multiplicatively
        return x * channel_output * spatial_output



def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage




class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        channel_list = [2048,1024,512,2048]
        if level==0:
            self.stride_level_1 = add_conv(1024, channel_list[0], 3, 2)
            self.stride_level_2 = add_conv(512, channel_list[0], 3, 2)
            self.expand = add_conv(channel_list[0]*3, 256, 3, 1)
            self.SpatialChannelAttention = SpatialChannelAttention(in_channels=channel_list[0]*3)  
        elif level==1:
            self.compress_level_0 = add_conv(2048, channel_list[1], 1, 1)
            self.stride_level_2 = add_conv(512, channel_list[1], 3, 2)
            self.expand = add_conv(channel_list[1]*3, 256, 3, 1)
            self.SpatialChannelAttention = SpatialChannelAttention(in_channels=channel_list[1]*3)  
        elif level==2:
            self.compress_level_0 = add_conv(2048,channel_list[2], 1, 1)
            self.compress_level_1 = add_conv(1024,channel_list[2], 1, 1)
            self.expand = add_conv(channel_list[2]*3, 256, 3, 1)
            self.SpatialChannelAttention = SpatialChannelAttention(in_channels=channel_list[2]*3)  
        elif level==3:
            self.stride_level_0 = add_conv(2048, channel_list[3], 3, 2)
            self.stride_level_1 = add_conv(1024, channel_list[3], 3, 2)
            self.stride_level_2 = add_conv(512, channel_list[3], 3, 2)
            self.expand = add_conv(channel_list[3]*3, 256, 3, 1)
            self.SpatialChannelAttention = SpatialChannelAttention(in_channels=channel_list[3]*3)  
        self.vis= vis


    def forward(self, x_list):        

        for i in x_list:
            if i.shape[1] == 512:
                x_level_2 = i
            if i.shape[1] == 1024:
                x_level_1 = i
            if i.shape[1] == 2048:
                x_level_0 = i

        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_1_resized =x_level_1
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_0_resized = nn.functional.interpolate(level_0_resized , size=(level_1_resized.size(2),level_1_resized.size(3)),mode='bilinear', align_corners=True)

            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_2_resized =x_level_2
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_0_resized = nn.functional.interpolate(level_0_resized , size=(level_2_resized.size(2),level_2_resized.size(3)),mode='bilinear', align_corners=True)

            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized =F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_1_resized = nn.functional.interpolate(level_1_resized , size=(level_2_resized.size(2),level_2_resized.size(3)),mode='bilinear', align_corners=True)
            

        elif self.level==3:
            level_0_resized = self.stride_level_0(x_level_0)

            level_1_downsampled_inter =F.max_pool2d(x_level_1, 3, stride=2, padding=1)
            level_1_resized = self.stride_level_1(level_1_downsampled_inter)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_downsampled_inter =F.max_pool2d(level_2_downsampled_inter, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)


        level_all = torch.cat((level_0_resized, level_1_resized, level_2_resized),1)
        level_all = self.SpatialChannelAttention(level_all)
        levels_weight = level_all

        out = self.expand(level_all)
        

        if self.vis:
            return out, levels_weight, level_all.sum(dim=1)
        else:
            return out
        





@MODELS.register_module()
class ChannelMapper(BaseModule):

    def __init__(
        self,        
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        num_outs: int = None,
        init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__()
        
        
        self.level_l = ASFF(level=2,rfb=False,vis=False)
        self.level_m = ASFF(level=1,rfb=False,vis=False)
        self.level_s1 = ASFF(level=0,rfb=False,vis=False)
        self.level_s2 = ConvModule(
                        2048,
                        256,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)
        act_cfg = act_cfg
    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:

        outs = [self.level_l(inputs),
                self.level_m(inputs),
                self.level_s1(inputs),
                self.level_s2(inputs[2]),
                ]

        return tuple(outs)




