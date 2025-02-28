# Adaptive Spatial-Channel Feature Fusion and Self-Calibrated Convolution Module

## Code Introduction
This code implements the method proposed in the paper "Adaptive spatial-channel feature fusion and self-calibrated convolution for early maize seedlings counting in UAV images" (DOI: 10.3389/fpls.2024.1496801), adapted for the MMDetection framework of OpenMMLab.

## 1. Direct Integration into MMDetection
Simply copy the files to the designated locations to activate the related functionalities:

### 1.1 Integrate Adaptive Feature Fusion Module
- Copy `channel_mapper_ASFF.py` to the `mmdetection/mmdet/models/necks/` directory.
- Rename the file to `channel_mapper.py`

### 1.2 Integrate Self-Calibrated Convolution Module
- 将Copy `resnet_RCConv.py` to the `mmdetection/mmdet/models/backbones/` directory.
- Rename the file to `resnet.py`

## 2. Application to Other Projects
If you need to apply these modules to other codes, we recommend the following:

### 2.1 Self-Calibrated Convolution Module (RCConv)
- You can use `RCConv.py`as a substitute for standard 3x3 convolutions, directly inserting it at any position where enhanced feature extraction capabilities are needed.
- This module effectively enhances the model's perception of spatial information.

### 2.2 Adaptive Spatial-Channel Feature Fusion Module (ASCFF)
- You can integrate the `ASCFF.py` mdule into the feature fusion part of any network (such as the neck module).
- Refer to the example: Check the implementation of the ASCFF module in `channel_mapper_ASCFF.py`.
- This module effectively integrates multi-scale features, enhancing the performance of small object detection.

## Citation
If you use this code in your research, please cite the following paper:

@article{sun2024adaptive,
  title={Adaptive spatial-channel feature fusion and self-calibrated convolution for early maize seedlings counting in UAV images},
  author={Sun, Zhenyuan and Yang, Zhi and Ding, Yimin and Sun, Boyan and Li, Saiju and Guo, Zhen and Zhu, Lei},
  journal={Frontiers in Plant Science},
  volume={15},
  pages={1496801},
  year={2024},
  publisher={Frontiers},
  doi={10.3389/fpls.2024.1496801}
}

