# 自适应空间通道特征融合和自校准卷积模块

## 代码简介
本代码实现了论文《Adaptive spatial-channel feature fusion and self-calibrated convolution for early maize seedlings counting in UAV images》(DOI: 10.3389/fpls.2024.1496801)中提出的方法，适配于OpenMMLab的MMDetection框架。

## 1. 直接集成到MMDetection
只需简单复制文件到指定位置即可启用相关功能：

### 1.1 集成自适应特征融合模块
- 将`channel_mapper_ASFF.py`复制到`mmdetection/mmdet/models/necks/`目录
- 将文件重命名为`channel_mapper.py`

### 1.2 集成自校准卷积模块
- 将`resnet_RCConv.py`复制到`mmdetection/mmdet/models/backbones/`目录
- 将文件重命名为`resnet.py`

## 2. 应用到其他项目
若需要将这些模块应用到其他代码中，我们建议：

### 2.1 自校准卷积模块(RCConv)
- 可将`RCConv.py`作为标准3×3卷积的替代品，直接插入到任何需要增强特征提取能力的位置
- 该模块能有效增强模型对空间信息的感知能力

### 2.2 自适应空间通道特征融合模块(ASCFF)
- 可将`ASCFF.py`模块集成到任何网络的特征融合部分（如neck模块）
- 参考示例：查看`channel_mapper_ASCFF.py`中ASCFF模块的实现方式
- 该模块能有效整合多尺度特征，提升小目标检测性能

## 引用
如果您在研究中使用了本代码，请引用以下论文：
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


