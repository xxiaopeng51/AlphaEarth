# AlphaEarth Foundations - Enhanced Global Multimodal Foundation Model

## 项目概述

本项目旨在复现并改进Google's AlphaEarth Foundations，结合Clay Foundation Model、SatCLIP、Prithvi等先进模型的设计理念，构建一个覆盖全球尺度、融合多模态数据、性能更优的地球观测基础模型。

## 核心特性

- **全球尺度覆盖**: 处理全球范围内的多模态地球观测数据
- **多模态融合**: 整合光学影像、雷达数据、气象数据、文本描述等多种模态
- **扩展法则优化**: 遵循scaling law，通过增加模型参数和数据量提升性能
- **时空建模**: 采用先进的时空注意力机制，捕捉地理远距离关联和时序动态

## 技术架构

### 模型组件
- **多模态编码器**: 处理不同模态的输入数据
- **时空精度模块(STP)**: 捕捉空间、时间和分辨率细节
- **统一表示空间**: 将不同模态对齐到共享表示空间
- **任务特定头**: 支持多种下游任务

### 数据模态
- 光学卫星影像 (Sentinel-2, Landsat, etc.)
- 雷达数据 (Sentinel-1)
- 气象数据 (ERA5, GFS)
- 地形数据 (DEM)
- 文本描述和元数据

## 项目结构

```
alphaearth-foundations/
├── configs/                 # 配置文件
├── data/                    # 数据处理模块
│   ├── datasets/           # 数据集定义
│   ├── transforms/         # 数据变换
│   └── utils/              # 数据工具
├── models/                  # 模型定义
│   ├── encoders/           # 多模态编码器
│   ├── fusion/             # 模态融合模块
│   └── heads/              # 任务特定头
├── training/                # 训练框架
│   ├── losses/             # 损失函数
│   ├── optimizers/         # 优化器
│   └── schedulers/         # 学习率调度器
├── evaluation/              # 评估框架
├── utils/                   # 工具函数
└── scripts/                 # 训练和评估脚本
```

## 快速开始

### 环境设置
```bash
pip install -r requirements.txt
```

### 数据准备
```bash
python scripts/prepare_data.py --config configs/data_config.yaml
```

### 模型训练
```bash
python scripts/train.py --config configs/train_config.yaml
```

### 模型评估
```bash
python scripts/evaluate.py --config configs/eval_config.yaml
```

## 参考模型

- [AlphaEarth Foundations](https://github.com/Brayden-Zhang/alphaearth-foundations)
- [Clay Foundation Model](https://github.com/Clay-foundation/model)
- [SatCLIP](https://github.com/microsoft/satclip)
- [Prithvi](https://github.com/NASA-IMPACT/prithvi)

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请查看CONTRIBUTING.md了解详细信息。

## 许可证

本项目采用MIT许可证。详见LICENSE文件。