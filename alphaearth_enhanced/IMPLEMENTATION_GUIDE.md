# AlphaEarth Enhanced - Implementation Guide

## 项目概述

AlphaEarth Enhanced 是一个综合了多个先进地球观测模型优点的多模态基础模型。本指南将帮助您理解如何实现和扩展这个模型。

## 快速开始

### 1. 环境设置

```bash
# 克隆仓库
git clone https://github.com/your-repo/alphaearth-enhanced.git
cd alphaearth-enhanced

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

#### 数据源
- **Sentinel-2**: 光学影像（13个波段）
- **Sentinel-1**: SAR数据（VV, VH极化）
- **Landsat**: 热红外数据
- **文本标注**: 地理描述和元数据

#### 数据组织结构
```
data/
├── sentinel2/
│   ├── tiles/
│   │   ├── 2023/
│   │   └── 2024/
│   └── metadata.json
├── sentinel1/
│   ├── scenes/
│   └── metadata.json
├── landsat/
│   ├── thermal/
│   └── metadata.json
└── annotations/
    ├── descriptions.json
    └── labels.csv
```

### 3. 模型训练

#### 预训练
```python
from models import AlphaEarthEnhanced
from training import Trainer
import yaml

# 加载配置
with open('configs/pretrain.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化模型
model = AlphaEarthEnhanced(**config['model'])

# 初始化训练器
trainer = Trainer(model, config)

# 开始预训练
trainer.pretrain()
```

#### 微调
```python
# 加载预训练权重
model.load_pretrained('checkpoints/pretrained.pth')

# 针对特定任务微调
trainer.finetune(task='land_cover_classification')
```

## 核心实现细节

### 1. 多模态数据处理

#### 光学数据预处理
```python
class OpticalPreprocessor:
    def __init__(self):
        self.band_stats = {
            'mean': [1370.19, 1184.35, ...],  # 13 bands
            'std': [633.90, 580.95, ...]
        }
    
    def process(self, image):
        # 大气校正
        image = self.atmospheric_correction(image)
        
        # 云掩膜
        cloud_mask = self.detect_clouds(image)
        
        # 归一化
        image = self.normalize(image)
        
        return image, cloud_mask
```

#### SAR数据预处理
```python
class SARPreprocessor:
    def process(self, sar_data):
        # 转换为dB
        sar_db = 10 * np.log10(sar_data + 1e-8)
        
        # 斑点滤波
        sar_filtered = self.speckle_filter(sar_db)
        
        # 提取特征
        vv_vh_ratio = sar_data[..., 0] / (sar_data[..., 1] + 1e-8)
        vv_vh_diff = sar_data[..., 0] - sar_data[..., 1]
        
        return np.stack([sar_filtered, vv_vh_ratio, vv_vh_diff])
```

### 2. 模型架构实现

#### Vision Transformer with MAE
```python
class VisionTransformerMAE(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 3D补丁嵌入
        self.patch_embed = PatchEmbed3D(...)
        
        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(...) for _ in range(depth)
        ])
        
        # MAE解码器
        self.decoder = nn.ModuleList([
            TransformerBlock(...) for _ in range(decoder_depth)
        ])
    
    def forward_mae(self, x, mask_ratio=0.75):
        # 编码可见补丁
        latent, mask, ids_restore = self.encode_with_mask(x, mask_ratio)
        
        # 解码重建
        pred = self.decode(latent, ids_restore)
        
        # 计算损失
        loss = self.reconstruction_loss(x, pred, mask)
        
        return loss, pred, mask
```

#### 多模态融合
```python
class MultimodalFusion(nn.Module):
    def __init__(self, fusion_type='cross_attention'):
        super().__init__()
        
        if fusion_type == 'cross_attention':
            self.fusion = CrossModalAttention(...)
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(...)
    
    def forward(self, features_dict):
        # features_dict: {'optical': ..., 'sar': ..., 'text': ...}
        
        # 跨模态注意力
        fused = self.fusion(features_dict)
        
        return fused
```

### 3. 训练策略

#### 自监督预训练
```python
def pretrain_step(model, batch):
    # MAE预训练
    optical = batch['optical']
    mae_loss, pred, mask = model.forward_mae(optical, mask_ratio=0.75)
    
    # 对比学习
    images = batch['images']
    texts = batch['texts']
    contrastive_loss = model.forward_contrastive(images, texts)
    
    # 总损失
    total_loss = mae_loss + 0.5 * contrastive_loss
    
    return total_loss
```

#### 多任务学习
```python
def multitask_training(model, batch):
    losses = {}
    
    # 分类任务
    if 'classification' in batch:
        pred = model(batch['data'], task='classification')
        losses['cls'] = F.cross_entropy(pred, batch['labels'])
    
    # 分割任务
    if 'segmentation' in batch:
        pred = model(batch['data'], task='segmentation')
        losses['seg'] = dice_loss(pred, batch['masks'])
    
    # 变化检测
    if 'change_detection' in batch:
        pred = model(batch['t1'], batch['t2'], task='change_detection')
        losses['change'] = F.binary_cross_entropy(pred, batch['changes'])
    
    return sum(losses.values())
```

## 扩展和优化

### 1. 模型缩放

#### 参数缩放
```python
MODEL_CONFIGS = {
    'small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'params': '86M'
    },
    'base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'params': '300M'
    },
    'large': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'params': '1B'
    },
    'huge': {
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'params': '2B'
    }
}
```

#### 数据缩放
```python
def scale_data_pipeline(scale_factor):
    if scale_factor == 'small':
        return {
            'batch_size': 32,
            'image_size': 112,
            'num_frames': 1,
            'data_fraction': 0.1
        }
    elif scale_factor == 'large':
        return {
            'batch_size': 256,
            'image_size': 224,
            'num_frames': 4,
            'data_fraction': 1.0
        }
```

### 2. 性能优化

#### Flash Attention
```python
from flash_attn import flash_attn_func

class FlashAttention(nn.Module):
    def forward(self, q, k, v):
        # 使用Flash Attention加速
        out = flash_attn_func(q, k, v, dropout_p=0.1)
        return out
```

#### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_step(model, batch, optimizer):
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 分布式训练
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)
    
    model = AlphaEarthEnhanced(...).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 训练循环
    for epoch in range(num_epochs):
        train_epoch(model, dataloader)
```

### 3. 添加新模态

#### 实现新的编码器
```python
class HyperspectralEncoder(nn.Module):
    def __init__(self, num_bands=200, embed_dim=768):
        super().__init__()
        
        # 光谱注意力
        self.spectral_attention = SpectralAttention(num_bands)
        
        # 降维
        self.dim_reduction = nn.Conv2d(num_bands, 64, 1)
        
        # 主干网络
        self.backbone = VisionTransformer(...)
    
    def forward(self, x):
        # x: [B, T, 200, H, W] - 200个光谱波段
        
        # 光谱注意力
        x = self.spectral_attention(x)
        
        # 降维
        x = self.dim_reduction(x)
        
        # 编码
        features = self.backbone(x)
        
        return features
```

#### 集成到主模型
```python
# 在AlphaEarthEnhanced中添加
self.hyperspectral_encoder = HyperspectralEncoder(...)

# 在forward中处理
if hyperspectral is not None:
    features['hyperspectral'] = self.hyperspectral_encoder(hyperspectral)
```

## 评估和基准测试

### 1. 评估指标

```python
class Evaluator:
    def __init__(self):
        self.metrics = {
            'classification': accuracy_score,
            'segmentation': iou_score,
            'change_detection': f1_score,
            'reconstruction': mse_loss
        }
    
    def evaluate(self, model, dataloader, task):
        predictions = []
        targets = []
        
        for batch in dataloader:
            pred = model(batch['data'], task=task)
            predictions.append(pred)
            targets.append(batch['target'])
        
        score = self.metrics[task](predictions, targets)
        return score
```

### 2. 下游任务评估

```python
DOWNSTREAM_TASKS = {
    'land_cover': {
        'dataset': 'EuroSAT',
        'metric': 'accuracy',
        'baseline': 0.85
    },
    'change_detection': {
        'dataset': 'OSCD',
        'metric': 'f1_score',
        'baseline': 0.75
    },
    'cloud_removal': {
        'dataset': 'SEN12MS-CR',
        'metric': 'ssim',
        'baseline': 0.80
    }
}
```

## 部署指南

### 1. 模型导出

```python
# 导出为ONNX
torch.onnx.export(
    model,
    dummy_input,
    "alphaearth.onnx",
    export_params=True,
    opset_version=11,
    input_names=['optical', 'sar', 'text'],
    output_names=['output']
)

# 导出为TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("alphaearth.pt")
```

### 2. 推理优化

```python
# 量化
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# 剪枝
from torch.nn.utils import prune

prune.l1_unstructured(model.backbone, name='weight', amount=0.2)
```

### 3. API服务

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    optical_path: str
    sar_path: str
    text_description: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    # 加载数据
    optical = load_image(request.optical_path)
    sar = load_sar(request.sar_path)
    text = request.text_description
    
    # 推理
    with torch.no_grad():
        output = model(optical, sar, text)
    
    return {"prediction": output.tolist()}
```

## 常见问题

### Q1: 如何处理不同分辨率的输入？
A: 使用自适应池化或插值将输入调整到固定大小，或使用Vision Transformer的灵活位置编码。

### Q2: 如何处理缺失的模态？
A: 使用零填充或学习的缺失标记，模型会自动适应可用的模态。

### Q3: 如何加速训练？
A: 使用混合精度训练、梯度累积、分布式训练和Flash Attention。

### Q4: 如何选择合适的模型大小？
A: 根据可用的计算资源和任务复杂度，从小模型开始，逐步扩展。

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

MIT License - 详见 LICENSE 文件

## 联系方式

- 项目主页: https://github.com/your-repo/alphaearth-enhanced
- 问题反馈: https://github.com/your-repo/alphaearth-enhanced/issues
- 邮件: alphaearth@example.com