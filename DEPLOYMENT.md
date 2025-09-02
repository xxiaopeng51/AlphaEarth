# AlphaEarth Foundations - 部署指南

## 项目概述

AlphaEarth Foundations 是一个增强的全球多模态基础模型，旨在复现并改进Google's AlphaEarth Foundations。该项目结合了Clay Foundation Model、SatCLIP、Prithvi等先进模型的设计理念，构建了一个覆盖全球尺度、融合多模态数据、性能更优的地球观测基础模型。

## 核心特性

- **全球尺度覆盖**: 处理全球范围内的多模态地球观测数据
- **多模态融合**: 整合光学影像、雷达数据、气象数据、文本描述等多种模态
- **扩展法则优化**: 遵循scaling law，通过增加模型参数和数据量提升性能
- **时空建模**: 采用先进的时空注意力机制，捕捉地理远距离关联和时序动态

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA support (推荐RTX 3090/4090或A100)
- **内存**: 至少32GB RAM (推荐64GB+)
- **存储**: 至少500GB可用空间 (推荐1TB+)
- **CPU**: 多核处理器 (推荐16核+)

### 软件要求
- **Python**: 3.8+
- **CUDA**: 11.8+
- **PyTorch**: 2.0+
- **其他依赖**: 见requirements.txt

## 安装步骤

### 1. 环境设置

```bash
# 创建虚拟环境
conda create -n alphaearth python=3.9
conda activate alphaearth

# 或使用venv
python -m venv alphaearth
source alphaearth/bin/activate  # Linux/Mac
# alphaearth\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 数据准备

### 1. 数据目录结构

```
data/
├── optical/          # 光学卫星影像
│   ├── sentinel2/    # Sentinel-2数据
│   ├── landsat/      # Landsat数据
│   └── ...
├── radar/            # 雷达数据
│   ├── sentinel1/    # Sentinel-1数据
│   └── ...
├── meteorological/   # 气象数据
│   ├── era5/         # ERA5数据
│   ├── gfs/          # GFS数据
│   └── ...
├── text/             # 文本数据
│   ├── annotations/  # 标注数据
│   └── ...
└── metadata/         # 元数据
    ├── train.csv     # 训练集元数据
    ├── val.csv       # 验证集元数据
    └── test.csv      # 测试集元数据
```

### 2. 数据格式

#### 光学影像
- **格式**: GeoTIFF (.tif) 或图像文件 (.jpg, .png)
- **分辨率**: 10m-30m
- **波段**: RGB, NIR, SWIR等
- **命名**: 建议使用时间戳和位置信息

#### 雷达数据
- **格式**: GeoTIFF (.tif)
- **极化**: VV, VH, HH, HV
- **分辨率**: 5m-20m

#### 气象数据
- **格式**: NetCDF (.nc) 或HDF5 (.h5)
- **变量**: 温度、降水、湿度、气压等
- **时间分辨率**: 小时级或日级

#### 文本数据
- **格式**: JSON (.json) 或CSV (.csv)
- **内容**: 描述、标注、元数据等

### 3. 元数据格式

```csv
id,optical_path,radar_path,meteorological_path,text_path,latitude,longitude,timestamp,labels
sample_001,optical/sentinel2/2023/001.tif,radar/sentinel1/2023/001.tif,meteorological/era5/2023/001.nc,text/annotations/001.json,40.7128,-74.0060,2023-01-01,class_1
```

## 模型训练

### 1. 单GPU训练

```bash
python scripts/train.py \
    --data_root /path/to/data \
    --metadata_file /path/to/metadata/train.csv \
    --config configs/train_config.yaml \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --output_dir ./outputs \
    --experiment_name my_experiment
```

### 2. 多GPU训练

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --data_root /path/to/data \
    --metadata_file /path/to/metadata/train.csv \
    --config configs/train_config.yaml \
    --batch_size 32 \
    --epochs 100 \
    --distributed \
    --world_size 4 \
    --output_dir ./outputs \
    --experiment_name my_experiment
```

### 3. 混合精度训练

```bash
python scripts/train.py \
    --data_root /path/to/data \
    --metadata_file /path/to/metadata/train.csv \
    --config configs/train_config.yaml \
    --use_amp \
    --batch_size 32 \
    --epochs 100 \
    --output_dir ./outputs \
    --experiment_name my_experiment
```

## 模型评估

### 1. 基本评估

```bash
python scripts/evaluate.py \
    --model_path ./outputs/checkpoints/best_model.pth \
    --data_root /path/to/data \
    --metadata_file /path/to/metadata/test.csv \
    --config configs/eval_config.yaml \
    --output_dir ./evaluation_results
```

### 2. 多任务评估

```bash
python scripts/evaluate.py \
    --model_path ./outputs/checkpoints/best_model.pth \
    --data_root /path/to/data \
    --metadata_file /path/to/metadata/test.csv \
    --config configs/eval_config.yaml \
    --tasks classification regression segmentation \
    --output_dir ./evaluation_results
```

## 模型推理

### 1. 单样本推理

```python
import torch
from models import AlphaEarthFoundations
from utils import load_checkpoint

# 加载模型
model = AlphaEarthFoundations()
checkpoint = load_checkpoint('./outputs/checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 准备数据
optical_data = torch.randn(1, 13, 224, 224)  # 光学数据
radar_data = torch.randn(1, 2, 224, 224)     # 雷达数据
meteorological_data = torch.randn(1, 24, 128) # 气象数据

# 推理
with torch.no_grad():
    outputs = model(
        optical_data=optical_data,
        radar_data=radar_data,
        meteorological_data=meteorological_data
    )

print(f"Global features shape: {outputs['global_features'].shape}")
```

### 2. 批量推理

```python
from torch.utils.data import DataLoader
from data import MultiModalDataset

# 创建数据集
dataset = MultiModalDataset(
    data_root='/path/to/data',
    metadata_file='/path/to/metadata/test.csv'
)

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# 批量推理
model.eval()
all_features = []

for batch in dataloader:
    with torch.no_grad():
        outputs = model(**batch)
        all_features.append(outputs['global_features'])

all_features = torch.cat(all_features, dim=0)
print(f"All features shape: {all_features.shape}")
```

## 性能优化

### 1. 内存优化

```python
# 启用梯度检查点
model = AlphaEarthFoundationsWithScaling(
    base_model=model,
    use_gradient_checkpointing=True
)

# 使用混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**batch)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 计算优化

```python
# 启用模型编译 (PyTorch 2.0+)
model = torch.compile(model)

# 使用channels_last内存格式
model = model.to(memory_format=torch.channels_last)
```

### 3. 分布式训练优化

```python
# 使用DDP
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])

# 使用DeepSpeed (可选)
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=deepspeed_config
)
```

## 监控和日志

### 1. Weights & Biases

```bash
# 安装wandb
pip install wandb

# 登录
wandb login

# 训练时启用wandb
python scripts/train.py \
    --use_wandb \
    --wandb_project alphaearth-foundations \
    --experiment_name my_experiment
```

### 2. TensorBoard

```bash
# 安装tensorboard
pip install tensorboard

# 启动tensorboard
tensorboard --logdir ./outputs/logs

# 在浏览器中访问 http://localhost:6006
```

## 故障排除

### 1. 常见问题

#### CUDA内存不足
```bash
# 减少批次大小
--batch_size 8

# 启用梯度累积
--gradient_accumulation_steps 4

# 使用混合精度
--use_amp
```

#### 数据加载慢
```bash
# 增加数据加载器工作进程
--num_workers 8

# 启用内存固定
--pin_memory
```

#### 模型收敛慢
```bash
# 调整学习率
--lr 5e-5

# 使用学习率调度器
--scheduler cosine_annealing

# 增加预热轮数
--warmup_epochs 20
```

### 2. 调试模式

```bash
# 启用详细日志
python scripts/train.py \
    --log_level DEBUG \
    --log_every 10

# 使用小数据集测试
python scripts/train.py \
    --max_samples 1000
```

## 部署到生产环境

### 1. 模型导出

```python
# 导出为TorchScript
model.eval()
example_input = {
    'optical_data': torch.randn(1, 13, 224, 224),
    'radar_data': torch.randn(1, 2, 224, 224),
    'meteorological_data': torch.randn(1, 24, 128)
}

traced_model = torch.jit.trace(model, example_input)
traced_model.save('alphaearth_model.pt')
```

### 2. 容器化部署

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "scripts/serve.py"]
```

### 3. API服务

```python
# scripts/serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = None

class InferenceRequest(BaseModel):
    optical_data: list
    radar_data: list
    meteorological_data: list

@app.on_event("startup")
async def load_model():
    global model
    model = torch.load('alphaearth_model.pt')
    model.eval()

@app.post("/predict")
async def predict(request: InferenceRequest):
    # 处理推理请求
    pass
```

## 贡献指南

### 1. 代码贡献

```bash
# Fork项目
git clone https://github.com/your-username/alphaearth-foundations.git
cd alphaearth-foundations

# 创建分支
git checkout -b feature/new-feature

# 提交更改
git add .
git commit -m "Add new feature"
git push origin feature/new-feature
```

### 2. 测试

```bash
# 运行测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_models.py
```

### 3. 文档

- 更新README.md
- 添加代码注释
- 更新API文档

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 联系方式

- 项目主页: https://github.com/your-username/alphaearth-foundations
- 问题反馈: https://github.com/your-username/alphaearth-foundations/issues
- 邮箱: your-email@example.com

## 致谢

感谢以下开源项目的贡献：
- [AlphaEarth Foundations](https://github.com/Brayden-Zhang/alphaearth-foundations)
- [Clay Foundation Model](https://github.com/Clay-foundation/model)
- [SatCLIP](https://github.com/microsoft/satclip)
- [Prithvi](https://github.com/NASA-IMPACT/prithvi)