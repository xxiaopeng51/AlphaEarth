#!/usr/bin/env python3
"""
Enhanced Earth Foundation Model Demo

演示脚本，展示模型的基本功能和使用方法。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from enhanced_earth.models.enhanced_earth_model import EnhancedEarthFoundationModel
from enhanced_earth.data.data_loaders import create_multimodal_dataloader
from enhanced_earth.utils.visualization import visualize_embeddings, plot_multimodal_data


def create_demo_model():
    """创建演示模型"""
    # 小型模型配置用于演示
    input_modalities = {
        "optical": {"channels": 6, "resolution": 10},  # 简化的光学数据
        "sar": {"channels": 2, "resolution": 10},      # VV, VH
        "environmental": {"channels": 4, "resolution": 1000}  # 简化的环境数据
    }
    
    model = EnhancedEarthFoundationModel(
        model_size="small",
        input_modalities=input_modalities,
        embed_dim=64,
        enable_text_alignment=True,
        enable_reconstruction=True
    )
    
    return model


def create_demo_data():
    """创建演示数据"""
    data_config = {
        "optical": {"source": "sentinel2", "channels": 6, "resolution": 10},
        "sar": {"source": "sentinel1", "channels": 2, "resolution": 10},
        "environmental": {"source": "era5", "channels": 4, "resolution": 1000}
    }
    
    dataloader = create_multimodal_dataloader(
        data_config=data_config,
        split="train",
        batch_size=2,
        num_samples=10,
        patch_size=128,
        time_window=8,
        synthetic_mode=True,
        num_workers=0  # 避免多进程问题
    )
    
    return dataloader


def demo_forward_pass():
    """演示前向传播"""
    print("=== Enhanced Earth Foundation Model Demo ===")
    
    # 创建模型和数据
    model = create_demo_model()
    dataloader = create_demo_data()
    
    # 获取一个batch
    batch = next(iter(dataloader))
    
    print(f"Model size: {model.get_model_size()}")
    print(f"Batch size: {batch['coordinates'].shape[0]}")
    print(f"Available modalities: {list(batch['multimodal_data'].keys())}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(
            multimodal_data=batch["multimodal_data"],
            timestamps=batch["timestamps"]["optical"],  # 使用光学数据的时间戳
            coordinates=batch["coordinates"],
            valid_periods=batch["valid_periods"],
            attention_mask=batch["attention_masks"]["optical"]
        )
    
    print(f"\\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value.shape}")
    
    return model, batch, outputs


def demo_embedding_analysis(model, batch, outputs):
    """演示嵌入分析"""
    print("\\n=== Embedding Analysis ===")
    
    embeddings = outputs["embeddings"]  # (B, H, W, 64)
    B, H, W, D = embeddings.shape
    
    # 计算嵌入统计
    embed_flat = embeddings.view(-1, D)
    embed_norms = torch.norm(embed_flat, p=2, dim=-1)
    
    print(f"Embedding statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Norm mean: {embed_norms.mean():.4f}")
    print(f"  Norm std: {embed_norms.std():.4f}")
    print(f"  Min norm: {embed_norms.min():.4f}")
    print(f"  Max norm: {embed_norms.max():.4f}")
    
    # 检查是否是单位向量
    unit_vector_check = torch.allclose(embed_norms, torch.ones_like(embed_norms), atol=1e-3)
    print(f"  Unit vector check: {unit_vector_check}")
    
    # 计算嵌入多样性
    pairwise_sim = torch.mm(embed_flat, embed_flat.t())
    mean_similarity = pairwise_sim.mean()
    print(f"  Mean pairwise similarity: {mean_similarity:.4f}")
    
    return embeddings


def demo_reconstruction(model, batch, outputs):
    """演示重建功能"""
    print("\\n=== Reconstruction Demo ===")
    
    if "reconstructions" not in outputs:
        print("Reconstruction not available in this model configuration")
        return
    
    reconstructions = outputs["reconstructions"]
    original_data = batch["multimodal_data"]
    
    for modality in reconstructions.keys():
        if modality in original_data:
            recon = reconstructions[modality]
            orig = original_data[modality]
            
            # 计算重建误差
            mse = torch.mean((recon - orig) ** 2)
            mae = torch.mean(torch.abs(recon - orig))
            
            print(f"  {modality} reconstruction:")
            print(f"    MSE: {mse:.6f}")
            print(f"    MAE: {mae:.6f}")
            print(f"    Original range: [{orig.min():.3f}, {orig.max():.3f}]")
            print(f"    Reconstructed range: [{recon.min():.3f}, {recon.max():.3f}]")


def demo_text_alignment(model, batch, outputs):
    """演示文本对齐功能"""
    print("\\n=== Text Alignment Demo ===")
    
    if not model.enable_text_alignment:
        print("Text alignment not enabled in this model")
        return
    
    # 创建示例文本描述
    text_descriptions = [
        "Satellite imagery of temperate forest region with mixed vegetation",
        "Agricultural landscape with crop fields and scattered settlements"
    ]
    
    print(f"Text descriptions:")
    for i, desc in enumerate(text_descriptions):
        print(f"  {i}: {desc}")
    
    # TODO: 实现文本tokenization和编码
    # 这里需要实际的文本处理管道
    print("Text encoding functionality needs tokenizer implementation")


def demo_temporal_interpolation(model, batch, outputs):
    """演示时间插值功能"""
    print("\\n=== Temporal Interpolation Demo ===")
    
    # 获取原始时间戳
    original_timestamps = batch["timestamps"]["optical"]  # (B, T)
    B, T = original_timestamps.shape
    
    # 创建新的目标时间戳 (在原始时间戳之间)
    target_timestamps = torch.zeros(B, T * 2 - 1)
    for b in range(B):
        orig_ts = original_timestamps[b]
        # 在相邻时间戳之间插值
        for t in range(T - 1):
            target_timestamps[b, t * 2] = orig_ts[t]
            target_timestamps[b, t * 2 + 1] = (orig_ts[t] + orig_ts[t + 1]) / 2
        target_timestamps[b, -1] = orig_ts[-1]
    
    print(f"Original timestamps shape: {original_timestamps.shape}")
    print(f"Target timestamps shape: {target_timestamps.shape}")
    
    # TODO: 实现时间插值
    print("Temporal interpolation functionality available in TemporalSummarizer")


def demo_scaling_analysis(model):
    """演示模型缩放分析"""
    print("\\n=== Model Scaling Analysis ===")
    
    model_stats = model.get_model_size()
    
    print(f"Model statistics:")
    for key, value in model_stats.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    # 估算内存使用
    param_memory = model_stats["total_parameters"] * 4 / (1024**3)  # GB (float32)
    print(f"  Estimated parameter memory: {param_memory:.2f} GB")
    
    # 估算训练内存 (参数 + 梯度 + 优化器状态)
    training_memory = param_memory * 4  # 近似估算
    print(f"  Estimated training memory: {training_memory:.2f} GB")


def demo_performance_benchmark():
    """性能基准测试"""
    print("\\n=== Performance Benchmark ===")
    
    model = create_demo_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建测试输入
    batch_size = 4
    test_input = {
        "multimodal_data": {
            "optical": torch.randn(batch_size, 8, 128, 128, 6).to(device),
            "sar": torch.randn(batch_size, 8, 128, 128, 2).to(device),
            "environmental": torch.randn(batch_size, 8, 128, 128, 4).to(device)
        },
        "timestamps": torch.randn(batch_size, 8).to(device),
        "coordinates": torch.randn(batch_size, 2).to(device),
        "valid_periods": torch.randn(batch_size, 2).to(device)
    }
    
    # 预热
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(**test_input)
    
    # 性能测试
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(**test_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    throughput = batch_size / avg_time
    
    print(f"Performance metrics:")
    print(f"  Average inference time: {avg_time:.3f} seconds")
    print(f"  Throughput: {throughput:.2f} samples/second")
    print(f"  Device: {device}")
    
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Peak GPU memory: {memory_used:.2f} GB")


def main_demo():
    """主演示函数"""
    print("Enhanced Earth Foundation Model Demo")
    print("=" * 50)
    
    # 基本前向传播演示
    model, batch, outputs = demo_forward_pass()
    
    # 嵌入分析
    embeddings = demo_embedding_analysis(model, batch, outputs)
    
    # 重建演示
    demo_reconstruction(model, batch, outputs)
    
    # 文本对齐演示
    demo_text_alignment(model, batch, outputs)
    
    # 时间插值演示
    demo_temporal_interpolation(model, batch, outputs)
    
    # 模型缩放分析
    demo_scaling_analysis(model)
    
    # 性能基准测试
    import time
    demo_performance_benchmark()
    
    print("\\n=== Demo Completed ===")
    print("To start training, run:")
    print("python scripts/train.py --config configs/global_model.yaml")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        main()
    else:
        main_demo()