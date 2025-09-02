#!/usr/bin/env python3
"""
Quick Demo for Enhanced Earth Foundation Model

简化的演示脚本，展示模型的基本功能。
"""

import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import numpy as np
from typing import Dict, Any

# 由于某些模块可能有导入问题，我们创建一个简化版本
print("Enhanced Earth Foundation Model - Quick Demo")
print("=" * 50)

def create_simple_model():
    """创建简化的演示模型"""
    class SimpleEnhancedEarthModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            # 简化的多模态编码器
            self.optical_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(6, 64, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d(16)
            )
            
            self.sar_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(2, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 128, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d(16)
            )
            
            # 时间编码器
            self.temporal_encoder = torch.nn.LSTM(128, 128, batch_first=True)
            
            # 最终投影到球面嵌入
            self.final_projection = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64)
            )
        
        def forward(self, optical_data, sar_data, timestamps):
            B, T, H, W = optical_data.shape[:4]
            
            # 编码光学数据
            optical_flat = optical_data.view(B * T, 6, H, W)
            optical_encoded = self.optical_encoder(optical_flat)  # (BT, 128, 16, 16)
            optical_pooled = optical_encoded.mean(dim=(2, 3))  # (BT, 128)
            optical_temporal = optical_pooled.view(B, T, 128)
            
            # 编码SAR数据
            sar_flat = sar_data.view(B * T, 2, H, W)
            sar_encoded = self.sar_encoder(sar_flat)  # (BT, 128, 16, 16)
            sar_pooled = sar_encoded.mean(dim=(2, 3))  # (BT, 128)
            sar_temporal = sar_pooled.view(B, T, 128)
            
            # 融合多模态特征
            fused_features = (optical_temporal + sar_temporal) / 2
            
            # 时间编码
            temporal_out, _ = self.temporal_encoder(fused_features)
            
            # 时间汇聚 (取最后一个时间步)
            final_features = temporal_out[:, -1, :]  # (B, 128)
            
            # 投影到球面嵌入
            embeddings = self.final_projection(final_features)  # (B, 64)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            
            return {
                "embeddings": embeddings,
                "temporal_features": temporal_out
            }
    
    return SimpleEnhancedEarthModel()


def create_demo_data():
    """创建演示数据"""
    batch_size = 2
    time_steps = 8
    height, width = 64, 64
    
    # 模拟多模态数据
    optical_data = torch.randn(batch_size, time_steps, height, width, 6) * 0.3 + 0.5
    sar_data = torch.randn(batch_size, time_steps, height, width, 2) * 5 - 10
    
    # 时间戳
    timestamps = torch.linspace(0, 365*24*3600*1000, time_steps).unsqueeze(0).repeat(batch_size, 1)
    
    # 地理坐标
    coordinates = torch.tensor([[45.0, 2.0], [35.0, -120.0]])  # 巴黎和加州
    
    # 有效时间段
    valid_periods = torch.tensor([
        [timestamps[0, 0], timestamps[0, -1]],
        [timestamps[1, 0], timestamps[1, -1]]
    ])
    
    return {
        "optical": optical_data,
        "sar": sar_data,
        "timestamps": timestamps,
        "coordinates": coordinates,
        "valid_periods": valid_periods
    }


def demo_forward_pass():
    """演示前向传播"""
    print("1. Creating model...")
    model = create_simple_model()
    
    print("2. Creating demo data...")
    data = create_demo_data()
    
    print("3. Running forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            optical_data=data["optical"],
            sar_data=data["sar"],
            timestamps=data["timestamps"]
        )
    
    print("4. Results:")
    embeddings = outputs["embeddings"]
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embedding norms: {torch.norm(embeddings, p=2, dim=-1)}")
    print(f"   Mean embedding: {embeddings.mean(dim=0)[:5]}...")  # 显示前5个维度
    
    return model, data, outputs


def demo_similarity_analysis(embeddings):
    """演示相似度分析"""
    print("\\n5. Similarity Analysis:")
    
    # 计算样本间相似度
    similarity = torch.mm(embeddings, embeddings.t())
    print(f"   Pairwise similarity matrix:")
    print(f"   {similarity}")
    
    # 分析嵌入分布
    mean_emb = embeddings.mean(dim=0)
    std_emb = embeddings.std(dim=0)
    print(f"   Embedding statistics:")
    print(f"     Mean magnitude: {torch.norm(mean_emb):.4f}")
    print(f"     Std magnitude: {torch.norm(std_emb):.4f}")


def demo_model_info(model):
    """演示模型信息"""
    print("\\n6. Model Information:")
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    
    # 显示模型结构
    print(f"   Model architecture:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"     {name}: {params:,} params")


def main():
    """主演示函数"""
    try:
        # 运行演示
        model, data, outputs = demo_forward_pass()
        
        # 相似度分析
        demo_similarity_analysis(outputs["embeddings"])
        
        # 模型信息
        demo_model_info(model)
        
        print("\\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\\nNext steps:")
        print("1. Install full dependencies: pip install -e .")
        print("2. Run full demo: python scripts/demo.py")
        print("3. Start training: python scripts/train.py --config configs/global_model.yaml")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        print("\\nThis is likely due to missing dependencies.")
        print("Please install the full requirements:")
        print("pip install -r requirements.txt")
        print("pip install -e .")


if __name__ == "__main__":
    main()