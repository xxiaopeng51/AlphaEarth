"""
多模态编码器

融合Clay的动态嵌入和SatCLIP的多模态处理能力，
支持光学、雷达、高光谱、激光雷达、环境数据等多种模态。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
from einops import rearrange, repeat

from .backbone import EnhancedTransformer
from .dynamic_embedding import DynamicModalityEmbedding


class ModalitySpecificEncoder(nn.Module):
    """模态特定编码器"""
    
    def __init__(
        self,
        modality_name: str,
        input_channels: int,
        d_model: int,
        patch_size: int = 16,
        resolution: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.modality_name = modality_name
        self.input_channels = input_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.resolution = resolution
        
        # 模态特定的预处理
        if modality_name == "optical":
            # 光学数据标准化和增强
            self.preprocess = self._create_optical_preprocess()
        elif modality_name == "sar":
            # SAR数据对数变换和标准化
            self.preprocess = self._create_sar_preprocess()
        elif modality_name == "hyperspectral":
            # 高光谱数据降维和标准化
            self.preprocess = self._create_hyperspectral_preprocess()
        elif modality_name == "lidar":
            # 激光雷达数据处理
            self.preprocess = self._create_lidar_preprocess()
        elif modality_name == "environmental":
            # 环境数据标准化
            self.preprocess = self._create_environmental_preprocess()
        else:
            self.preprocess = nn.Identity()
        
        # 动态嵌入层 (借鉴Clay的设计)
        self.dynamic_embedding = DynamicModalityEmbedding(
            input_channels=input_channels,
            d_model=d_model,
            patch_size=patch_size,
            modality_name=modality_name
        )
        
        # 模态特定的Transformer层
        self.modality_transformer = EnhancedTransformer(
            d_model=d_model,
            num_heads=d_model // 64,
            num_layers=2,  # 浅层模态特定处理
            dropout=dropout
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def _create_optical_preprocess(self) -> nn.Module:
        """光学数据预处理 (Sentinel-2, Landsat)"""
        return nn.Sequential(
            # 标准化到[0,1]范围
            nn.Lambda(lambda x: torch.clamp(x / 10000.0, 0, 1)),
            # 可选的光谱增强
            nn.Conv2d(self.input_channels, self.input_channels, 3, padding=1, groups=self.input_channels),
            nn.BatchNorm2d(self.input_channels),
            nn.GELU()
        )
    
    def _create_sar_preprocess(self) -> nn.Module:
        """SAR数据预处理 (Sentinel-1)"""
        return nn.Sequential(
            # 对数变换减少动态范围
            nn.Lambda(lambda x: torch.log(torch.clamp(x, min=1e-8))),
            # 标准化
            nn.Lambda(lambda x: (x - x.mean(dim=(-2, -1), keepdim=True)) / 
                                (x.std(dim=(-2, -1), keepdim=True) + 1e-8))
        )
    
    def _create_hyperspectral_preprocess(self) -> nn.Module:
        """高光谱数据预处理"""
        return nn.Sequential(
            # PCA降维 (从242维到64维)
            nn.Conv2d(self.input_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 光谱特征增强
            nn.Conv2d(64, self.input_channels, 3, padding=1),
            nn.BatchNorm2d(self.input_channels)
        )
    
    def _create_lidar_preprocess(self) -> nn.Module:
        """激光雷达数据预处理 (GEDI)"""
        return nn.Sequential(
            # 高度数据标准化
            nn.Lambda(lambda x: x / 100.0),  # 假设高度以米为单位
            # 空间插值增强
            nn.Conv2d(self.input_channels, self.input_channels, 5, padding=2),
            nn.BatchNorm2d(self.input_channels),
            nn.ReLU()
        )
    
    def _create_environmental_preprocess(self) -> nn.Module:
        """环境数据预处理 (ERA5-Land等)"""
        return nn.Sequential(
            # 多变量标准化
            nn.Lambda(lambda x: (x - x.mean(dim=(-2, -1), keepdim=True)) / 
                                (x.std(dim=(-2, -1), keepdim=True) + 1e-8))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) 模态数据
        Returns:
            (B, T, H, W, d_model) 编码后的特征
        """
        B, T, H, W, C = x.shape
        
        # 重塑为(BT, C, H, W)进行预处理
        x_reshaped = rearrange(x, 'b t h w c -> (b t) c h w')
        
        # 模态特定预处理
        x_preprocessed = self.preprocess(x_reshaped)
        
        # 重塑回时序格式
        x_temporal = rearrange(x_preprocessed, '(b t) c h w -> b t h w c', b=B, t=T)
        
        # 动态嵌入
        x_embedded = self.dynamic_embedding(x_temporal)  # (B, T, H, W, d_model)
        
        # 模态特定Transformer处理
        # 重塑为序列格式进行attention
        x_seq = rearrange(x_embedded, 'b t h w d -> b (t h w) d')
        x_transformed = self.modality_transformer(x_seq)
        x_output = rearrange(x_transformed, 'b (t h w) d -> b t h w d', t=T, h=H, w=W)
        
        # 输出投影和dropout
        x_output = self.output_projection(x_output)
        x_output = self.dropout(x_output)
        
        return x_output


class MultiModalEncoder(nn.Module):
    """
    多模态编码器
    
    整合多种地球观测数据模态，包括光学、雷达、高光谱等，
    通过注意力机制实现自适应融合。
    """
    
    def __init__(
        self,
        modalities: Dict[str, Dict],
        d_model: int,
        fusion_strategy: str = "attention",
        dropout: float = 0.1
    ):
        super().__init__()
        self.modalities = modalities
        self.d_model = d_model
        self.fusion_strategy = fusion_strategy
        
        # 为每个模态创建编码器
        self.modality_encoders = nn.ModuleDict()
        for name, config in modalities.items():
            self.modality_encoders[name] = ModalitySpecificEncoder(
                modality_name=name,
                input_channels=config["channels"],
                d_model=d_model,
                resolution=config.get("resolution", 10),
                dropout=dropout
            )
        
        # 多模态融合模块
        if fusion_strategy == "attention":
            self.fusion_module = CrossModalAttentionFusion(
                d_model=d_model,
                num_modalities=len(modalities),
                num_heads=d_model // 64,
                dropout=dropout
            )
        elif fusion_strategy == "concat":
            self.fusion_module = ConcatenationFusion(
                d_model=d_model,
                num_modalities=len(modalities)
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # 输出标准化
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(self, multimodal_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multimodal_data: 多模态数据字典
        Returns:
            (B, T, H, W, d_model) 融合后的特征
        """
        # 编码各个模态
        encoded_modalities = {}
        for name, data in multimodal_data.items():
            if name in self.modality_encoders:
                encoded_modalities[name] = self.modality_encoders[name](data)
        
        # 多模态融合
        fused_features = self.fusion_module(encoded_modalities)
        
        # 输出标准化
        fused_features = self.output_norm(fused_features)
        
        return fused_features


class CrossModalAttentionFusion(nn.Module):
    """跨模态注意力融合"""
    
    def __init__(self, d_model: int, num_modalities: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        
        # 模态查询生成器
        self.modality_queries = nn.Parameter(
            torch.randn(num_modalities, d_model) * 0.02
        )
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 融合权重学习
        self.fusion_weights = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, encoded_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            encoded_modalities: {modality_name: (B, T, H, W, d_model)}
        Returns:
            (B, T, H, W, d_model) 融合特征
        """
        modality_names = list(encoded_modalities.keys())
        modality_features = list(encoded_modalities.values())
        
        if not modality_features:
            raise ValueError("No modality data provided")
        
        # 获取形状信息
        B, T, H, W, D = modality_features[0].shape
        
        # 将所有模态特征堆叠
        stacked_features = torch.stack(modality_features, dim=0)  # (M, B, T, H, W, D)
        M = len(modality_features)
        
        # 重塑为序列格式进行注意力计算
        stacked_seq = rearrange(stacked_features, 'm b t h w d -> (b t h w) m d')
        
        # 使用学习的模态查询
        queries = repeat(self.modality_queries[:M], 'm d -> (b t h w) m d', 
                        b=B, t=T, h=H, w=W)
        
        # 跨模态注意力
        attended_features, _ = self.cross_attention(
            query=queries,
            key=stacked_seq,
            value=stacked_seq
        )  # (BTHW, M, D)
        
        # 计算融合权重
        concat_features = rearrange(attended_features, 'bthw m d -> bthw (m d)')
        fusion_weights = self.fusion_weights(concat_features)  # (BTHW, M)
        
        # 加权融合
        weighted_features = torch.einsum('bthw m, bthw m d -> bthw d', 
                                       fusion_weights, attended_features)
        
        # 输出投影
        fused_output = self.output_projection(weighted_features)
        
        # 重塑回原始格式
        return rearrange(fused_output, '(b t h w) d -> b t h w d', 
                        b=B, t=T, h=H, w=W)


class ConcatenationFusion(nn.Module):
    """简单的拼接融合策略"""
    
    def __init__(self, d_model: int, num_modalities: int):
        super().__init__()
        self.projection = nn.Linear(d_model * num_modalities, d_model)
        
    def forward(self, encoded_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """拼接所有模态特征并投影到目标维度"""
        modality_features = list(encoded_modalities.values())
        concatenated = torch.cat(modality_features, dim=-1)
        return self.projection(concatenated)