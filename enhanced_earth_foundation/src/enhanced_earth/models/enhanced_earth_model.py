"""
Enhanced Earth Foundation Model

主模型类，整合了多模态编码、STP处理、时间汇聚和对比学习等功能。
相比原始AlphaEarth，增加了更强的多模态支持和可扩展性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import math
from einops import rearrange, repeat

from .multimodal_encoder import MultiModalEncoder
from .stp_encoder import EnhancedSTPEncoder
from .temporal_summarizer import TemporalSummarizer
from .text_encoder import TextEncoder
from .decoder import VonMisesFisherDecoder
from .position_encoding import GeospatialPositionEncoding


class EnhancedEarthFoundationModel(nn.Module):
    """
    Enhanced Earth Foundation Model
    
    核心改进:
    1. 支持更多模态数据 (光学、雷达、高光谱、激光雷达、文本)
    2. 改进的STP架构，增加跨模态信息交换
    3. 动态嵌入支持任意波段配置
    4. 更强的位置编码和时间建模
    5. 可扩展的模型尺寸 (small/base/large/xlarge)
    """
    
    def __init__(
        self,
        model_size: str = "base",
        input_modalities: Optional[Dict[str, Dict]] = None,
        decode_modalities: Optional[Dict[str, int]] = None,
        embed_dim: int = 64,
        enable_text_alignment: bool = True,
        enable_reconstruction: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        """
        Args:
            model_size: 模型尺寸 ("small", "base", "large", "xlarge")
            input_modalities: 输入模态配置
                {
                    "optical": {"channels": 13, "resolution": 10},  # Sentinel-2 + Landsat
                    "sar": {"channels": 4, "resolution": 10},       # Sentinel-1 VV/VH + coherence
                    "hyperspectral": {"channels": 242, "resolution": 30},  # PRISMA
                    "lidar": {"channels": 3, "resolution": 25},     # GEDI height/cover/quality
                    "environmental": {"channels": 10, "resolution": 1000}  # ERA5-Land variables
                }
            decode_modalities: 需要重建的模态及其通道数
            embed_dim: 最终嵌入维度 (默认64维球面嵌入)
            enable_text_alignment: 是否启用文本对齐
            enable_reconstruction: 是否启用重建任务
        """
        super().__init__()
        
        # 默认多模态配置
        if input_modalities is None:
            input_modalities = {
                "optical": {"channels": 13, "resolution": 10},
                "sar": {"channels": 4, "resolution": 10}, 
                "environmental": {"channels": 8, "resolution": 1000}
            }
        
        if decode_modalities is None:
            decode_modalities = {
                name: config["channels"] 
                for name, config in input_modalities.items()
            }
        
        self.input_modalities = input_modalities
        self.decode_modalities = decode_modalities
        self.embed_dim = embed_dim
        self.enable_text_alignment = enable_text_alignment
        self.enable_reconstruction = enable_reconstruction
        
        # 模型尺寸配置
        size_configs = {
            "small": {"d_model": 512, "num_layers": 6, "num_heads": 8},
            "base": {"d_model": 768, "num_layers": 12, "num_heads": 12}, 
            "large": {"d_model": 1024, "num_layers": 24, "num_heads": 16},
            "xlarge": {"d_model": 1536, "num_layers": 32, "num_heads": 24}
        }
        
        config = size_configs[model_size]
        self.d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        
        # STP维度配置 (相对于d_model的比例)
        self.space_dim = self.d_model
        self.time_dim = self.d_model // 2
        self.precision_dim = self.d_model // 4
        
        # 核心组件
        self.multimodal_encoder = MultiModalEncoder(
            modalities=input_modalities,
            d_model=self.d_model,
            dropout=dropout
        )
        
        self.position_encoding = GeospatialPositionEncoding(
            d_model=self.d_model,
            max_len=10000
        )
        
        self.stp_encoder = EnhancedSTPEncoder(
            space_dim=self.space_dim,
            time_dim=self.time_dim, 
            precision_dim=self.precision_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=dropout
        )
        
        self.temporal_summarizer = TemporalSummarizer(
            feature_dim=self.d_model,
            embed_dim=embed_dim,
            num_heads=self.num_heads
        )
        
        # 可选组件
        if enable_text_alignment:
            self.text_encoder = TextEncoder(
                d_model=self.d_model,
                vocab_size=50000,
                max_length=256
            )
            
        if enable_reconstruction:
            self.decoder = VonMisesFisherDecoder(
                embed_dim=embed_dim,
                decode_modalities=decode_modalities,
                d_model=self.d_model
            )
        
        # 用于对比学习的投影头
        self.embedding_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(
        self,
        multimodal_data: Dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        coordinates: torch.Tensor,  # (B, 2) [lat, lon]
        valid_periods: torch.Tensor,  # (B, 2) [start_time, end_time]
        text_descriptions: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            multimodal_data: 多模态输入数据
                {
                    "optical": (B, T, H, W, C_opt),
                    "sar": (B, T, H, W, C_sar),
                    ...
                }
            timestamps: (B, T) 时间戳
            coordinates: (B, 2) 地理坐标 [lat, lon]
            valid_periods: (B, 2) 有效时间段 [start, end]
            text_descriptions: (B, max_text_len) 可选的文本描述
            attention_mask: (B, T) 注意力掩码
            return_features: 是否返回中间特征
            
        Returns:
            Dict包含:
            - embeddings: (B, H, W, embed_dim) 球面嵌入
            - text_embeddings: (B, embed_dim) 文本嵌入 (如果有文本输入)
            - reconstructions: Dict[str, Tensor] 重建结果 (如果启用重建)
            - features: 中间特征 (如果return_features=True)
        """
        B, T = timestamps.shape
        device = timestamps.device
        
        # 1. 多模态编码
        encoded_features = self.multimodal_encoder(multimodal_data)  # (B, T, H, W, d_model)
        
        # 2. 添加位置和时间编码
        features_with_pos = self.position_encoding(
            encoded_features, timestamps, coordinates
        )
        
        # 3. STP编码处理
        stp_features = self.stp_encoder(
            features_with_pos, timestamps, attention_mask
        )  # (B, T, H, W, d_model)
        
        # 4. 时间汇聚得到最终嵌入
        embeddings = self.temporal_summarizer(
            stp_features, timestamps, valid_periods, attention_mask
        )  # (B, H, W, embed_dim)
        
        # 5. 投影用于对比学习
        projected_embeddings = self.embedding_projection(embeddings)
        
        outputs = {
            "embeddings": embeddings,
            "projected_embeddings": projected_embeddings
        }
        
        # 6. 文本编码 (如果启用)
        if self.enable_text_alignment and text_descriptions is not None:
            text_embeddings = self.text_encoder(text_descriptions)
            outputs["text_embeddings"] = text_embeddings
        
        # 7. 重建任务 (如果启用)
        if self.enable_reconstruction:
            reconstructions = self.decoder(embeddings, multimodal_data)
            outputs["reconstructions"] = reconstructions
        
        # 8. 返回中间特征 (如果需要)
        if return_features:
            outputs["features"] = {
                "encoded": encoded_features,
                "with_position": features_with_pos,
                "stp_processed": stp_features
            }
        
        return outputs
    
    def encode_text(self, text_descriptions: torch.Tensor) -> torch.Tensor:
        """单独编码文本"""
        if not self.enable_text_alignment:
            raise ValueError("Text alignment not enabled")
        return self.text_encoder(text_descriptions)
    
    def encode_multimodal(
        self,
        multimodal_data: Dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        coordinates: torch.Tensor,
        valid_periods: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """单独编码多模态数据，返回嵌入"""
        outputs = self.forward(
            multimodal_data=multimodal_data,
            timestamps=timestamps,
            coordinates=coordinates,
            valid_periods=valid_periods,
            attention_mask=attention_mask,
            return_features=False
        )
        return outputs["embeddings"]
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads
        }