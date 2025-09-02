"""
Temporal Summarizer

改进的时间汇聚模块，支持连续时间和灵活的汇聚策略。
相比原始AlphaEarth，增加了更强的时间建模能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math
from einops import rearrange, repeat

from .backbone import EnhancedAttention, RMSNorm


class ContinuousTimeEncoder(nn.Module):
    """连续时间编码器，支持任意时间间隔的插值"""
    
    def __init__(self, dim: int, max_time_range: float = 365 * 24 * 3600 * 1000):  # 1年的毫秒数
        super().__init__()
        self.dim = dim
        self.max_time_range = max_time_range
        
        # 多尺度时间编码
        self.time_scales = [1, 24, 24*7, 24*30, 24*365]  # 小时、天、周、月、年
        
        self.time_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, dim // len(self.time_scales)),
                nn.GELU(),
                nn.Linear(dim // len(self.time_scales), dim // len(self.time_scales))
            ) for _ in self.time_scales
        ])
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, T) 时间戳 (毫秒)
        Returns:
            (B, T, dim) 连续时间编码
        """
        B, T = timestamps.shape
        
        # 标准化时间戳
        t_norm = timestamps / self.max_time_range  # 归一化到[0, 1]
        
        # 多尺度时间编码
        time_encodings = []
        for scale, encoder in zip(self.time_scales, self.time_encoders):
            t_scaled = (t_norm * scale).unsqueeze(-1)  # (B, T, 1)
            encoded = encoder(t_scaled)  # (B, T, dim//num_scales)
            time_encodings.append(encoded)
        
        # 拼接多尺度编码
        full_encoding = torch.cat(time_encodings, dim=-1)  # (B, T, dim)
        
        # 融合处理
        return self.fusion(full_encoding)


class SummaryPeriodEncoder(nn.Module):
    """汇总时期编码器，生成时间查询向量"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # 时期编码器
        self.period_encoder = nn.Sequential(
            nn.Linear(2, dim),  # [start_time, end_time]
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
        # 可学习的查询生成
        self.query_generator = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, valid_periods: torch.Tensor) -> torch.Tensor:
        """
        Args:
            valid_periods: (B, 2) [start_time, end_time] 有效时间段
        Returns:
            (B, dim) 时间查询向量
        """
        # 编码时间段
        period_encoded = self.period_encoder(valid_periods)
        
        # 生成查询向量
        query = self.query_generator(period_encoded)
        
        return query


class AdaptiveTimePooling(nn.Module):
    """
    自适应时间池化
    
    使用注意力机制在时间维度上进行汇聚，
    支持不同的汇聚策略和时间权重学习。
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        pooling_strategy: str = "attention",  # "attention", "weighted_avg", "transformer"
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.pooling_strategy = pooling_strategy
        
        if pooling_strategy == "attention":
            # 单查询多头注意力
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        elif pooling_strategy == "weighted_avg":
            # 学习时间权重
            self.time_weights = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Linear(dim // 2, 1),
                nn.Sigmoid()
            )
        elif pooling_strategy == "transformer":
            # 使用transformer进行时间汇聚
            self.time_transformer = EnhancedAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.pooling_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        self.output_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        features: torch.Tensor,  # (B, T, H, W, dim)
        query: torch.Tensor,     # (B, dim) 查询向量
        mask: Optional[torch.Tensor] = None  # (B, T) 时间掩码
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, H, W, dim) 时序特征
            query: (B, dim) 时间查询向量
            mask: (B, T) 时间掩码
        Returns:
            (B, H, W, dim) 汇聚后的特征
        """
        B, T, H, W, D = features.shape
        
        if self.pooling_strategy == "attention":
            # 单查询注意力池化
            # 重塑为 (B*H*W, T, dim)
            feat_seq = rearrange(features, 'b t h w d -> (b h w) t d')
            
            # 查询向量扩展到所有空间位置
            queries = repeat(query, 'b d -> (b h w) 1 d', h=H, w=W)
            
            # 时间掩码处理
            if mask is not None:
                time_mask = repeat(mask, 'b t -> (b h w) t', h=H, w=W)
            else:
                time_mask = None
            
            # 注意力池化
            pooled_feat, _ = self.attention_pooling(
                query=queries,
                key=feat_seq,
                value=feat_seq,
                key_padding_mask=~time_mask if time_mask is not None else None
            )  # (BHW, 1, dim)
            
            pooled_feat = pooled_feat.squeeze(1)  # (BHW, dim)
            
        elif self.pooling_strategy == "weighted_avg":
            # 加权平均池化
            feat_seq = rearrange(features, 'b t h w d -> (b h w) t d')
            
            # 计算时间权重
            time_weights = self.time_weights(feat_seq)  # (BHW, T, 1)
            
            # 应用掩码
            if mask is not None:
                mask_expanded = repeat(mask, 'b t -> (b h w) t 1', h=H, w=W)
                time_weights = time_weights * mask_expanded
            
            # 标准化权重
            time_weights = F.softmax(time_weights, dim=1)
            
            # 加权汇聚
            pooled_feat = (feat_seq * time_weights).sum(dim=1)  # (BHW, dim)
            
        elif self.pooling_strategy == "transformer":
            # Transformer池化
            feat_seq = rearrange(features, 'b t h w d -> (b h w) t d')
            
            # 添加池化token
            pooling_tokens = repeat(self.pooling_token, '1 1 d -> bhw 1 d', bhw=B*H*W)
            feat_with_token = torch.cat([pooling_tokens, feat_seq], dim=1)  # (BHW, T+1, dim)
            
            # Transformer处理
            processed = self.time_transformer(feat_with_token)
            pooled_feat = processed[:, 0]  # 取池化token的输出
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # 输出投影
        pooled_feat = self.output_projection(pooled_feat)
        pooled_feat = self.dropout(pooled_feat)
        
        # 重塑为空间格式
        return rearrange(pooled_feat, '(b h w) d -> b h w d', b=B, h=H, w=W)


class TemporalSummarizer(nn.Module):
    """
    时间汇聚器主类
    
    整合连续时间编码、自适应池化和球面嵌入生成
    """
    
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int = 64,
        num_heads: int = 8,
        pooling_strategy: str = "attention",
        dropout: float = 0.1,
        use_continuous_time: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.use_continuous_time = use_continuous_time
        
        # 连续时间编码器
        if use_continuous_time:
            self.continuous_time_encoder = ContinuousTimeEncoder(feature_dim)
        
        # 汇总时期编码器
        self.summary_encoder = SummaryPeriodEncoder(feature_dim)
        
        # 自适应时间池化
        self.time_pooling = AdaptiveTimePooling(
            dim=feature_dim,
            num_heads=num_heads,
            pooling_strategy=pooling_strategy,
            dropout=dropout
        )
        
        # 球面嵌入投影
        self.sphere_projection = SphereEmbeddingProjection(
            input_dim=feature_dim,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # 可选的时间插值模块
        self.time_interpolator = TemporalInterpolator(feature_dim)
    
    def forward(
        self,
        features: torch.Tensor,      # (B, T, H, W, feature_dim)
        timestamps: torch.Tensor,    # (B, T) 时间戳
        valid_periods: torch.Tensor, # (B, 2) [start, end]
        mask: Optional[torch.Tensor] = None  # (B, T) 掩码
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, H, W, feature_dim) 时序特征
            timestamps: (B, T) 时间戳
            valid_periods: (B, 2) 有效时间段
            mask: (B, T) 时间掩码
        Returns:
            (B, H, W, embed_dim) 球面嵌入
        """
        # 1. 连续时间编码 (如果启用)
        if self.use_continuous_time:
            time_encoded = self.continuous_time_encoder(timestamps)  # (B, T, feature_dim)
            features = features + time_encoded.unsqueeze(-2).unsqueeze(-2)
        
        # 2. 生成汇总查询
        summary_query = self.summary_encoder(valid_periods)  # (B, feature_dim)
        
        # 3. 时间池化
        pooled_features = self.time_pooling(features, summary_query, mask)  # (B, H, W, feature_dim)
        
        # 4. 投影到球面嵌入
        sphere_embeddings = self.sphere_projection(pooled_features)  # (B, H, W, embed_dim)
        
        return sphere_embeddings
    
    def interpolate_time(
        self,
        features: torch.Tensor,
        source_timestamps: torch.Tensor,
        target_timestamps: torch.Tensor
    ) -> torch.Tensor:
        """时间插值功能"""
        return self.time_interpolator(features, source_timestamps, target_timestamps)


class SphereEmbeddingProjection(nn.Module):
    """球面嵌入投影，生成单位球面上的嵌入向量"""
    
    def __init__(self, input_dim: int, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 投影网络
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 可选的温度参数学习
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, input_dim) 输入特征
        Returns:
            (B, H, W, embed_dim) L2标准化的球面嵌入
        """
        # 投影到目标维度
        embeddings = self.projection(x)  # (B, H, W, embed_dim)
        
        # L2标准化到单位球面
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # 可选的温度缩放
        embeddings = embeddings / self.temperature
        
        return embeddings


class TemporalInterpolator(nn.Module):
    """时间插值器，支持任意时间点的特征插值"""
    
    def __init__(self, dim: int, interpolation_method: str = "neural"):
        super().__init__()
        self.dim = dim
        self.interpolation_method = interpolation_method
        
        if interpolation_method == "neural":
            # 神经插值网络
            self.interpolation_net = nn.Sequential(
                nn.Linear(dim + 2, dim * 2),  # +2 for time delta encoding
                nn.GELU(),
                nn.Linear(dim * 2, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
        
        # 时间差编码
        self.time_delta_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 2)
        )
    
    def forward(
        self,
        features: torch.Tensor,        # (B, T, H, W, dim)
        source_timestamps: torch.Tensor,  # (B, T) 
        target_timestamps: torch.Tensor   # (B, T_target)
    ) -> torch.Tensor:
        """
        在目标时间点插值特征
        
        Args:
            features: 源时间点的特征
            source_timestamps: 源时间戳
            target_timestamps: 目标时间戳
        Returns:
            (B, T_target, H, W, dim) 插值后的特征
        """
        B, T_src, H, W, D = features.shape
        T_target = target_timestamps.shape[1]
        
        if self.interpolation_method == "linear":
            # 简单线性插值
            interpolated = self._linear_interpolate(
                features, source_timestamps, target_timestamps
            )
        elif self.interpolation_method == "neural":
            # 神经网络插值
            interpolated = self._neural_interpolate(
                features, source_timestamps, target_timestamps
            )
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
        
        return interpolated
    
    def _linear_interpolate(
        self,
        features: torch.Tensor,
        source_timestamps: torch.Tensor,
        target_timestamps: torch.Tensor
    ) -> torch.Tensor:
        """线性插值实现"""
        # 简化实现：使用最近邻插值
        B, T_src, H, W, D = features.shape
        T_target = target_timestamps.shape[1]
        
        # 为每个目标时间找到最近的源时间
        time_diffs = torch.abs(
            target_timestamps.unsqueeze(-1) - source_timestamps.unsqueeze(1)
        )  # (B, T_target, T_src)
        
        nearest_indices = torch.argmin(time_diffs, dim=-1)  # (B, T_target)
        
        # 根据最近邻索引选择特征
        batch_indices = torch.arange(B, device=features.device).unsqueeze(1)
        interpolated = features[batch_indices, nearest_indices]  # (B, T_target, H, W, D)
        
        return interpolated
    
    def _neural_interpolate(
        self,
        features: torch.Tensor,
        source_timestamps: torch.Tensor,
        target_timestamps: torch.Tensor
    ) -> torch.Tensor:
        """神经网络插值实现"""
        B, T_src, H, W, D = features.shape
        T_target = target_timestamps.shape[1]
        
        interpolated_features = []
        
        for t_idx in range(T_target):
            target_time = target_timestamps[:, t_idx:t_idx+1]  # (B, 1)
            
            # 计算时间差
            time_deltas = target_time.unsqueeze(-1) - source_timestamps.unsqueeze(1)  # (B, 1, T_src)
            time_deltas_norm = time_deltas / (source_timestamps.max() - source_timestamps.min() + 1e-8)
            
            # 编码时间差
            time_delta_enc = self.time_delta_encoder(time_deltas_norm.unsqueeze(-1))  # (B, 1, T_src, 2)
            
            # 为每个源时间点生成插值权重
            interpolation_weights = []
            for src_idx in range(T_src):
                src_feat = features[:, src_idx]  # (B, H, W, D)
                delta_enc = time_delta_enc[:, 0, src_idx]  # (B, 2)
                
                # 拼接特征和时间差编码
                feat_with_time = torch.cat([
                    src_feat, 
                    delta_enc.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1)
                ], dim=-1)  # (B, H, W, D+2)
                
                # 神经插值
                interpolated_feat = self.interpolation_net(feat_with_time)  # (B, H, W, D)
                interpolation_weights.append(interpolated_feat)
            
            # 加权平均
            weights_stack = torch.stack(interpolation_weights, dim=1)  # (B, T_src, H, W, D)
            
            # 计算注意力权重
            attn_weights = F.softmax(
                torch.sum(weights_stack * features, dim=-1), dim=1
            )  # (B, T_src, H, W)
            
            # 加权汇聚
            interpolated_t = torch.sum(
                weights_stack * attn_weights.unsqueeze(-1), dim=1
            )  # (B, H, W, D)
            
            interpolated_features.append(interpolated_t)
        
        return torch.stack(interpolated_features, dim=1)  # (B, T_target, H, W, D)