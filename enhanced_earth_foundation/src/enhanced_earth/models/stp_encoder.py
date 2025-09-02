"""
Enhanced Space-Time-Precision (STP) Encoder

改进的STP编码器，相比原始AlphaEarth增加了：
1. 更强的跨尺度信息交换
2. 自适应注意力机制
3. 多模态信息融合
4. 更好的时间建模
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
from einops import rearrange, repeat

from .backbone import EnhancedTransformer
from .stp_operators import SpaceOperator, TimeOperator, PrecisionOperator
from .pyramid_exchange import LearnedPyramidExchange


class EnhancedSTPBlock(nn.Module):
    """增强的STP块，增加了自适应融合和跨模态交换"""
    
    def __init__(
        self,
        space_dim: int = 1024,
        time_dim: int = 512, 
        precision_dim: int = 256,
        num_heads: int = 16,
        dropout: float = 0.1,
        use_adaptive_fusion: bool = True
    ):
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        self.use_adaptive_fusion = use_adaptive_fusion
        
        # 三个操作符
        self.space_op = SpaceOperator(space_dim, num_heads, dropout)
        self.time_op = TimeOperator(time_dim, num_heads, dropout)
        self.precision_op = PrecisionOperator(precision_dim, dropout)
        
        # 改进的金字塔交换
        self.pyramid_exchange = LearnedPyramidExchange(
            space_dim=space_dim,
            time_dim=time_dim,
            precision_dim=precision_dim,
            use_learned_resampling=True
        )
        
        # 自适应融合模块
        if use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveFusionModule(
                dims=[space_dim, time_dim, precision_dim],
                num_heads=num_heads,
                dropout=dropout
            )
        
        # 输出标准化
        self.space_norm = nn.LayerNorm(space_dim)
        self.time_norm = nn.LayerNorm(time_dim)
        self.precision_norm = nn.LayerNorm(precision_dim)
    
    def forward(
        self,
        space_x: torch.Tensor,
        time_x: torch.Tensor, 
        precision_x: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            space_x: (B, T, H_s, W_s, space_dim) 空间特征
            time_x: (B, T, H_t, W_t, time_dim) 时间特征
            precision_x: (B, T, H_p, W_p, precision_dim) 精度特征
            timestamps: (B, T) 时间戳
            attention_mask: (B, T) 注意力掩码
        """
        # 1. 应用各自的操作符
        space_processed = self.space_op(space_x, attention_mask)
        time_processed = self.time_op(time_x, timestamps, attention_mask)
        precision_processed = self.precision_op(precision_x)
        
        # 2. 金字塔信息交换
        space_exchanged, time_exchanged, precision_exchanged = self.pyramid_exchange(
            space_processed, time_processed, precision_processed
        )
        
        # 3. 自适应融合 (如果启用)
        if self.use_adaptive_fusion:
            space_fused, time_fused, precision_fused = self.adaptive_fusion(
                space_exchanged, time_exchanged, precision_exchanged, timestamps
            )
        else:
            space_fused = space_exchanged
            time_fused = time_exchanged  
            precision_fused = precision_exchanged
        
        # 4. 输出标准化
        space_out = self.space_norm(space_fused)
        time_out = self.time_norm(time_fused)
        precision_out = self.precision_norm(precision_fused)
        
        return space_out, time_out, precision_out


class AdaptiveFusionModule(nn.Module):
    """自适应融合模块，学习不同尺度特征的最优组合"""
    
    def __init__(
        self,
        dims: List[int],
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        
        # 跨尺度注意力
        self.cross_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=min(num_heads, dim // 64),
                dropout=dropout,
                batch_first=True
            ) for dim in dims
        ])
        
        # 时间条件门控
        self.temporal_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, dim),  # 时间编码维度
                nn.Sigmoid()
            ) for dim in dims
        ])
        
        # 融合权重学习
        self.fusion_weights = nn.Sequential(
            nn.Linear(sum(dims), sum(dims) // 2),
            nn.GELU(),
            nn.Linear(sum(dims) // 2, len(dims)),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        precision_x: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """自适应融合三个尺度的特征"""
        B, T = timestamps.shape
        
        # 时间编码 (简单的正弦编码)
        time_enc = self._encode_time(timestamps)  # (B, T, 64)
        
        features = [space_x, time_x, precision_x]
        fused_features = []
        
        for i, (feat, gate, attn) in enumerate(zip(
            features, self.temporal_gates, self.cross_scale_attention
        )):
            # 时间门控
            time_gate = gate(time_enc)  # (B, T, dim)
            gated_feat = feat * time_gate.unsqueeze(-2).unsqueeze(-2)
            
            # 跨尺度自注意力 (在时间维度上)
            B, T, H, W, D = gated_feat.shape
            feat_seq = rearrange(gated_feat, 'b t h w d -> (b h w) t d')
            
            attended_feat, _ = attn(feat_seq, feat_seq, feat_seq)
            attended_feat = rearrange(attended_feat, '(b h w) t d -> b t h w d',
                                    b=B, h=H, w=W)
            
            fused_features.append(attended_feat)
        
        return tuple(fused_features)
    
    def _encode_time(self, timestamps: torch.Tensor, dim: int = 64) -> torch.Tensor:
        """简单的时间正弦编码"""
        B, T = timestamps.shape
        
        # 标准化时间戳到[0, 1]
        t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        
        # 正弦位置编码
        position = t_norm.unsqueeze(-1)  # (B, T, 1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=timestamps.device) * 
                           -(math.log(10000.0) / dim))
        
        pos_enc = torch.zeros(B, T, dim, device=timestamps.device)
        pos_enc[:, :, 0::2] = torch.sin(position * div_term)
        pos_enc[:, :, 1::2] = torch.cos(position * div_term)
        
        return pos_enc


class EnhancedSTPEncoder(nn.Module):
    """
    增强的STP编码器
    
    主要改进:
    1. 可配置的层数和维度
    2. 残差连接和梯度检查点
    3. 自适应融合机制
    4. 更好的正则化
    """
    
    def __init__(
        self,
        space_dim: int = 1024,
        time_dim: int = 512,
        precision_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
        use_adaptive_fusion: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # STP块堆叠
        self.stp_blocks = nn.ModuleList([
            EnhancedSTPBlock(
                space_dim=space_dim,
                time_dim=time_dim,
                precision_dim=precision_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_adaptive_fusion=use_adaptive_fusion
            ) for _ in range(num_layers)
        ])
        
        # 层间残差连接
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.1) for _ in range(num_layers)
        ])
        
        # 最终输出投影到统一维度
        self.output_projection = nn.Linear(space_dim, space_dim)
        self.output_norm = nn.LayerNorm(space_dim)
    
    def forward(
        self,
        features: torch.Tensor,  # (B, T, H, W, d_model)
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, H, W, d_model) 输入特征
            timestamps: (B, T) 时间戳
            attention_mask: (B, T) 注意力掩码
        Returns:
            (B, T, H, W, d_model) 处理后的特征
        """
        B, T, H, W, D = features.shape
        
        # 初始化三个尺度的特征
        # Space: 1/16 分辨率
        space_features = F.interpolate(
            rearrange(features, 'b t h w d -> (b t) d h w'),
            scale_factor=1/16, mode='bilinear', align_corners=False
        )
        space_features = rearrange(space_features, '(b t) d h w -> b t h w d', b=B, t=T)
        space_features = F.linear(space_features, 
                                 torch.eye(D, self.stp_blocks[0].space_dim, device=features.device))
        
        # Time: 1/8 分辨率  
        time_features = F.interpolate(
            rearrange(features, 'b t h w d -> (b t) d h w'),
            scale_factor=1/8, mode='bilinear', align_corners=False
        )
        time_features = rearrange(time_features, '(b t) d h w -> b t h w d', b=B, t=T)
        time_features = F.linear(time_features,
                               torch.eye(D, self.stp_blocks[0].time_dim, device=features.device))
        
        # Precision: 1/2 分辨率
        precision_features = F.interpolate(
            rearrange(features, 'b t h w d -> (b t) d h w'),
            scale_factor=1/2, mode='bilinear', align_corners=False
        )
        precision_features = rearrange(precision_features, '(b t) d h w -> b t h w d', b=B, t=T)
        precision_features = F.linear(precision_features,
                                    torch.eye(D, self.stp_blocks[0].precision_dim, device=features.device))
        
        # 逐层处理
        for i, (stp_block, layer_scale) in enumerate(zip(self.stp_blocks, self.layer_scales)):
            
            # 保存残差
            space_residual = space_features
            time_residual = time_features
            precision_residual = precision_features
            
            # STP处理
            if self.use_gradient_checkpointing and self.training:
                space_out, time_out, precision_out = torch.utils.checkpoint.checkpoint(
                    stp_block, space_features, time_features, precision_features, timestamps, attention_mask
                )
            else:
                space_out, time_out, precision_out = stp_block(
                    space_features, time_features, precision_features, timestamps, attention_mask
                )
            
            # 残差连接和层缩放
            space_features = space_residual + layer_scale * space_out
            time_features = time_residual + layer_scale * time_out
            precision_features = precision_residual + layer_scale * precision_out
        
        # 将空间特征上采样回原始分辨率作为主要输出
        space_output = F.interpolate(
            rearrange(space_features, 'b t h w d -> (b t) d h w'),
            size=(H, W), mode='bilinear', align_corners=False
        )
        space_output = rearrange(space_output, '(b t) d h w -> b t h w d', b=B, t=T)
        
        # 投影到目标维度
        output = self.output_projection(space_output)
        output = self.output_norm(output)
        
        return output


class EnhancedSTPEncoder(nn.Module):
    """
    增强的STP编码器主类
    
    包含多个STP块的堆叠，支持不同的配置和优化策略
    """
    
    def __init__(
        self,
        space_dim: int = 1024,
        time_dim: int = 512,
        precision_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
        use_adaptive_fusion: bool = True,
        layer_drop_rate: float = 0.0  # 随机层丢弃率
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.layer_drop_rate = layer_drop_rate
        
        # 输入投影层
        self.input_projections = nn.ModuleDict({
            "space": nn.Linear(space_dim, space_dim),
            "time": nn.Linear(time_dim, time_dim),
            "precision": nn.Linear(precision_dim, precision_dim)
        })
        
        # STP块堆叠
        self.stp_blocks = nn.ModuleList([
            EnhancedSTPBlock(
                space_dim=space_dim,
                time_dim=time_dim,
                precision_dim=precision_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_adaptive_fusion=use_adaptive_fusion
            ) for _ in range(num_layers)
        ])
        
        # 层间归一化
        self.layer_norms = nn.ModuleList([
            nn.ModuleDict({
                "space": nn.LayerNorm(space_dim),
                "time": nn.LayerNorm(time_dim), 
                "precision": nn.LayerNorm(precision_dim)
            }) for _ in range(num_layers)
        ])
        
        # 最终融合层
        self.final_fusion = FinalScaleFusion(
            space_dim=space_dim,
            time_dim=time_dim,
            precision_dim=precision_dim,
            output_dim=space_dim,
            num_heads=num_heads
        )
    
    def forward(
        self,
        features: torch.Tensor,  # (B, T, H, W, d_model)
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        主要前向传播
        
        Args:
            features: (B, T, H, W, d_model) 多模态融合后的特征
            timestamps: (B, T) 时间戳
            attention_mask: (B, T) 注意力掩码
        Returns:
            (B, T, H, W, d_model) 经过STP处理的特征
        """
        B, T, H, W, D = features.shape
        
        # 初始化不同尺度的特征表示
        space_features, time_features, precision_features = self._initialize_multiscale_features(
            features
        )
        
        # 逐层STP处理
        for i, (stp_block, layer_norm) in enumerate(zip(self.stp_blocks, self.layer_norms)):
            
            # 随机层丢弃 (训练时)
            if self.training and self.layer_drop_rate > 0:
                if torch.rand(1).item() < self.layer_drop_rate:
                    continue
            
            # 层前归一化
            space_normed = layer_norm["space"](space_features)
            time_normed = layer_norm["time"](time_features)
            precision_normed = layer_norm["precision"](precision_features)
            
            # STP处理
            if self.use_gradient_checkpointing and self.training:
                space_out, time_out, precision_out = torch.utils.checkpoint.checkpoint(
                    stp_block, space_normed, time_normed, precision_normed, 
                    timestamps, attention_mask
                )
            else:
                space_out, time_out, precision_out = stp_block(
                    space_normed, time_normed, precision_normed, 
                    timestamps, attention_mask
                )
            
            # 残差连接
            space_features = space_features + space_out
            time_features = time_features + time_out
            precision_features = precision_features + precision_out
        
        # 最终多尺度融合
        final_output = self.final_fusion(
            space_features, time_features, precision_features
        )
        
        return final_output
    
    def _initialize_multiscale_features(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始化多尺度特征表示"""
        B, T, H, W, D = features.shape
        
        # 空间尺度: 1/16 分辨率
        space_h, space_w = H // 16, W // 16
        space_features = F.interpolate(
            rearrange(features, 'b t h w d -> (b t) d h w'),
            size=(space_h, space_w), mode='bilinear', align_corners=False
        )
        space_features = rearrange(space_features, '(b t) d h w -> b t h w d', b=B, t=T)
        space_features = self.input_projections["space"](space_features)
        
        # 时间尺度: 1/8 分辨率
        time_h, time_w = H // 8, W // 8
        time_features = F.interpolate(
            rearrange(features, 'b t h w d -> (b t) d h w'),
            size=(time_h, time_w), mode='bilinear', align_corners=False
        )
        time_features = rearrange(time_features, '(b t) d h w -> b t h w d', b=B, t=T)
        time_features = self.input_projections["time"](time_features)
        
        # 精度尺度: 1/2 分辨率
        precision_h, precision_w = H // 2, W // 2
        precision_features = F.interpolate(
            rearrange(features, 'b t h w d -> (b t) d h w'),
            size=(precision_h, precision_w), mode='bilinear', align_corners=False
        )
        precision_features = rearrange(precision_features, '(b t) d h w -> b t h w d', b=B, t=T)
        precision_features = self.input_projections["precision"](precision_features)
        
        return space_features, time_features, precision_features


class FinalScaleFusion(nn.Module):
    """最终多尺度融合模块"""
    
    def __init__(
        self,
        space_dim: int,
        time_dim: int,
        precision_dim: int,
        output_dim: int,
        num_heads: int = 8
    ):
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        self.output_dim = output_dim
        
        # 尺度特定投影
        self.scale_projections = nn.ModuleDict({
            "space": nn.Linear(space_dim, output_dim),
            "time": nn.Linear(time_dim, output_dim),
            "precision": nn.Linear(precision_dim, output_dim)
        })
        
        # 注意力融合
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 最终投影
        self.final_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(
        self,
        space_features: torch.Tensor,
        time_features: torch.Tensor, 
        precision_features: torch.Tensor
    ) -> torch.Tensor:
        """融合三个尺度的特征到统一输出"""
        B, T, _, _, _ = space_features.shape
        
        # 将所有特征上采样到相同的空间分辨率 (以space为准)
        target_h, target_w = space_features.shape[2:4]
        
        # 上采样time和precision特征
        time_upsampled = F.interpolate(
            rearrange(time_features, 'b t h w d -> (b t) d h w'),
            size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        time_upsampled = rearrange(time_upsampled, '(b t) d h w -> b t h w d', b=B, t=T)
        
        precision_upsampled = F.interpolate(
            rearrange(precision_features, 'b t h w d -> (b t) d h w'),
            size=(target_h, target_w), mode='bilinear', align_corners=False
        )
        precision_upsampled = rearrange(precision_upsampled, '(b t) d h w -> b t h w d', b=B, t=T)
        
        # 投影到统一维度
        space_proj = self.scale_projections["space"](space_features)
        time_proj = self.scale_projections["time"](time_upsampled)
        precision_proj = self.scale_projections["precision"](precision_upsampled)
        
        # 堆叠为序列进行注意力融合
        stacked_features = torch.stack([space_proj, time_proj, precision_proj], dim=2)  # (B, T, 3, H, W, D)
        seq_features = rearrange(stacked_features, 'b t s h w d -> (b t h w) s d')
        
        # 自注意力融合
        fused_seq, _ = self.fusion_attention(seq_features, seq_features, seq_features)
        
        # 平均池化融合多尺度信息
        fused_features = fused_seq.mean(dim=1)  # (BTHW, D)
        
        # 最终投影
        output = self.final_projection(fused_features)
        
        # 重塑回原始格式
        return rearrange(output, '(b t h w) d -> b t h w d', 
                        b=B, t=T, h=target_h, w=target_w)