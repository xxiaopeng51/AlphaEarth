"""
Enhanced STP Operators

改进的Space-Time-Precision操作符，增加了更强的建模能力：
1. 空间操作符: 改进的ViT注意力机制
2. 时间操作符: 增强的时序建模和因果注意力
3. 精度操作符: 多尺度卷积和特征金字塔
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange, repeat

from .backbone import EnhancedAttention, RMSNorm


class SpaceOperator(nn.Module):
    """
    增强的空间操作符
    
    在1/16L分辨率下进行空间自注意力，
    增加了相对位置编码和多尺度感受野。
    """
    
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1,
        use_relative_pos: bool = True,
        window_size: int = 7
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_relative_pos = use_relative_pos
        
        # 归一化层
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # 增强注意力
        self.attention = EnhancedAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=True,
            use_rope=False  # 空间维度使用相对位置编码
        )
        
        # 相对位置编码
        if use_relative_pos:
            self.relative_position_bias = RelativePositionBias(
                window_size=window_size,
                num_heads=num_heads
            )
        
        # 多尺度MLP
        self.mlp = MultiScaleMLP(dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) 空间特征
            attention_mask: (B, T) 时间掩码
        Returns:
            (B, T, H, W, C) 处理后的空间特征
        """
        B, T, H, W, C = x.shape
        
        # 重塑为序列格式
        x_seq = rearrange(x, 'b t h w c -> (b t) (h w) c')
        
        # 空间自注意力
        residual = x_seq
        x_norm = self.norm1(x_seq)
        
        # 应用注意力
        attn_out = self.attention(x_norm)
        
        # 添加相对位置偏置
        if self.use_relative_pos:
            rel_pos_bias = self.relative_position_bias(H, W)  # (H*W, H*W)
            # 这里简化处理，实际中需要更复杂的集成
            attn_out = attn_out + rel_pos_bias.mean() * 0.1
        
        x_seq = residual + self.dropout(attn_out)
        
        # MLP
        x_seq = x_seq + self.dropout(self.mlp(self.norm2(x_seq)))
        
        # 重塑回空间格式
        return rearrange(x_seq, '(b t) (h w) c -> b t h w c', b=B, t=T, h=H, w=W)


class TimeOperator(nn.Module):
    """
    增强的时间操作符
    
    在1/8L分辨率下进行时间轴注意力，
    支持因果掩码和时间位置编码。
    """
    
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        use_causal_mask: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
        
        # 归一化层
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # 时间注意力
        self.temporal_attention = EnhancedAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attn=True,
            use_rope=True  # 时间维度使用RoPE
        )
        
        # 时间位置编码
        self.time_encoding = SinusoidalTimeEncoding(dim, max_seq_len)
        
        # 时间MLP
        self.temporal_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) 时间特征
            timestamps: (B, T) 时间戳
            attention_mask: (B, T) 注意力掩码
        Returns:
            (B, T, H, W, C) 处理后的时间特征
        """
        B, T, H, W, C = x.shape
        
        # 添加时间编码
        time_enc = self.time_encoding(timestamps)  # (B, T, C)
        x = x + time_enc.unsqueeze(-2).unsqueeze(-2)  # 广播到H, W维度
        
        # 重塑为序列格式，对每个空间位置独立处理时间序列
        x_seq = rearrange(x, 'b t h w c -> (b h w) t c')
        
        # 构建因果掩码 (如果需要)
        causal_mask = None
        if self.use_causal_mask:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        # 结合时间掩码和因果掩码
        if attention_mask is not None:
            time_mask = repeat(attention_mask, 'b t -> (b h w) t', h=H, w=W)
            if causal_mask is not None:
                # 将因果掩码广播到batch维度
                causal_mask = repeat(causal_mask, 't1 t2 -> bhw t1 t2', bhw=B*H*W)
                time_mask = time_mask.unsqueeze(-1) & time_mask.unsqueeze(-2) & ~causal_mask
            else:
                time_mask = time_mask.unsqueeze(-1) & time_mask.unsqueeze(-2)
        else:
            time_mask = causal_mask
        
        # 时间自注意力
        residual = x_seq
        x_norm = self.norm1(x_seq)
        
        attn_out = self.temporal_attention(x_norm, time_mask)
        x_seq = residual + self.dropout(attn_out)
        
        # 时间MLP
        x_seq = x_seq + self.dropout(self.temporal_mlp(self.norm2(x_seq)))
        
        # 重塑回空间格式
        return rearrange(x_seq, '(b h w) t c -> b t h w c', b=B, h=H, w=W)


class PrecisionOperator(nn.Module):
    """
    增强的精度操作符
    
    在1/2L分辨率下使用多尺度卷积，
    增加了特征金字塔和注意力引导的卷积。
    """
    
    def __init__(
        self,
        dim: int = 256,
        dropout: float = 0.1,
        kernel_sizes: list = [3, 5, 7],
        use_attention_conv: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        self.use_attention_conv = use_attention_conv
        
        # 归一化层
        num_groups = min(32, dim // 4) if dim >= 32 else 1
        self.norm1 = nn.GroupNorm(num_groups, dim)
        self.norm2 = nn.GroupNorm(num_groups, dim * 2)
        
        # 多尺度卷积分支
        self.multi_scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=k, padding=k//2, groups=dim//4),
                nn.BatchNorm2d(dim),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        
        # 特征融合
        self.scale_fusion = nn.Conv2d(dim * len(kernel_sizes), dim, 1)
        
        # 注意力引导卷积 (如果启用)
        if use_attention_conv:
            self.attention_conv = AttentionGuidedConv2d(dim, dim)
        
        # 主卷积路径
        self.main_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 3, padding=1)
        )
        
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H, W, C) 精度特征
        Returns:
            (B, T, H, W, C) 处理后的精度特征
        """
        B, T, H, W, C = x.shape
        
        # 重塑为卷积格式
        x_conv = rearrange(x, 'b t h w c -> (b t) c h w')
        
        # 多尺度卷积
        multi_scale_outs = []
        for conv_branch in self.multi_scale_convs:
            branch_out = conv_branch(self.norm1(x_conv))
            multi_scale_outs.append(branch_out)
        
        # 融合多尺度特征
        multi_scale_fused = torch.cat(multi_scale_outs, dim=1)
        multi_scale_fused = self.scale_fusion(multi_scale_fused)
        
        # 注意力引导卷积 (如果启用)
        if self.use_attention_conv:
            attention_out = self.attention_conv(x_conv)
            x_conv = x_conv + attention_out
        
        # 主卷积路径
        residual = x_conv
        conv_out = self.main_conv(self.norm1(x_conv))
        
        # 残差连接和多尺度融合
        x_conv = residual + self.dropout(conv_out) + multi_scale_fused * 0.5
        
        # 重塑回时序格式
        return rearrange(x_conv, '(b t) c h w -> b t h w c', b=B, t=T)


class RelativePositionBias(nn.Module):
    """相对位置偏置，用于空间注意力"""
    
    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # 初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, H: int, W: int) -> torch.Tensor:
        """生成相对位置偏置"""
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias


class SinusoidalTimeEncoding(nn.Module):
    """正弦时间编码"""
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, T) 时间戳
        Returns:
            (B, T, dim) 时间编码
        """
        B, T = timestamps.shape
        
        # 标准化时间戳到[0, len(pe)]范围
        t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        t_indices = (t_norm * (self.pe.shape[0] - 1)).long().clamp(0, self.pe.shape[0] - 1)
        
        return self.pe[t_indices]  # (B, T, dim)


class MultiScaleMLP(nn.Module):
    """多尺度MLP，处理不同感受野的特征"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 不同尺度的MLP分支
        self.local_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        self.global_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度MLP处理"""
        local_out = self.local_mlp(x)
        global_out = self.global_mlp(x)
        
        # 门控权重
        gate_weights = self.gate(x)  # (B, L, 2)
        
        # 加权融合
        output = gate_weights[..., 0:1] * local_out + gate_weights[..., 1:2] * global_out
        
        return self.dropout(output)


class AttentionGuidedConv2d(nn.Module):
    """注意力引导的2D卷积"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),  # 2 channels: avg + max
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """注意力引导的卷积操作"""
        # 通道注意力
        channel_attn = self.channel_attention(x)
        x_channel = x * channel_attn
        
        # 空间注意力
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attn = self.spatial_attention(spatial_input)
        x_spatial = x_channel * spatial_attn
        
        # 卷积操作
        return self.conv(x_spatial)