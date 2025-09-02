"""
Learned Pyramid Exchange

改进的金字塔信息交换机制，支持多尺度特征的自适应融合。
相比原始的双线性插值，使用可学习的上采样和下采样。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from einops import rearrange


class LearnedUpsampling(nn.Module):
    """可学习的上采样模块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        use_conv_transpose: bool = True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_conv_transpose = use_conv_transpose
        
        if use_conv_transpose and scale_factor > 1:
            # 使用转置卷积进行上采样
            stride = int(scale_factor)
            self.upsample = nn.ConvTranspose2d(
                in_channels, out_channels, 
                kernel_size=stride * 2, 
                stride=stride, 
                padding=stride // 2
            )
        else:
            # 使用插值 + 卷积
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)


class LearnedDownsampling(nn.Module):
    """可学习的下采样模块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        use_strided_conv: bool = True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        
        if use_strided_conv and scale_factor < 1:
            # 使用步长卷积进行下采样
            stride = int(1 / scale_factor)
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=stride * 2, stride=stride, padding=stride // 2),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            # 使用池化 + 卷积
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=int(1/scale_factor), stride=int(1/scale_factor)),
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(x)


class AdaptiveResampling(nn.Module):
    """自适应重采样模块，根据特征内容调整采样策略"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: float,
        num_heads: int = 8
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 特征重要性评估
        self.importance_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 自适应权重生成
        self.weight_generator = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, out_channels)
        )
        
        # 基础重采样
        if scale_factor > 1:
            self.base_resample = LearnedUpsampling(in_channels, out_channels, scale_factor)
        else:
            self.base_resample = LearnedDownsampling(in_channels, out_channels, scale_factor)
        
        # 残差连接投影
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (BT, C, H, W) 输入特征
        Returns:
            (BT, out_C, H', W') 重采样后的特征
        """
        # 评估特征重要性
        importance = self.importance_estimator(x)  # (BT, 1, 1, 1)
        
        # 基础重采样
        resampled = self.base_resample(x)
        
        # 自适应权重调制
        global_feat = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # (BT, C)
        adaptive_weights = self.weight_generator(global_feat)  # (BT, out_C)
        adaptive_weights = adaptive_weights.unsqueeze(-1).unsqueeze(-1)  # (BT, out_C, 1, 1)
        
        # 应用自适应权重
        resampled = resampled * adaptive_weights
        
        # 重要性加权
        resampled = resampled * importance
        
        return resampled


class LearnedPyramidExchange(nn.Module):
    """
    可学习的金字塔交换模块
    
    改进的多尺度信息交换，使用自适应重采样和注意力机制
    """
    
    def __init__(
        self,
        space_dim: int = 1024,
        time_dim: int = 512,
        precision_dim: int = 256,
        use_learned_resampling: bool = True,
        use_cross_scale_attention: bool = True
    ):
        super().__init__()
        self.space_dim = space_dim
        self.time_dim = time_dim
        self.precision_dim = precision_dim
        self.use_learned_resampling = use_learned_resampling
        self.use_cross_scale_attention = use_cross_scale_attention
        
        # 重采样模块
        if use_learned_resampling:
            # Space -> Time (1/16 -> 1/8, scale_factor = 2.0)
            self.space_to_time = AdaptiveResampling(space_dim, time_dim, 2.0)
            
            # Space -> Precision (1/16 -> 1/2, scale_factor = 8.0)
            self.space_to_precision = AdaptiveResampling(space_dim, precision_dim, 8.0)
            
            # Time -> Space (1/8 -> 1/16, scale_factor = 0.5)
            self.time_to_space = AdaptiveResampling(time_dim, space_dim, 0.5)
            
            # Time -> Precision (1/8 -> 1/2, scale_factor = 4.0)
            self.time_to_precision = AdaptiveResampling(time_dim, precision_dim, 4.0)
            
            # Precision -> Space (1/2 -> 1/16, scale_factor = 0.125)
            self.precision_to_space = AdaptiveResampling(precision_dim, space_dim, 0.125)
            
            # Precision -> Time (1/2 -> 1/8, scale_factor = 0.25)
            self.precision_to_time = AdaptiveResampling(precision_dim, time_dim, 0.25)
        else:
            # 使用简单的插值重采样
            self.space_to_time = lambda x: F.interpolate(x, scale_factor=2.0, mode='bilinear')
            # ... 其他重采样函数
        
        # 跨尺度注意力 (如果启用)
        if use_cross_scale_attention:
            self.cross_scale_attention = CrossScaleAttention(
                dims=[space_dim, time_dim, precision_dim]
            )
        
        # 融合权重学习
        self.fusion_weights = nn.ModuleDict({
            "space": nn.Sequential(
                nn.Linear(space_dim + time_dim + precision_dim, space_dim),
                nn.GELU(),
                nn.Linear(space_dim, 3),
                nn.Softmax(dim=-1)
            ),
            "time": nn.Sequential(
                nn.Linear(space_dim + time_dim + precision_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, 3),
                nn.Softmax(dim=-1)
            ),
            "precision": nn.Sequential(
                nn.Linear(space_dim + time_dim + precision_dim, precision_dim),
                nn.GELU(),
                nn.Linear(precision_dim, 3),
                nn.Softmax(dim=-1)
            )
        })
    
    def forward(
        self,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        precision_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            space_x: (B, T, H_s, W_s, space_dim) 空间特征
            time_x: (B, T, H_t, W_t, time_dim) 时间特征  
            precision_x: (B, T, H_p, W_p, precision_dim) 精度特征
        Returns:
            交换后的三个尺度特征
        """
        B, T = space_x.shape[:2]
        
        # 重塑为卷积格式
        space_conv = rearrange(space_x, 'b t h w c -> (b t) c h w')
        time_conv = rearrange(time_x, 'b t h w c -> (b t) c h w')
        precision_conv = rearrange(precision_x, 'b t h w c -> (b t) c h w')
        
        # 跨尺度重采样
        if self.use_learned_resampling:
            # Space接收来自Time和Precision的信息
            time_to_space = self.time_to_space(time_conv)
            precision_to_space = self.precision_to_space(precision_conv)
            
            # Time接收来自Space和Precision的信息
            space_to_time = self.space_to_time(space_conv)
            precision_to_time = self.precision_to_time(precision_conv)
            
            # Precision接收来自Space和Time的信息
            space_to_precision = self.space_to_precision(space_conv)
            time_to_precision = self.time_to_precision(time_conv)
        else:
            # 简单插值重采样 (fallback)
            target_space_size = space_conv.shape[2:]
            target_time_size = time_conv.shape[2:]
            target_precision_size = precision_conv.shape[2:]
            
            time_to_space = F.interpolate(time_conv, size=target_space_size, mode='bilinear')
            precision_to_space = F.interpolate(precision_conv, size=target_space_size, mode='bilinear')
            
            space_to_time = F.interpolate(space_conv, size=target_time_size, mode='bilinear')
            precision_to_time = F.interpolate(precision_conv, size=target_time_size, mode='bilinear')
            
            space_to_precision = F.interpolate(space_conv, size=target_precision_size, mode='bilinear')
            time_to_precision = F.interpolate(time_conv, size=target_precision_size, mode='bilinear')
        
        # 学习融合权重
        # 为每个尺度计算全局特征用于权重生成
        space_global = F.adaptive_avg_pool2d(space_conv, 1).squeeze(-1).squeeze(-1)  # (BT, space_dim)
        time_global = F.adaptive_avg_pool2d(time_conv, 1).squeeze(-1).squeeze(-1)    # (BT, time_dim)
        precision_global = F.adaptive_avg_pool2d(precision_conv, 1).squeeze(-1).squeeze(-1)  # (BT, precision_dim)
        
        # 拼接全局特征
        global_concat = torch.cat([space_global, time_global, precision_global], dim=-1)
        
        # 计算融合权重
        space_weights = self.fusion_weights["space"](global_concat)  # (BT, 3)
        time_weights = self.fusion_weights["time"](global_concat)    # (BT, 3)
        precision_weights = self.fusion_weights["precision"](global_concat)  # (BT, 3)
        
        # 应用权重进行融合
        space_fused = (space_weights[:, 0:1, None, None] * space_conv + 
                      space_weights[:, 1:2, None, None] * time_to_space +
                      space_weights[:, 2:3, None, None] * precision_to_space)
        
        time_fused = (time_weights[:, 0:1, None, None] * space_to_time +
                     time_weights[:, 1:2, None, None] * time_conv +
                     time_weights[:, 2:3, None, None] * precision_to_time)
        
        precision_fused = (precision_weights[:, 0:1, None, None] * space_to_precision +
                          precision_weights[:, 1:2, None, None] * time_to_precision +
                          precision_weights[:, 2:3, None, None] * precision_conv)
        
        # 跨尺度注意力 (如果启用)
        if self.use_cross_scale_attention:
            space_fused, time_fused, precision_fused = self.cross_scale_attention(
                space_fused, time_fused, precision_fused
            )
        
        # 重塑回时序格式
        space_out = rearrange(space_fused, '(b t) c h w -> b t h w c', b=B, t=T)
        time_out = rearrange(time_fused, '(b t) c h w -> b t h w c', b=B, t=T)
        precision_out = rearrange(precision_fused, '(b t) c h w -> b t h w c', b=B, t=T)
        
        return space_out, time_out, precision_out


class CrossScaleAttention(nn.Module):
    """跨尺度注意力机制"""
    
    def __init__(self, dims: list, num_heads: int = 8):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        
        # 为每个尺度创建查询投影
        self.query_projections = nn.ModuleList([
            nn.Linear(dim, dim) for dim in dims
        ])
        
        # 跨尺度注意力
        self.cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=min(num_heads, dim // 64),
                batch_first=True
            ) for dim in dims
        ])
        
        # 输出投影
        self.output_projections = nn.ModuleList([
            nn.Linear(dim, dim) for dim in dims
        ])
    
    def forward(
        self,
        space_x: torch.Tensor,
        time_x: torch.Tensor,
        precision_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        跨尺度注意力处理
        
        Args:
            space_x: (BT, space_dim, H_s, W_s)
            time_x: (BT, time_dim, H_t, W_t)
            precision_x: (BT, precision_dim, H_p, W_p)
        """
        features = [space_x, time_x, precision_x]
        outputs = []
        
        for i, (feat, query_proj, cross_attn, output_proj) in enumerate(zip(
            features, self.query_projections, self.cross_attentions, self.output_projections
        )):
            BT, C, H, W = feat.shape
            
            # 转换为序列格式
            feat_seq = rearrange(feat, 'bt c h w -> bt (h w) c')
            
            # 生成查询
            queries = query_proj(feat_seq)
            
            # 构建键值对 (来自其他尺度)
            other_features = [f for j, f in enumerate(features) if j != i]
            if other_features:
                # 将其他尺度特征插值到当前尺度
                other_resized = []
                for other_feat in other_features:
                    other_resized_feat = F.interpolate(other_feat, size=(H, W), mode='bilinear')
                    other_resized.append(rearrange(other_resized_feat, 'bt c h w -> bt (h w) c'))
                
                # 拼接其他尺度特征作为键值
                keys_values = torch.cat(other_resized, dim=1)  # (BT, (H*W)*2, C)
                
                # 跨尺度注意力
                attended_feat, _ = cross_attn(queries, keys_values, keys_values)
                
                # 残差连接
                feat_seq = feat_seq + attended_feat
            
            # 输出投影
            feat_out = output_proj(feat_seq)
            
            # 重塑回卷积格式
            feat_out = rearrange(feat_out, 'bt (h w) c -> bt c h w', h=H, w=W)
            outputs.append(feat_out)
        
        return tuple(outputs)