"""
Geospatial Position Encoding

地理空间位置编码，融合了地理坐标、时间和分辨率信息。
借鉴SatCLIP和Clay的位置编码设计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import numpy as np
from einops import rearrange, repeat


class GeographicCoordinateEncoding(nn.Module):
    """地理坐标编码，处理经纬度信息"""
    
    def __init__(self, dim: int, max_lat: float = 90.0, max_lon: float = 180.0):
        super().__init__()
        self.dim = dim
        self.max_lat = max_lat
        self.max_lon = max_lon
        
        # 经纬度编码器
        self.lat_encoder = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim // 2)
        )
        
        self.lon_encoder = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(), 
            nn.Linear(dim // 4, dim // 2)
        )
        
        # 融合层
        self.coord_fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coordinates: (B, 2) [lat, lon] 地理坐标
        Returns:
            (B, dim) 地理坐标编码
        """
        lat, lon = coordinates[:, 0:1], coordinates[:, 1:2]
        
        # 标准化坐标
        lat_norm = lat / self.max_lat  # [-1, 1]
        lon_norm = lon / self.max_lon  # [-1, 1]
        
        # 编码经纬度
        lat_encoded = self.lat_encoder(lat_norm)  # (B, dim//2)
        lon_encoded = self.lon_encoder(lon_norm)  # (B, dim//2)
        
        # 拼接和融合
        coord_encoded = torch.cat([lat_encoded, lon_encoded], dim=-1)  # (B, dim)
        coord_fused = self.coord_fusion(coord_encoded)
        
        return coord_fused


class MultiScalePositionEncoding(nn.Module):
    """多尺度位置编码，支持不同分辨率的数据"""
    
    def __init__(self, dim: int, max_resolution: int = 1000):
        super().__init__()
        self.dim = dim
        self.max_resolution = max_resolution
        
        # 分辨率编码器
        self.resolution_encoder = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim // 4)
        )
        
        # 2D正弦位置编码
        self.pos_encoding_2d = nn.Parameter(
            self._create_2d_pos_encoding(dim * 3 // 4), requires_grad=False
        )
    
    def _create_2d_pos_encoding(self, dim: int, height: int = 256, width: int = 256) -> torch.Tensor:
        """创建2D正弦位置编码"""
        pe = torch.zeros(height, width, dim)
        
        # Y方向编码
        y_pos = torch.arange(height).unsqueeze(1).float()
        # X方向编码  
        x_pos = torch.arange(width).unsqueeze(0).float()
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           (-math.log(10000.0) / dim))
        
        # Y方向
        pe[:, :, 0::4] = torch.sin(y_pos * div_term).unsqueeze(1).expand(-1, width, -1)
        pe[:, :, 1::4] = torch.cos(y_pos * div_term).unsqueeze(1).expand(-1, width, -1)
        
        # X方向
        pe[:, :, 2::4] = torch.sin(x_pos * div_term).unsqueeze(0).expand(height, -1, -1)
        pe[:, :, 3::4] = torch.cos(x_pos * div_term).unsqueeze(0).expand(height, -1, -1)
        
        return pe
    
    def forward(self, H: int, W: int, resolution: float) -> torch.Tensor:
        """
        Args:
            H, W: 空间维度
            resolution: 数据分辨率 (米)
        Returns:
            (H, W, dim) 位置编码
        """
        # 分辨率编码
        res_norm = torch.tensor([resolution / self.max_resolution], dtype=torch.float32)
        res_encoded = self.resolution_encoder(res_norm.unsqueeze(0))  # (1, dim//4)
        
        # 获取2D位置编码
        if H <= self.pos_encoding_2d.shape[0] and W <= self.pos_encoding_2d.shape[1]:
            pos_2d = self.pos_encoding_2d[:H, :W]  # (H, W, 3*dim//4)
        else:
            # 如果超出预计算范围，重新计算
            pos_2d = self._create_2d_pos_encoding(self.dim * 3 // 4, H, W)
            pos_2d = pos_2d.to(self.pos_encoding_2d.device)
        
        # 拼接分辨率编码
        res_broadcasted = res_encoded.unsqueeze(0).unsqueeze(0).expand(H, W, -1)  # (H, W, dim//4)
        full_encoding = torch.cat([pos_2d, res_broadcasted], dim=-1)  # (H, W, dim)
        
        return full_encoding


class TemporalPositionEncoding(nn.Module):
    """时间位置编码，支持连续时间和周期性模式"""
    
    def __init__(self, dim: int, max_time_range: float = 365 * 24 * 3600 * 1000):
        super().__init__()
        self.dim = dim
        self.max_time_range = max_time_range
        
        # 线性时间编码
        self.linear_time_encoder = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim // 2)
        )
        
        # 周期性时间编码 (年、月、日、小时)
        self.periodic_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, dim // 8),  # sin, cos
                nn.GELU(),
                nn.Linear(dim // 8, dim // 8)
            ) for _ in range(4)  # 年、月、日、小时
        ])
        
        # 融合层
        self.temporal_fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: (B, T) 时间戳 (毫秒)
        Returns:
            (B, T, dim) 时间位置编码
        """
        B, T = timestamps.shape
        
        # 标准化时间戳
        t_norm = timestamps / self.max_time_range  # (B, T)
        
        # 线性时间编码
        linear_encoded = self.linear_time_encoder(t_norm.unsqueeze(-1))  # (B, T, dim//2)
        
        # 周期性编码
        periodic_encodings = []
        
        # 年周期 (365天)
        year_phase = (t_norm * 365) * 2 * math.pi
        year_enc = torch.stack([torch.sin(year_phase), torch.cos(year_phase)], dim=-1)
        year_encoded = self.periodic_encoders[0](year_enc)
        periodic_encodings.append(year_encoded)
        
        # 月周期 (30天)
        month_phase = (t_norm * 365 / 30) * 2 * math.pi
        month_enc = torch.stack([torch.sin(month_phase), torch.cos(month_phase)], dim=-1)
        month_encoded = self.periodic_encoders[1](month_enc)
        periodic_encodings.append(month_encoded)
        
        # 日周期 (24小时)
        day_phase = (t_norm * 365 * 24) * 2 * math.pi
        day_enc = torch.stack([torch.sin(day_phase), torch.cos(day_phase)], dim=-1)
        day_encoded = self.periodic_encoders[2](day_enc)
        periodic_encodings.append(day_encoded)
        
        # 小时周期
        hour_phase = (t_norm * 365 * 24 * 60) * 2 * math.pi
        hour_enc = torch.stack([torch.sin(hour_phase), torch.cos(hour_phase)], dim=-1)
        hour_encoded = self.periodic_encoders[3](hour_enc)
        periodic_encodings.append(hour_encoded)
        
        # 拼接所有编码
        periodic_concat = torch.cat(periodic_encodings, dim=-1)  # (B, T, dim//2)
        full_temporal = torch.cat([linear_encoded, periodic_concat], dim=-1)  # (B, T, dim)
        
        # 融合处理
        return self.temporal_fusion(full_temporal)


class GeospatialPositionEncoding(nn.Module):
    """
    地理空间位置编码主类
    
    整合地理坐标、时间和空间位置信息
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 地理坐标编码
        self.geographic_encoding = GeographicCoordinateEncoding(d_model // 3)
        
        # 多尺度空间位置编码
        self.spatial_encoding = MultiScalePositionEncoding(d_model // 3)
        
        # 时间位置编码
        self.temporal_encoding = TemporalPositionEncoding(d_model // 3)
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        features: torch.Tensor,      # (B, T, H, W, d_model)
        timestamps: torch.Tensor,    # (B, T)
        coordinates: torch.Tensor,   # (B, 2) [lat, lon]
        resolution: float = 10.0     # 数据分辨率
    ) -> torch.Tensor:
        """
        添加地理空间位置编码
        
        Args:
            features: (B, T, H, W, d_model) 输入特征
            timestamps: (B, T) 时间戳
            coordinates: (B, 2) 地理坐标
            resolution: 数据分辨率
        Returns:
            (B, T, H, W, d_model) 添加位置编码后的特征
        """
        B, T, H, W, D = features.shape
        
        # 1. 地理坐标编码
        geo_encoded = self.geographic_encoding(coordinates)  # (B, d_model//3)
        geo_broadcasted = geo_encoded.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, T, H, W, -1)
        
        # 2. 空间位置编码
        spatial_encoded = self.spatial_encoding(H, W, resolution)  # (H, W, d_model//3)
        spatial_broadcasted = spatial_encoded.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
        
        # 3. 时间位置编码
        temporal_encoded = self.temporal_encoding(timestamps)  # (B, T, d_model//3)
        temporal_broadcasted = temporal_encoded.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, H, W, -1)
        
        # 4. 拼接所有位置编码
        position_encoding = torch.cat([
            geo_broadcasted,
            spatial_broadcasted, 
            temporal_broadcasted
        ], dim=-1)  # (B, T, H, W, d_model)
        
        # 5. 最终融合
        position_encoding = self.final_fusion(position_encoding)
        
        # 6. 添加到输入特征
        encoded_features = features + position_encoding
        
        return self.dropout(encoded_features)