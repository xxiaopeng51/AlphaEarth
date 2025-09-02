"""
SAR (Synthetic Aperture Radar) Encoder
Handles Sentinel-1 and other SAR data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from einops import rearrange
import numpy as np

from ..backbone import SpatioTemporalViT


class SARPreprocessing(nn.Module):
    """Preprocessing module for SAR data"""
    
    def __init__(
        self,
        in_channels: int = 2,  # VV and VH polarizations
        out_channels: int = 8,
        use_phase: bool = False,
    ):
        super().__init__()
        self.use_phase = use_phase
        
        # Feature extraction from SAR polarizations
        self.polarization_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Ratio and difference features
        self.ratio_conv = nn.Sequential(
            nn.Conv2d(1, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        self.diff_conv = nn.Sequential(
            nn.Conv2d(1, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        # Combine all features
        total_channels = out_channels + out_channels // 2 + out_channels // 2
        self.fusion = nn.Conv2d(total_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess SAR data
        
        Args:
            x: SAR data [B, 2, H, W] with VV and VH channels
        
        Returns:
            Preprocessed features [B, out_channels, H, W]
        """
        vv = x[:, 0:1, :, :]
        vh = x[:, 1:2, :, :]
        
        # Basic polarization features
        pol_features = self.polarization_conv(x)
        
        # VV/VH ratio (avoiding division by zero)
        ratio = vv / (vh + 1e-8)
        ratio_features = self.ratio_conv(ratio)
        
        # VV-VH difference
        diff = vv - vh
        diff_features = self.diff_conv(diff)
        
        # Combine all features
        combined = torch.cat([pol_features, ratio_features, diff_features], dim=1)
        output = self.fusion(combined)
        
        return output


class SpeckleFilter(nn.Module):
    """Learnable speckle noise reduction for SAR"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply speckle filtering"""
        attention = self.filter(x)
        return x * attention


class SAREncoder(nn.Module):
    """
    Encoder for SAR (Synthetic Aperture Radar) data
    
    Features:
    - Dual polarization support (VV, VH)
    - Speckle noise handling
    - Interferometric SAR support
    - All-weather capability representation
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 4,
        in_chans: int = 2,  # VV and VH polarizations
        embed_dim: int = 768,
        depth: int = 6,  # Smaller depth for SAR
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_preprocessing: bool = True,
        use_speckle_filter: bool = True,
        preprocessed_channels: int = 8,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_preprocessing = use_preprocessing
        self.use_speckle_filter = use_speckle_filter
        
        # SAR-specific preprocessing
        if use_preprocessing:
            self.preprocessing = SARPreprocessing(
                in_channels=in_chans,
                out_channels=preprocessed_channels,
            )
            backbone_in_chans = preprocessed_channels
        else:
            self.preprocessing = nn.Identity()
            backbone_in_chans = in_chans
        
        # Speckle filtering
        if use_speckle_filter:
            self.speckle_filter = SpeckleFilter(backbone_in_chans)
        else:
            self.speckle_filter = nn.Identity()
        
        # Log transformation for SAR data (convert to dB)
        self.log_transform = LogTransform()
        
        # Main encoder backbone
        self.backbone = SpatioTemporalViT(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=backbone_in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_cls_token=True,
        )
        
        # Incidence angle encoding
        self.incidence_angle_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        incidence_angle: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
        return_features: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of SAR encoder
        
        Args:
            x: SAR data [B, T, C, H, W] or [B, C, H, W]
            incidence_angle: Radar incidence angle [B, 1]
            metadata: Optional metadata
            return_features: Return intermediate features
        
        Returns:
            Encoded SAR features
        """
        # Add time dimension if needed
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        B, T, C, H, W = x.shape
        
        # Apply log transformation (convert to dB)
        x = self.log_transform(x)
        
        # Reshape for preprocessing
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        # SAR preprocessing
        x = self.preprocessing(x)
        
        # Speckle filtering
        x = self.speckle_filter(x)
        
        # Reshape back
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        
        # Forward through backbone
        features = self.backbone(x, return_features=return_features)
        
        # Add incidence angle information if provided
        if incidence_angle is not None:
            angle_features = self.incidence_angle_embed(incidence_angle)
            if len(features.shape) == 3:
                # Add to sequence features
                angle_features = angle_features.unsqueeze(1)
                features = features + angle_features
            else:
                features = features + angle_features
        
        return features


class LogTransform(nn.Module):
    """Log transformation for SAR data (linear to dB conversion)"""
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply log transformation"""
        return 10 * torch.log10(x + self.epsilon)