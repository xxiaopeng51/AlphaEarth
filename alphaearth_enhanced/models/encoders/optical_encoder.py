"""
Optical Imagery Encoder for Multi-spectral Satellite Data
Handles Sentinel-2, Landsat, and other optical sensors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from einops import rearrange
import numpy as np

from ..backbone import SpatioTemporalViT, VisionTransformerMAE


class SpectralAttention(nn.Module):
    """Attention mechanism for spectral bands"""
    
    def __init__(self, num_bands: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_bands, num_bands // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands // reduction, num_bands, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral attention"""
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BandNormalization(nn.Module):
    """Normalization for multi-spectral bands"""
    
    def __init__(
        self,
        num_bands: int,
        sentinel2_bands: bool = True,
    ):
        super().__init__()
        self.num_bands = num_bands
        
        # Default statistics for Sentinel-2 bands
        if sentinel2_bands:
            # Mean and std for 13 Sentinel-2 bands
            self.register_buffer('mean', torch.tensor([
                1370.19, 1184.35, 1119.65, 1003.07,  # B1-B4
                1257.54, 1819.76, 2073.34, 2049.46,  # B5-B8
                2212.34, 658.52, 12.51,              # B8A-B10
                1819.61, 1274.13                     # B11-B12
            ])[:num_bands])
            
            self.register_buffer('std', torch.tensor([
                633.90, 580.95, 551.14, 620.90,      # B1-B4
                669.24, 756.55, 867.37, 936.77,      # B5-B8
                955.13, 413.48, 65.48,               # B8A-B10
                978.88, 752.04                       # B11-B12
            ])[:num_bands])
        else:
            # Learnable normalization parameters
            self.register_parameter('mean', nn.Parameter(torch.zeros(num_bands)))
            self.register_parameter('std', nn.Parameter(torch.ones(num_bands)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize multi-spectral bands"""
        # x shape: [B, C, H, W] or [B, T, C, H, W]
        if len(x.shape) == 5:
            mean = self.mean.view(1, 1, -1, 1, 1)
            std = self.std.view(1, 1, -1, 1, 1)
        else:
            mean = self.mean.view(1, -1, 1, 1)
            std = self.std.view(1, -1, 1, 1)
        
        return (x - mean) / (std + 1e-8)


class OpticalEncoder(nn.Module):
    """
    Encoder for optical satellite imagery
    
    Features:
    - Multi-spectral band support (RGB, NIR, SWIR, etc.)
    - Temporal sequence handling
    - Cloud masking awareness
    - Resolution adaptation
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 4,
        in_chans: int = 13,  # Sentinel-2 bands
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_spectral_attention: bool = True,
        use_band_normalization: bool = True,
        use_mae: bool = False,
        sensor_type: str = "sentinel2",  # or "landsat", "modis", "custom"
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.sensor_type = sensor_type
        
        # Band normalization
        if use_band_normalization:
            self.band_norm = BandNormalization(
                in_chans,
                sentinel2_bands=(sensor_type == "sentinel2")
            )
        else:
            self.band_norm = nn.Identity()
        
        # Spectral attention
        if use_spectral_attention:
            self.spectral_attention = SpectralAttention(in_chans)
        else:
            self.spectral_attention = nn.Identity()
        
        # Band grouping for different resolutions (Sentinel-2 specific)
        if sensor_type == "sentinel2":
            self.band_groups = {
                '10m': [1, 2, 3, 7],      # B2, B3, B4, B8
                '20m': [4, 5, 6, 8, 10, 11],  # B5, B6, B7, B8A, B11, B12
                '60m': [0, 9]             # B1, B10
            }
        
        # Main encoder backbone
        if use_mae:
            self.backbone = VisionTransformerMAE(
                img_size=img_size,
                patch_size=patch_size,
                num_frames=num_frames,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
        else:
            self.backbone = SpatioTemporalViT(
                img_size=img_size,
                patch_size=patch_size,
                num_frames=num_frames,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
            )
        
        # Cloud mask processing
        self.cloud_mask_embed = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.cloud_mask_fusion = nn.Conv2d(in_chans + 16, in_chans, kernel_size=1)
    
    def process_cloud_mask(
        self,
        x: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process cloud mask and fuse with input
        
        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W]
            cloud_mask: Cloud mask [B, 1, H, W] or [B, T, 1, H, W]
        
        Returns:
            Processed input with cloud information
        """
        if cloud_mask is None:
            return x
        
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            cloud_mask = rearrange(cloud_mask, 'b t c h w -> (b t) c h w')
        
        # Embed cloud mask
        cloud_features = self.cloud_mask_embed(cloud_mask)
        
        # Concatenate and fuse
        x_combined = torch.cat([x, cloud_features], dim=1)
        x = self.cloud_mask_fusion(x_combined)
        
        if len(x.shape) == 4 and T > 1:
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        
        return x
    
    def forward_mae(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for MAE pretraining"""
        if hasattr(self.backbone, 'forward_mae'):
            return self.backbone.forward_mae(x, mask_ratio)
        else:
            raise NotImplementedError("MAE not supported with current backbone")
    
    def decode(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Decode for MAE reconstruction"""
        if hasattr(self.backbone, 'forward_decoder'):
            return self.backbone.forward_decoder(latent, ids_restore)
        else:
            raise NotImplementedError("Decoder not available")
    
    def forward(
        self,
        x: torch.Tensor,
        cloud_mask: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
        return_features: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of optical encoder
        
        Args:
            x: Input optical imagery [B, T, C, H, W] or [B, C, H, W]
            cloud_mask: Optional cloud mask
            metadata: Optional metadata (acquisition time, sun angle, etc.)
            return_features: Return intermediate features
        
        Returns:
            Encoded features
        """
        # Add time dimension if needed
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        B, T, C, H, W = x.shape
        
        # Normalize bands
        x = self.band_norm(x)
        
        # Apply spectral attention
        if T > 1:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.spectral_attention(x)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        else:
            x = x.squeeze(1)
            x = self.spectral_attention(x)
            x = x.unsqueeze(1)
        
        # Process cloud mask if provided
        x = self.process_cloud_mask(x, cloud_mask)
        
        # Forward through backbone
        features = self.backbone(x, return_features=return_features)
        
        return features
    
    def extract_band_features(
        self,
        x: torch.Tensor,
        band_indices: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Extract features from specific spectral bands
        
        Args:
            x: Input tensor [B, C, H, W]
            band_indices: Indices of bands to extract
        
        Returns:
            Band-specific features
        """
        if band_indices is None:
            return x
        
        return x[:, band_indices, :, :]
    
    def get_resolution_features(
        self,
        x: torch.Tensor,
        resolution: str = "10m",
    ) -> torch.Tensor:
        """
        Get features at specific resolution (Sentinel-2 specific)
        
        Args:
            x: Input tensor
            resolution: Target resolution ("10m", "20m", "60m")
        
        Returns:
            Resolution-specific features
        """
        if self.sensor_type != "sentinel2" or resolution not in self.band_groups:
            return x
        
        band_indices = self.band_groups[resolution]
        return self.extract_band_features(x, band_indices)