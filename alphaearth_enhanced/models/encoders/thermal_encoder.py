"""
Thermal Infrared Encoder for Temperature and Heat Signatures
Handles Landsat thermal bands and other thermal sensors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from einops import rearrange

from ..backbone import SpatioTemporalViT


class TemperatureCalibration(nn.Module):
    """Calibration module for thermal data"""
    
    def __init__(
        self,
        in_channels: int = 2,
        calibration_type: str = "landsat",  # or "modis", "viirs"
    ):
        super().__init__()
        self.calibration_type = calibration_type
        
        # Learnable calibration parameters
        self.gain = nn.Parameter(torch.ones(in_channels))
        self.offset = nn.Parameter(torch.zeros(in_channels))
        
        # Temperature range normalization
        self.temp_min = 250.0  # Kelvin
        self.temp_max = 350.0  # Kelvin
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calibrate thermal data
        
        Args:
            x: Raw thermal data [B, C, H, W]
        
        Returns:
            Calibrated temperature data
        """
        # Apply calibration
        x = x * self.gain.view(1, -1, 1, 1) + self.offset.view(1, -1, 1, 1)
        
        # Normalize to [0, 1] based on expected temperature range
        x = (x - self.temp_min) / (self.temp_max - self.temp_min)
        x = torch.clamp(x, 0, 1)
        
        return x


class ThermalFeatureExtractor(nn.Module):
    """Extract thermal features and patterns"""
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 16,
    ):
        super().__init__()
        
        # Temperature gradient extraction
        self.gradient_x = nn.Conv2d(in_channels, out_channels // 2, 
                                    kernel_size=(1, 3), padding=(0, 1))
        self.gradient_y = nn.Conv2d(in_channels, out_channels // 2,
                                    kernel_size=(3, 1), padding=(1, 0))
        
        # Heat pattern detection
        self.heat_patterns = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Combine features
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract thermal features"""
        # Temperature gradients
        grad_x = self.gradient_x(x)
        grad_y = self.gradient_y(x)
        gradients = torch.cat([grad_x, grad_y], dim=1)
        
        # Heat patterns
        patterns = self.heat_patterns(x)
        
        # Combine
        features = torch.cat([gradients, patterns], dim=1)
        return self.fusion(features)


class ThermalEncoder(nn.Module):
    """
    Encoder for thermal infrared imagery
    
    Features:
    - Temperature calibration
    - Heat signature detection
    - Urban heat island analysis
    - Diurnal temperature variation handling
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 32,  # Larger patches for lower resolution thermal
        num_frames: int = 4,
        in_chans: int = 2,  # Landsat Band 10, 11
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_calibration: bool = True,
        use_feature_extraction: bool = True,
        sensor_type: str = "landsat",
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.sensor_type = sensor_type
        
        # Temperature calibration
        if use_calibration:
            self.calibration = TemperatureCalibration(
                in_channels=in_chans,
                calibration_type=sensor_type,
            )
        else:
            self.calibration = nn.Identity()
        
        # Thermal feature extraction
        feature_channels = in_chans
        if use_feature_extraction:
            self.feature_extractor = ThermalFeatureExtractor(
                in_channels=in_chans,
                out_channels=16,
            )
            feature_channels = 16
        else:
            self.feature_extractor = nn.Identity()
        
        # Main encoder backbone
        self.backbone = SpatioTemporalViT(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=feature_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_cls_token=True,
        )
        
        # Time-of-day encoding for thermal patterns
        self.time_encoder = nn.Sequential(
            nn.Linear(2, 32),  # Hour and season
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )
        
        # Surface type embedding (urban, vegetation, water, etc.)
        self.surface_type_embed = nn.Embedding(5, embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        time_info: Optional[torch.Tensor] = None,  # [B, 2] hour, season
        surface_types: Optional[torch.Tensor] = None,  # [B] surface type indices
        metadata: Optional[Dict] = None,
        return_features: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of thermal encoder
        
        Args:
            x: Thermal imagery [B, T, C, H, W] or [B, C, H, W]
            time_info: Time of day and season information
            surface_types: Surface type indices
            metadata: Optional metadata
            return_features: Return intermediate features
        
        Returns:
            Encoded thermal features
        """
        # Add time dimension if needed
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        B, T, C, H, W = x.shape
        
        # Reshape for preprocessing
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        # Calibrate thermal data
        x = self.calibration(x)
        
        # Extract thermal features
        x = self.feature_extractor(x)
        
        # Reshape back
        _, C_new, _, _ = x.shape
        x = rearrange(x, '(b t) c h w -> b t c h w', b=B, t=T)
        
        # Forward through backbone
        features = self.backbone(x, return_features=return_features)
        
        # Add time encoding if provided
        if time_info is not None:
            time_features = self.time_encoder(time_info)
            if len(features.shape) == 3:
                time_features = time_features.unsqueeze(1)
            features = features + time_features
        
        # Add surface type information if provided
        if surface_types is not None:
            surface_features = self.surface_type_embed(surface_types)
            if len(features.shape) == 3:
                surface_features = surface_features.unsqueeze(1)
            features = features + surface_features
        
        return features
    
    def extract_heat_signatures(
        self,
        x: torch.Tensor,
        threshold: float = 0.8,
    ) -> torch.Tensor:
        """
        Extract urban heat island signatures
        
        Args:
            x: Thermal data
            threshold: Temperature threshold for heat detection
        
        Returns:
            Heat signature mask
        """
        # Calibrate to temperature
        x_calibrated = self.calibration(x)
        
        # Detect high temperature regions
        heat_mask = (x_calibrated > threshold).float()
        
        # Apply morphological operations for better signatures
        kernel = torch.ones(1, 1, 3, 3, device=x.device) / 9
        heat_mask = F.conv2d(heat_mask, kernel, padding=1)
        heat_mask = (heat_mask > 0.5).float()
        
        return heat_mask