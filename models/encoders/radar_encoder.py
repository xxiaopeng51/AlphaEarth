"""
Radar satellite imagery encoder for AlphaEarth Foundations model.

This module implements the radar encoder that processes SAR (Synthetic Aperture Radar)
data from sources like Sentinel-1, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import timm
import numpy as np


class RadarEncoder(nn.Module):
    """
    Radar satellite imagery encoder for SAR data processing.
    
    This encoder processes SAR imagery with different polarizations (VV, VH, HH, HV)
    and extracts radar-specific features including backscatter patterns,
    texture information, and temporal coherence.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 768,
        input_channels: int = 2,  # VV, VH polarizations
        patch_size: int = 16,
        image_size: int = 224,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.output_dim = output_dim
        
        # Load pre-trained vision transformer
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
            **kwargs
        )
        
        # Get backbone output dimension
        backbone_dim = self.backbone.num_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Input projection for different number of channels
        if input_channels != 3:
            self.input_projection = nn.Conv2d(
                input_channels, 3,
                kernel_size=1, stride=1, padding=0
            )
        else:
            self.input_projection = nn.Identity()
        
        # Radar-specific preprocessing layers
        self.radar_preprocessing = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Feature projection to desired output dimension
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for spatial information
        self.spatial_encoding = nn.Parameter(
            torch.randn(1, (image_size // patch_size) ** 2 + 1, backbone_dim)
        )
        
        # Radar-specific feature extraction
        self.radar_features = nn.ModuleDict({
            'texture': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 256)
            ),
            'backscatter': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 256)
            ),
            'polarization': nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 256)
            )
        })
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the radar encoder.
        
        Args:
            x: Input radar imagery tensor of shape (B, C, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - 'features': Encoded features (B, N, D)
                - 'global_features': Global pooled features (B, D)
                - 'radar_features': Radar-specific features
        """
        batch_size = x.shape[0]
        
        # Input projection for different channel numbers
        x = self.input_projection(x)
        
        # Radar-specific preprocessing
        radar_preprocessed = self.radar_preprocessing(x)
        
        # Extract radar-specific features
        radar_features = {}
        for feature_type, feature_extractor in self.radar_features.items():
            radar_features[feature_type] = feature_extractor(radar_preprocessed)
        
        # Extract features using backbone
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
        
        # Add spatial encoding
        if features.dim() == 3:  # (B, N, D)
            features = features + self.spatial_encoding
        else:  # (B, D) - need to reshape
            seq_len = (self.image_size // self.patch_size) ** 2 + 1
            features = features.unsqueeze(1).expand(-1, seq_len, -1)
            features = features + self.spatial_encoding
        
        # Project to output dimension
        projected_features = self.feature_projection(features)
        
        # Global pooling for global features
        global_features = projected_features.mean(dim=1)  # (B, D)
        
        result = {
            'features': projected_features,
            'global_features': global_features,
            'radar_features': radar_features
        }
        
        if return_features:
            result['spatial_features'] = projected_features[:, 1:, :]  # Remove CLS token
        
        return result
    
    def compute_coherence(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal coherence between two radar acquisitions.
        
        Args:
            x1: First radar acquisition (B, C, H, W)
            x2: Second radar acquisition (B, C, H, W)
            
        Returns:
            Coherence map (B, 1, H, W)
        """
        # Encode both acquisitions
        features1 = self.forward(x1)['spatial_features']
        features2 = self.forward(x2)['spatial_features']
        
        # Compute complex coherence
        # Convert to complex representation
        features1_complex = torch.complex(features1, torch.zeros_like(features1))
        features2_complex = torch.complex(features2, torch.zeros_like(features2))
        
        # Compute coherence
        numerator = torch.abs(torch.sum(features1_complex * torch.conj(features2_complex), dim=-1))
        denominator = torch.sqrt(
            torch.sum(torch.abs(features1_complex) ** 2, dim=-1) *
            torch.sum(torch.abs(features2_complex) ** 2, dim=-1)
        )
        
        coherence = numerator / (denominator + 1e-8)
        
        return coherence.unsqueeze(1)
    
    def extract_polarization_features(
        self,
        vv: torch.Tensor,
        vh: torch.Tensor,
        hh: Optional[torch.Tensor] = None,
        hv: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract polarization-specific features from multi-polarization SAR data.
        
        Args:
            vv: VV polarization (B, 1, H, W)
            vh: VH polarization (B, 1, H, W)
            hh: HH polarization (B, 1, H, W), optional
            hv: HV polarization (B, 1, H, V), optional
            
        Returns:
            Dictionary containing polarization features
        """
        # Combine available polarizations
        polarizations = [vv, vh]
        if hh is not None:
            polarizations.append(hh)
        if hv is not None:
            polarizations.append(hv)
        
        # Stack polarizations
        x = torch.cat(polarizations, dim=1)
        
        # Encode
        features = self.forward(x)
        
        # Compute polarization ratios and indices
        polarization_features = {}
        
        # VV/VH ratio
        vv_vh_ratio = vv / (vh + 1e-8)
        polarization_features['vv_vh_ratio'] = vv_vh_ratio
        
        # Cross-polarization ratio
        if hh is not None:
            hh_vv_ratio = hh / (vv + 1e-8)
            polarization_features['hh_vv_ratio'] = hh_vv_ratio
        
        # Polarization entropy (simplified)
        total_power = vv + vh
        if hh is not None:
            total_power += hh
        if hv is not None:
            total_power += hv
            
        vv_prob = vv / (total_power + 1e-8)
        vh_prob = vh / (total_power + 1e-8)
        
        entropy = -(vv_prob * torch.log(vv_prob + 1e-8) + 
                   vh_prob * torch.log(vh_prob + 1e-8))
        
        if hh is not None:
            hh_prob = hh / (total_power + 1e-8)
            entropy -= hh_prob * torch.log(hh_prob + 1e-8)
        
        polarization_features['entropy'] = entropy
        
        return {
            'features': features,
            'polarization_features': polarization_features
        }


class RadarEncoderWithTemporal(nn.Module):
    """
    Radar encoder with temporal modeling for time series SAR data.
    """
    
    def __init__(
        self,
        radar_encoder: RadarEncoder,
        temporal_layers: int = 2,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.radar_encoder = radar_encoder
        self.temporal_layers = temporal_layers
        
        # Temporal attention layers
        self.temporal_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(temporal_layers)
        ])
        
        self.temporal_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(temporal_layers)
        ])
        
        # Coherence modeling
        self.coherence_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal modeling and coherence analysis.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) where T is temporal dimension
            temporal_mask: Optional mask for temporal attention
            
        Returns:
            Dictionary containing temporal-aware features and coherence
        """
        batch_size, temporal_len = x.shape[:2]
        
        # Encode each temporal frame
        temporal_features = []
        coherence_features = []
        
        for t in range(temporal_len):
            frame_features = self.radar_encoder(x[:, t])
            temporal_features.append(frame_features['features'])
            
            # Compute coherence with previous frame if available
            if t > 0:
                coherence = self.radar_encoder.compute_coherence(
                    x[:, t-1], x[:, t]
                )
                coherence_features.append(coherence)
        
        # Stack temporal features
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Apply temporal attention
        for layer_idx in range(self.temporal_layers):
            attn_output, _ = self.temporal_attention[layer_idx](
                temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1),
                temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1),
                temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1),
                key_padding_mask=temporal_mask
            )
            
            temporal_features = temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1)
            temporal_features = self.temporal_norm[layer_idx](temporal_features + attn_output)
            temporal_features = temporal_features.view(batch_size, temporal_len, -1, temporal_features.shape[-1])
        
        # Global temporal features
        global_temporal_features = temporal_features.mean(dim=1).mean(dim=1)
        
        return {
            'temporal_features': temporal_features,
            'global_temporal_features': global_temporal_features,
            'coherence_features': coherence_features,
            'frame_features': temporal_features.mean(dim=2)
        }