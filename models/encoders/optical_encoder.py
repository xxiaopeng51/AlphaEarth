"""
Optical satellite imagery encoder for AlphaEarth Foundations model.

This module implements the optical encoder that processes multi-spectral satellite imagery
from sources like Sentinel-2, Landsat, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import timm
from torchvision import transforms


class OpticalEncoder(nn.Module):
    """
    Optical satellite imagery encoder using Vision Transformer architecture.
    
    This encoder processes multi-spectral satellite imagery and extracts
    spatial-spectral features for downstream tasks.
    """
    
    def __init__(
        self,
        model_name: str = "vit_large_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 1024,
        input_channels: int = 13,  # Sentinel-2 bands
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
            num_classes=0,  # Remove classification head
            global_pool='',  # No global pooling
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
        
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the optical encoder.
        
        Args:
            x: Input optical imagery tensor of shape (B, C, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - 'features': Encoded features (B, N, D)
                - 'global_features': Global pooled features (B, D)
                - 'spatial_features': Spatial features (B, H*W, D) if return_features=True
        """
        batch_size = x.shape[0]
        
        # Input projection for different channel numbers
        x = self.input_projection(x)
        
        # Extract features using backbone
        if hasattr(self.backbone, 'forward_features'):
            # For ViT models
            features = self.backbone.forward_features(x)
        else:
            # For other models, use forward
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
            'global_features': global_features
        }
        
        if return_features:
            result['spatial_features'] = projected_features[:, 1:, :]  # Remove CLS token
        
        return result
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get patch embeddings for spatial analysis.
        
        Args:
            x: Input optical imagery tensor
            
        Returns:
            Patch embeddings of shape (B, H*W, D)
        """
        # Input projection
        x = self.input_projection(x)
        
        # Extract patch embeddings
        if hasattr(self.backbone, 'patch_embed'):
            patch_embeddings = self.backbone.patch_embed(x)
        else:
            # Fallback for other architectures
            features = self.backbone.forward_features(x)
            patch_embeddings = features[:, 1:, :]  # Remove CLS token
        
        return patch_embeddings
    
    def encode_multiscale(
        self, 
        x: torch.Tensor,
        scales: List[int] = [224, 448, 896]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode optical imagery at multiple scales for multi-resolution analysis.
        
        Args:
            x: Input optical imagery tensor
            scales: List of scales to process
            
        Returns:
            Dictionary containing features at different scales
        """
        multiscale_features = {}
        
        for scale in scales:
            # Resize input to scale
            x_scaled = F.interpolate(
                x, size=(scale, scale), 
                mode='bilinear', align_corners=False
            )
            
            # Encode at this scale
            features = self.forward(x_scaled)
            multiscale_features[f'scale_{scale}'] = features
        
        return multiscale_features


class OpticalEncoderWithTemporal(nn.Module):
    """
    Optical encoder with temporal modeling capabilities for time series data.
    """
    
    def __init__(
        self,
        optical_encoder: OpticalEncoder,
        temporal_layers: int = 2,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.optical_encoder = optical_encoder
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
        
        self.temporal_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(temporal_layers)
        ])
        
    def forward(
        self, 
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal modeling.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) where T is temporal dimension
            temporal_mask: Optional mask for temporal attention
            
        Returns:
            Dictionary containing temporal-aware features
        """
        batch_size, temporal_len = x.shape[:2]
        
        # Encode each temporal frame
        temporal_features = []
        for t in range(temporal_len):
            frame_features = self.optical_encoder(x[:, t])
            temporal_features.append(frame_features['features'])
        
        # Stack temporal features: (B, T, N, D)
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Apply temporal attention
        for layer_idx in range(self.temporal_layers):
            # Self-attention across temporal dimension
            attn_output, _ = self.temporal_attention[layer_idx](
                temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1),
                temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1),
                temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1),
                key_padding_mask=temporal_mask
            )
            
            # Residual connection and layer norm
            temporal_features = temporal_features.view(batch_size * temporal_features.shape[2], temporal_len, -1)
            temporal_features = self.temporal_norm[layer_idx](temporal_features + attn_output)
            
            # Feed-forward network
            ffn_output = self.temporal_ffn[layer_idx](temporal_features)
            temporal_features = temporal_features + ffn_output
            
            # Reshape back
            temporal_features = temporal_features.view(batch_size, temporal_len, -1, temporal_features.shape[-1])
        
        # Global temporal features
        global_temporal_features = temporal_features.mean(dim=1).mean(dim=1)  # (B, D)
        
        return {
            'temporal_features': temporal_features,
            'global_temporal_features': global_temporal_features,
            'frame_features': temporal_features.mean(dim=2)  # (B, T, D)
        }