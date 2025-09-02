"""
Segmentation head for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SegmentationHead(nn.Module):
    """
    Segmentation head for pixel-level classification tasks.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 20,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        decoder_type: str = "unet",
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.decoder_type = decoder_type
        
        if decoder_type == "unet":
            self.decoder = UNetDecoder(input_dim, num_classes, hidden_dim, dropout)
        elif decoder_type == "fpn":
            self.decoder = FPNDecoder(input_dim, num_classes, hidden_dim, dropout)
        elif decoder_type == "psp":
            self.decoder = PSPDecoder(input_dim, num_classes, hidden_dim, dropout)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through segmentation head.
        
        Args:
            features: Input features (B, L, D) or (B, D)
            spatial_coords: Optional spatial coordinates for spatial-aware segmentation
            
        Returns:
            Dictionary containing:
                - 'logits': Segmentation logits (B, num_classes, H, W)
                - 'probabilities': Segmentation probabilities (B, num_classes, H, W)
                - 'predictions': Segmentation predictions (B, H, W)
        """
        # Reshape features if needed
        if features.dim() == 2:  # (B, D)
            # Assume square spatial layout
            batch_size = features.shape[0]
            spatial_size = int((features.shape[1] // self.input_dim) ** 0.5)
            features = features.view(batch_size, spatial_size, spatial_size, -1)
            features = features.permute(0, 3, 1, 2)  # (B, D, H, W)
        elif features.dim() == 3:  # (B, L, D)
            # Reshape from sequence to spatial
            batch_size, seq_len, feat_dim = features.shape
            spatial_size = int(seq_len ** 0.5)
            features = features.view(batch_size, spatial_size, spatial_size, feat_dim)
            features = features.permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # Decode features
        logits = self.decoder(features, spatial_coords)
        
        # Compute probabilities and predictions
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }


class UNetDecoder(nn.Module):
    """
    U-Net style decoder for segmentation.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(input_dim, hidden_dim, dropout),
            DecoderBlock(hidden_dim, hidden_dim // 2, dropout),
            DecoderBlock(hidden_dim // 2, hidden_dim // 4, dropout),
            DecoderBlock(hidden_dim // 4, hidden_dim // 8, dropout)
        ])
        
        # Final classification layer
        self.classifier = nn.Conv2d(hidden_dim // 8, num_classes, kernel_size=1)
        
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through U-Net decoder."""
        x = features
        
        # Decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network decoder for segmentation.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        ])
        
        # FPN blocks
        self.fpn_blocks = nn.ModuleList([
            FPNBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(4)
        ])
        
        # Final classification layer
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through FPN decoder."""
        # Lateral connection
        x = self.lateral_convs[0](features)
        
        # FPN blocks
        for block in self.fpn_blocks:
            x = block(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits


class PSPDecoder(nn.Module):
    """
    Pyramid Scene Parsing decoder for segmentation.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Pyramid pooling module
        self.pyramid_pooling = PyramidPooling(input_dim, hidden_dim, dropout)
        
        # Final classification layer
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through PSP decoder."""
        # Pyramid pooling
        x = self.pyramid_pooling(features)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits


class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net style architecture.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block."""
        return self.block(x)


class FPNBlock(nn.Module):
    """
    Feature Pyramid Network block.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FPN block."""
        return self.block(x)


class PyramidPooling(nn.Module):
    """
    Pyramid pooling module for PSP decoder.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Different pooling sizes
        self.pool_sizes = [1, 2, 3, 6]
        
        # Pooling branches
        self.pool_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Conv2d(in_channels, out_channels // len(self.pool_sizes), kernel_size=1),
                nn.BatchNorm2d(out_channels // len(self.pool_sizes)),
                nn.ReLU(inplace=True)
            )
            for pool_size in self.pool_sizes
        ])
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pyramid pooling."""
        batch_size, channels, height, width = x.shape
        
        # Pooling branches
        pool_outputs = []
        for branch in self.pool_branches:
            pool_out = branch(x)
            pool_out = F.interpolate(pool_out, size=(height, width), mode='bilinear', align_corners=False)
            pool_outputs.append(pool_out)
        
        # Concatenate with original features
        concatenated = torch.cat([x] + pool_outputs, dim=1)
        
        # Final projection
        output = self.final_proj(concatenated)
        
        return output