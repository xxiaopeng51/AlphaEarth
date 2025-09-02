"""
Spatial-Temporal Precision (STP) module for AlphaEarth Foundations model.

This module implements the core STP module that captures spatial, temporal,
and resolution details for global-scale earth observation modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class SpatialTemporalPrecision(nn.Module):
    """
    Spatial-Temporal Precision (STP) module for capturing multi-scale spatiotemporal patterns.
    
    This module processes spatial, temporal, and resolution information to capture
    long-range geographical dependencies and temporal dynamics in earth observation data.
    """
    
    def __init__(
        self,
        spatial_attention_layers: int = 3,
        temporal_attention_layers: int = 2,
        resolution_attention_layers: int = 2,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1,
        max_spatial_distance: int = 1000,
        max_temporal_distance: int = 365,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_spatial_distance = max_spatial_distance
        self.max_temporal_distance = max_temporal_distance
        
        # Spatial attention layers
        self.spatial_attention = nn.ModuleList([
            SpatialAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_distance=max_spatial_distance
            )
            for _ in range(spatial_attention_layers)
        ])
        
        # Temporal attention layers
        self.temporal_attention = nn.ModuleList([
            TemporalAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                max_distance=max_temporal_distance
            )
            for _ in range(temporal_attention_layers)
        ])
        
        # Resolution attention layers
        self.resolution_attention = nn.ModuleList([
            ResolutionAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(resolution_attention_layers)
        ])
        
        # Spatial encoding
        self.spatial_encoding = SpatialEncoding(hidden_dim)
        
        # Temporal encoding
        self.temporal_encoding = TemporalEncoding(hidden_dim)
        
        # Resolution encoding
        self.resolution_encoding = ResolutionEncoding(hidden_dim)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None,
        temporal_coords: Optional[torch.Tensor] = None,
        resolution_info: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through STP module.
        
        Args:
            features: Input features (B, L, D)
            spatial_coords: Spatial coordinates (B, L, 2) [lat, lon]
            temporal_coords: Temporal coordinates (B, L, 1) [timestamp]
            resolution_info: Resolution information (B, L, 1) [resolution_level]
            attention_masks: Optional attention masks (B, L)
            
        Returns:
            Dictionary containing:
                - 'spatial_features': Spatial-aware features
                - 'temporal_features': Temporal-aware features
                - 'resolution_features': Resolution-aware features
                - 'fused_features': Fused STP features
        """
        batch_size, seq_len, _ = features.shape
        
        # Add positional encodings
        if spatial_coords is not None:
            spatial_enc = self.spatial_encoding(spatial_coords)
            features = features + spatial_enc
        
        if temporal_coords is not None:
            temporal_enc = self.temporal_encoding(temporal_coords)
            features = features + temporal_enc
        
        if resolution_info is not None:
            resolution_enc = self.resolution_encoding(resolution_info)
            features = features + resolution_enc
        
        # Apply spatial attention
        spatial_features = features
        for layer in self.spatial_attention:
            spatial_features = layer(
                spatial_features,
                spatial_coords,
                attention_masks
            )
        
        # Apply temporal attention
        temporal_features = features
        for layer in self.temporal_attention:
            temporal_features = layer(
                temporal_features,
                temporal_coords,
                attention_masks
            )
        
        # Apply resolution attention
        resolution_features = features
        for layer in self.resolution_attention:
            resolution_features = layer(
                resolution_features,
                resolution_info,
                attention_masks
            )
        
        # Fuse spatial, temporal, and resolution features
        combined_features = torch.cat([
            spatial_features,
            temporal_features,
            resolution_features
        ], dim=-1)
        
        fused_features = self.feature_fusion(combined_features)
        output_features = self.output_projection(fused_features)
        
        return {
            'spatial_features': spatial_features,
            'temporal_features': temporal_features,
            'resolution_features': resolution_features,
            'fused_features': output_features
        }


class SpatialAttentionLayer(nn.Module):
    """
    Spatial attention layer with geographical distance awareness.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1,
        max_distance: int = 1000
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_distance = max_distance
        
        # Attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Spatial distance encoding
        self.distance_encoding = nn.Embedding(max_distance, num_heads)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through spatial attention layer.
        
        Args:
            features: Input features (B, L, D)
            spatial_coords: Spatial coordinates (B, L, 2)
            attention_masks: Optional attention masks
            
        Returns:
            Spatial-aware features (B, L, D)
        """
        # Standard attention
        attn_output, _ = self.attention(
            features, features, features,
            key_padding_mask=attention_masks
        )
        
        # Add spatial distance bias if coordinates are provided
        if spatial_coords is not None:
            spatial_bias = self._compute_spatial_bias(spatial_coords)
            attn_output = attn_output + spatial_bias
        
        # Residual connection and layer norm
        features = self.layer_norm(features + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(features)
        features = features + ffn_output
        
        return features
    
    def _compute_spatial_bias(
        self,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial distance bias for attention.
        
        Args:
            spatial_coords: Spatial coordinates (B, L, 2)
            
        Returns:
            Spatial bias (B, L, D)
        """
        batch_size, seq_len, _ = spatial_coords.shape
        
        # Compute pairwise distances
        coords_expanded = spatial_coords.unsqueeze(2)  # (B, L, 1, 2)
        coords_transposed = spatial_coords.unsqueeze(1)  # (B, 1, L, 2)
        
        # Euclidean distance in lat-lon space (simplified)
        distances = torch.norm(coords_expanded - coords_transposed, p=2, dim=-1)
        
        # Quantize distances for embedding lookup
        distance_indices = torch.clamp(
            (distances * 100).long(),  # Scale and quantize
            0, self.max_distance - 1
        )
        
        # Get distance embeddings
        distance_embeddings = self.distance_encoding(distance_indices)  # (B, L, L, num_heads)
        
        # Reshape to match attention output
        distance_bias = distance_embeddings.permute(0, 3, 1, 2)  # (B, num_heads, L, L)
        distance_bias = distance_bias.reshape(batch_size, seq_len, self.hidden_dim)
        
        return distance_bias


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer with temporal distance awareness.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1,
        max_distance: int = 365
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_distance = max_distance
        
        # Attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Temporal distance encoding
        self.distance_encoding = nn.Embedding(max_distance, num_heads)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through temporal attention layer.
        
        Args:
            features: Input features (B, L, D)
            temporal_coords: Temporal coordinates (B, L, 1)
            attention_masks: Optional attention masks
            
        Returns:
            Temporal-aware features (B, L, D)
        """
        # Standard attention
        attn_output, _ = self.attention(
            features, features, features,
            key_padding_mask=attention_masks
        )
        
        # Add temporal distance bias if coordinates are provided
        if temporal_coords is not None:
            temporal_bias = self._compute_temporal_bias(temporal_coords)
            attn_output = attn_output + temporal_bias
        
        # Residual connection and layer norm
        features = self.layer_norm(features + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(features)
        features = features + ffn_output
        
        return features
    
    def _compute_temporal_bias(
        self,
        temporal_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal distance bias for attention.
        
        Args:
            temporal_coords: Temporal coordinates (B, L, 1)
            
        Returns:
            Temporal bias (B, L, D)
        """
        batch_size, seq_len, _ = temporal_coords.shape
        
        # Compute pairwise temporal distances
        coords_expanded = temporal_coords.unsqueeze(2)  # (B, L, 1, 1)
        coords_transposed = temporal_coords.unsqueeze(1)  # (B, 1, L, 1)
        
        distances = torch.abs(coords_expanded - coords_transposed).squeeze(-1)
        
        # Quantize distances for embedding lookup
        distance_indices = torch.clamp(
            distances.long(),
            0, self.max_distance - 1
        )
        
        # Get distance embeddings
        distance_embeddings = self.distance_encoding(distance_indices)  # (B, L, L, num_heads)
        
        # Reshape to match attention output
        distance_bias = distance_embeddings.permute(0, 3, 1, 2)  # (B, num_heads, L, L)
        distance_bias = distance_bias.reshape(batch_size, seq_len, self.hidden_dim)
        
        return distance_bias


class ResolutionAttentionLayer(nn.Module):
    """
    Resolution attention layer for multi-resolution feature processing.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Resolution-aware processing
        self.resolution_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        features: torch.Tensor,
        resolution_info: Optional[torch.Tensor] = None,
        attention_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through resolution attention layer.
        
        Args:
            features: Input features (B, L, D)
            resolution_info: Resolution information (B, L, 1)
            attention_masks: Optional attention masks
            
        Returns:
            Resolution-aware features (B, L, D)
        """
        # Apply resolution-aware processing
        if resolution_info is not None:
            # Scale features based on resolution
            resolution_scale = 1.0 / (resolution_info + 1e-8)
            features = features * resolution_scale
        
        # Standard attention
        attn_output, _ = self.attention(
            features, features, features,
            key_padding_mask=attention_masks
        )
        
        # Residual connection and layer norm
        features = self.layer_norm(features + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(features)
        features = features + ffn_output
        
        return features


class SpatialEncoding(nn.Module):
    """
    Spatial positional encoding for geographical coordinates.
    """
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Learnable spatial embeddings
        self.lat_embedding = nn.Linear(1, hidden_dim // 2)
        self.lon_embedding = nn.Linear(1, hidden_dim // 2)
        
    def forward(self, spatial_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial encoding.
        
        Args:
            spatial_coords: Spatial coordinates (B, L, 2) [lat, lon]
            
        Returns:
            Spatial encoding (B, L, D)
        """
        lat, lon = spatial_coords[..., 0:1], spatial_coords[..., 1:2]
        
        lat_enc = self.lat_embedding(lat)
        lon_enc = self.lon_embedding(lon)
        
        spatial_enc = torch.cat([lat_enc, lon_enc], dim=-1)
        
        return spatial_enc


class TemporalEncoding(nn.Module):
    """
    Temporal positional encoding for timestamps.
    """
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Learnable temporal embeddings
        self.temporal_embedding = nn.Linear(1, hidden_dim)
        
    def forward(self, temporal_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal encoding.
        
        Args:
            temporal_coords: Temporal coordinates (B, L, 1)
            
        Returns:
            Temporal encoding (B, L, D)
        """
        return self.temporal_embedding(temporal_coords)


class ResolutionEncoding(nn.Module):
    """
    Resolution positional encoding for resolution information.
    """
    
    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Learnable resolution embeddings
        self.resolution_embedding = nn.Linear(1, hidden_dim)
        
    def forward(self, resolution_info: torch.Tensor) -> torch.Tensor:
        """
        Compute resolution encoding.
        
        Args:
            resolution_info: Resolution information (B, L, 1)
            
        Returns:
            Resolution encoding (B, L, D)
        """
        return self.resolution_embedding(resolution_info)