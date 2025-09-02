"""
Meteorological data encoder for AlphaEarth Foundations model.

This module implements the meteorological encoder that processes weather and climate data
from sources like ERA5, GFS, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class MeteorologicalEncoder(nn.Module):
    """
    Meteorological data encoder for weather and climate data processing.
    
    This encoder processes meteorological variables like temperature, precipitation,
    humidity, pressure, wind speed/direction, etc., and extracts temporal-spatial
    patterns for weather prediction and climate analysis.
    """
    
    def __init__(
        self,
        input_dim: int = 128,  # Number of meteorological variables
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        output_dim: int = 512,
        dropout: float = 0.1,
        max_sequence_length: int = 1000,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_sequence_length = max_sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for temporal sequences
        self.positional_encoding = nn.Parameter(
            torch.randn(max_sequence_length, hidden_dim)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Meteorological variable embeddings
        self.variable_embeddings = nn.Embedding(input_dim, hidden_dim)
        
        # Spatial encoding for gridded data
        self.spatial_encoding = nn.Parameter(
            torch.randn(1, 1, hidden_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        variable_ids: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the meteorological encoder.
        
        Args:
            x: Input meteorological data tensor of shape (B, T, V) or (B, T, H, W, V)
               where T is temporal dimension, V is variable dimension
            variable_ids: Optional variable IDs for variable-specific embeddings
            temporal_mask: Optional mask for temporal attention
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - 'features': Encoded features (B, T, D) or (B, T*H*W, D)
                - 'global_features': Global pooled features (B, D)
                - 'temporal_features': Temporal features if return_features=True
        """
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # Handle different input shapes
        if x.dim() == 3:  # (B, T, V)
            seq_len = x.shape[1]
            x_flat = x
        elif x.dim() == 5:  # (B, T, H, W, V)
            seq_len = x.shape[1]
            height, width = x.shape[2], x.shape[3]
            x_flat = x.view(batch_size, seq_len, height * width, -1)
            x_flat = x_flat.view(batch_size, seq_len * height * width, -1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        
        # Input projection
        x_projected = self.input_projection(x_flat)
        
        # Add variable embeddings if provided
        if variable_ids is not None:
            if variable_ids.dim() == 2:  # (B, T)
                variable_ids = variable_ids.unsqueeze(-1).expand(-1, -1, x_flat.shape[-1])
            elif variable_ids.dim() == 4:  # (B, T, H, W)
                variable_ids = variable_ids.view(batch_size, seq_len * height * width, 1)
                variable_ids = variable_ids.expand(-1, -1, x_flat.shape[-1])
            
            variable_emb = self.variable_embeddings(variable_ids)
            x_projected = x_projected + variable_emb
        
        # Add positional encoding
        seq_length = x_projected.shape[1]
        if seq_length <= self.max_sequence_length:
            pos_encoding = self.positional_encoding[:seq_length]
            x_projected = x_projected + pos_encoding.unsqueeze(0)
        else:
            # For longer sequences, interpolate positional encoding
            pos_encoding = F.interpolate(
                self.positional_encoding.T.unsqueeze(0),
                size=seq_length,
                mode='linear',
                align_corners=False
            ).squeeze(0).T
            x_projected = x_projected + pos_encoding.unsqueeze(0)
        
        # Add spatial encoding for gridded data
        if x.dim() == 5:
            x_projected = x_projected + self.spatial_encoding
        
        # Apply transformer encoder
        encoded_features = self.transformer_encoder(
            x_projected, 
            src_key_padding_mask=temporal_mask
        )
        
        # Output projection
        projected_features = self.output_projection(encoded_features)
        
        # Global pooling
        if temporal_mask is not None:
            # Masked global pooling
            mask_expanded = temporal_mask.unsqueeze(-1).expand_as(projected_features)
            masked_features = projected_features.masked_fill(mask_expanded, 0)
            global_features = masked_features.sum(dim=1) / (~mask_expanded).sum(dim=1).clamp(min=1)
        else:
            global_features = projected_features.mean(dim=1)
        
        result = {
            'features': projected_features,
            'global_features': global_features
        }
        
        if return_features:
            result['temporal_features'] = projected_features
        
        return result
    
    def encode_weather_forecast(
        self,
        historical_data: torch.Tensor,
        forecast_horizon: int = 24
    ) -> Dict[str, torch.Tensor]:
        """
        Encode historical weather data for forecasting.
        
        Args:
            historical_data: Historical weather data (B, T_hist, V)
            forecast_horizon: Number of time steps to forecast
            
        Returns:
            Dictionary containing forecast features
        """
        # Encode historical data
        hist_features = self.forward(historical_data)
        
        # Generate forecast features using autoregressive approach
        forecast_features = []
        current_features = hist_features['features'][:, -1:, :]  # Last timestep
        
        for _ in range(forecast_horizon):
            # Simple autoregressive prediction (can be enhanced with more sophisticated methods)
            next_features = self.output_projection(current_features)
            forecast_features.append(next_features)
            current_features = next_features
        
        forecast_features = torch.cat(forecast_features, dim=1)
        
        return {
            'historical_features': hist_features['features'],
            'forecast_features': forecast_features,
            'global_forecast_features': forecast_features.mean(dim=1)
        }
    
    def compute_weather_anomalies(
        self,
        current_data: torch.Tensor,
        climatology_data: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weather anomalies by comparing current data with climatology.
        
        Args:
            current_data: Current weather data (B, T, V)
            climatology_data: Climatological averages (B, T, V)
            
        Returns:
            Dictionary containing anomaly features
        """
        # Encode current and climatology data
        current_features = self.forward(current_data)
        climatology_features = self.forward(climatology_data)
        
        # Compute anomalies
        anomaly_features = current_features['features'] - climatology_features['features']
        
        # Compute anomaly magnitude
        anomaly_magnitude = torch.norm(anomaly_features, dim=-1, keepdim=True)
        
        return {
            'anomaly_features': anomaly_features,
            'anomaly_magnitude': anomaly_magnitude,
            'current_features': current_features['features'],
            'climatology_features': climatology_features['features']
        }


class MeteorologicalEncoderWithSpatial(nn.Module):
    """
    Meteorological encoder with spatial modeling for gridded weather data.
    """
    
    def __init__(
        self,
        met_encoder: MeteorologicalEncoder,
        spatial_layers: int = 2,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.met_encoder = met_encoder
        self.spatial_layers = spatial_layers
        
        # Spatial attention layers
        self.spatial_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(spatial_layers)
        ])
        
        self.spatial_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(spatial_layers)
        ])
        
        # Spatial convolution for local patterns
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with spatial modeling for gridded data.
        
        Args:
            x: Input tensor of shape (B, T, H, W, V)
            spatial_mask: Optional mask for spatial attention
            
        Returns:
            Dictionary containing spatial-aware features
        """
        batch_size, temporal_len, height, width = x.shape[:4]
        
        # Encode each temporal frame
        temporal_features = []
        for t in range(temporal_len):
            frame_data = x[:, t]  # (B, H, W, V)
            frame_features = self.met_encoder(frame_data.view(batch_size, height * width, -1))
            temporal_features.append(frame_features['features'])
        
        # Stack temporal features: (B, T, H*W, D)
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Reshape for spatial processing: (B*T, H*W, D)
        spatial_features = temporal_features.view(batch_size * temporal_len, height * width, -1)
        
        # Apply spatial attention
        for layer_idx in range(self.spatial_layers):
            attn_output, _ = self.spatial_attention[layer_idx](
                spatial_features, spatial_features, spatial_features,
                key_padding_mask=spatial_mask
            )
            spatial_features = self.spatial_norm[layer_idx](spatial_features + attn_output)
        
        # Reshape back: (B, T, H, W, D)
        spatial_features = spatial_features.view(batch_size, temporal_len, height, width, -1)
        
        # Apply spatial convolution
        conv_features = []
        for t in range(temporal_len):
            frame_conv = self.spatial_conv(spatial_features[:, t].permute(0, 3, 1, 2))
            conv_features.append(frame_conv.permute(0, 2, 3, 1))
        
        conv_features = torch.stack(conv_features, dim=1)
        
        # Combine attention and convolution features
        combined_features = spatial_features + conv_features
        
        # Global features
        global_features = combined_features.mean(dim=(1, 2, 3))  # (B, D)
        
        return {
            'spatial_features': combined_features,
            'global_spatial_features': global_features,
            'temporal_features': combined_features.mean(dim=(2, 3))  # (B, T, D)
        }