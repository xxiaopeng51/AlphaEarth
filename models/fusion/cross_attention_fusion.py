"""
Cross-attention fusion module for multi-modal data integration.

This module implements cross-attention mechanisms to fuse features from different
modalities (optical, radar, meteorological, text) in the AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module for multi-modal feature integration.
    
    This module uses cross-attention mechanisms to fuse features from different
    modalities, allowing each modality to attend to relevant information from
    other modalities.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        dropout: float = 0.1,
        modalities: List[str] = ["optical", "radar", "meteorological", "text"],
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.modalities = modalities
        self.num_modalities = len(modalities)
        
        # Modality-specific projections to common dimension
        self.modality_projections = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for modality in modalities
        })
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Self-attention layers for each modality
        self.self_attention_layers = nn.ModuleDict({
            modality: nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                )
                for _ in range(num_layers)
            ])
            for modality in modalities
        })
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            modality: nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
            for modality in modalities
        })
        
        # Feed-forward networks
        self.ffns = nn.ModuleDict({
            modality: nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                )
                for _ in range(num_layers)
            ])
            for modality in modalities
        })
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * num_modalities),
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through cross-attention fusion.
        
        Args:
            modality_features: Dictionary of features from each modality
                Each tensor has shape (B, L, D) where L is sequence length
            attention_masks: Optional attention masks for each modality
            
        Returns:
            Dictionary containing:
                - 'fused_features': Fused features (B, L, D)
                - 'modality_features': Updated modality features
                - 'attention_weights': Attention weights from cross-attention
        """
        batch_size = next(iter(modality_features.values())).shape[0]
        
        # Project all modalities to common dimension
        projected_features = {}
        for modality in self.modalities:
            if modality in modality_features:
                projected_features[modality] = self.modality_projections[modality](
                    modality_features[modality]
                )
        
        # Store attention weights
        attention_weights = {}
        
        # Apply cross-attention layers
        for layer_idx in range(self.num_layers):
            # Self-attention for each modality
            for modality in self.modalities:
                if modality in projected_features:
                    # Self-attention
                    self_attn_output, self_attn_weights = self.self_attention_layers[modality][layer_idx](
                        projected_features[modality],
                        projected_features[modality],
                        projected_features[modality],
                        key_padding_mask=attention_masks.get(modality) if attention_masks else None
                    )
                    
                    # Residual connection and layer norm
                    projected_features[modality] = self.layer_norms[modality][layer_idx](
                        projected_features[modality] + self_attn_output
                    )
                    
                    # Feed-forward network
                    ffn_output = self.ffns[modality][layer_idx](projected_features[modality])
                    projected_features[modality] = projected_features[modality] + ffn_output
            
            # Cross-attention between modalities
            cross_attn_output, cross_attn_weights = self.cross_attention_layers[layer_idx](
                projected_features,
                attention_masks
            )
            
            # Update features with cross-attention
            for modality in self.modalities:
                if modality in projected_features:
                    projected_features[modality] = projected_features[modality] + cross_attn_output[modality]
            
            # Store attention weights
            attention_weights[f'layer_{layer_idx}'] = {
                'cross_attention': cross_attn_weights,
                'self_attention': {
                    modality: self.self_attention_layers[modality][layer_idx]._get_attention_weights()
                    for modality in self.modalities if modality in projected_features
                }
            }
        
        # Concatenate all modality features
        concatenated_features = torch.cat(list(projected_features.values()), dim=-1)
        
        # Final output projection
        fused_features = self.output_projection(concatenated_features)
        
        return {
            'fused_features': fused_features,
            'modality_features': projected_features,
            'attention_weights': attention_weights
        }


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer for multi-modal fusion.
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
        self.head_dim = hidden_dim // num_heads
        
        # Cross-attention for each modality pair
        self.cross_attentions = nn.ModuleDict()
        
        # Initialize cross-attention for all modality pairs
        modalities = ["optical", "radar", "meteorological", "text"]
        for query_mod in modalities:
            self.cross_attentions[query_mod] = nn.ModuleDict()
            for key_mod in modalities:
                if query_mod != key_mod:
                    self.cross_attentions[query_mod][key_mod] = nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True
                    )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through cross-attention layer.
        
        Args:
            modality_features: Dictionary of features from each modality
            attention_masks: Optional attention masks
            
        Returns:
            Tuple of (updated_features, attention_weights)
        """
        updated_features = {}
        attention_weights = {}
        
        for query_mod, query_features in modality_features.items():
            if query_mod not in self.cross_attentions:
                updated_features[query_mod] = query_features
                continue
            
            # Collect features from other modalities
            key_value_features = []
            key_value_masks = []
            
            for key_mod, key_features in modality_features.items():
                if key_mod != query_mod and key_mod in self.cross_attentions[query_mod]:
                    key_value_features.append(key_features)
                    if attention_masks and key_mod in attention_masks:
                        key_value_masks.append(attention_masks[key_mod])
            
            if not key_value_features:
                updated_features[query_mod] = query_features
                continue
            
            # Concatenate key-value features
            kv_features = torch.cat(key_value_features, dim=1)  # (B, L_kv, D)
            
            # Concatenate key-value masks if available
            kv_mask = None
            if key_value_masks:
                kv_mask = torch.cat(key_value_masks, dim=1)
            
            # Cross-attention
            attn_output, attn_weights = self.cross_attentions[query_mod][key_mod](
                query_features,
                kv_features,
                kv_features,
                key_padding_mask=kv_mask
            )
            
            # Output projection and residual connection
            updated_features[query_mod] = query_features + self.dropout(
                self.output_projection(attn_output)
            )
            
            attention_weights[query_mod] = attn_weights
        
        return updated_features, attention_weights


class HierarchicalCrossAttention(nn.Module):
    """
    Hierarchical cross-attention for multi-scale feature fusion.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_levels: int = 3,
        num_heads: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        
        # Multi-scale projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_levels)
        ])
        
        # Hierarchical cross-attention layers
        self.hierarchical_layers = nn.ModuleList([
            CrossAttentionLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_levels)
        ])
        
        # Level-specific processing
        self.level_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_levels)
        ])
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        scales: List[int] = [1, 2, 4]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical cross-attention.
        
        Args:
            modality_features: Dictionary of features from each modality
            scales: List of scale factors for multi-scale processing
            
        Returns:
            Dictionary containing hierarchical fused features
        """
        hierarchical_features = []
        
        for level in range(self.num_levels):
            # Scale features
            scale_factor = scales[level] if level < len(scales) else 1
            
            scaled_features = {}
            for modality, features in modality_features.items():
                if scale_factor > 1:
                    # Downsample features
                    scaled_features[modality] = F.avg_pool1d(
                        features.transpose(1, 2),
                        kernel_size=scale_factor,
                        stride=scale_factor
                    ).transpose(1, 2)
                else:
                    scaled_features[modality] = features
            
            # Project to common dimension
            projected_features = {}
            for modality, features in scaled_features.items():
                projected_features[modality] = self.scale_projections[level](features)
            
            # Apply cross-attention
            fused_features, _ = self.hierarchical_layers[level](projected_features)
            
            # Process level-specific features
            level_features = []
            for modality, features in fused_features.items():
                level_features.append(self.level_processors[level](features))
            
            # Combine modality features at this level
            level_combined = torch.cat(level_features, dim=-1)
            hierarchical_features.append(level_combined)
        
        # Final hierarchical fusion
        final_features = torch.cat(hierarchical_features, dim=-1)
        output_features = self.final_fusion(final_features)
        
        return {
            'hierarchical_features': hierarchical_features,
            'fused_features': output_features
        }