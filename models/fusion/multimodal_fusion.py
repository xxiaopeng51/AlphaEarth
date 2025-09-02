"""
Multi-modal fusion module for combining features from different modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module for combining features from different modalities.
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_modalities: int = 4,
        dropout: float = 0.1,
        fusion_type: str = "attention"
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        if fusion_type == "attention":
            self.fusion_layer = AttentionFusion(hidden_dim, num_modalities, dropout)
        elif fusion_type == "concat":
            self.fusion_layer = ConcatFusion(hidden_dim, num_modalities, dropout)
        elif fusion_type == "gated":
            self.fusion_layer = GatedFusion(hidden_dim, num_modalities, dropout)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        stp_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal fusion.
        
        Args:
            modality_features: Dictionary of features from each modality
            stp_features: Spatial-temporal precision features
            
        Returns:
            Dictionary containing fused features
        """
        return self.fusion_layer(modality_features, stp_features)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multi-modal features.
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
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
        modality_features: Dict[str, torch.Tensor],
        stp_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through attention fusion."""
        # Stack modality features
        modality_list = list(modality_features.values())
        stacked_features = torch.stack(modality_list, dim=1)  # (B, M, D)
        
        # Use STP features as query
        query = stp_features.unsqueeze(1)  # (B, 1, D)
        
        # Attention fusion
        fused_features, attention_weights = self.attention(
            query, stacked_features, stacked_features
        )
        
        # Output projection
        output_features = self.output_projection(fused_features.squeeze(1))
        
        return {
            'multimodal_features': stacked_features,
            'global_features': output_features,
            'attention_weights': attention_weights
        }


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion of multi-modal features.
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * (num_modalities + 1), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        stp_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through concat fusion."""
        # Concatenate all features
        feature_list = list(modality_features.values()) + [stp_features]
        concatenated_features = torch.cat(feature_list, dim=-1)
        
        # Output projection
        output_features = self.output_projection(concatenated_features)
        
        return {
            'multimodal_features': torch.stack(list(modality_features.values()), dim=1),
            'global_features': output_features
        }


class GatedFusion(nn.Module):
    """
    Gated fusion of multi-modal features.
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # Gating mechanism
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        stp_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through gated fusion."""
        modality_list = list(modality_features.values())
        gated_features = []
        
        for i, (modality_feat, gate) in enumerate(zip(modality_list, self.gates)):
            # Compute gate
            gate_input = torch.cat([modality_feat, stp_features], dim=-1)
            gate_weights = gate(gate_input)
            
            # Apply gating
            gated_feat = modality_feat * gate_weights
            gated_features.append(gated_feat)
        
        # Combine gated features
        combined_features = torch.stack(gated_features, dim=1).mean(dim=1)
        
        # Output projection
        output_features = self.output_projection(combined_features)
        
        return {
            'multimodal_features': torch.stack(modality_list, dim=1),
            'global_features': output_features
        }