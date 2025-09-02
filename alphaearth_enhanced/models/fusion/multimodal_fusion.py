"""
Multimodal Fusion Module for Combining Different Data Sources
Implements various fusion strategies including cross-attention and gated fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat
from timm.models.layers import DropPath


class CrossModalAttention(nn.Module):
    """Cross-modal attention for feature interaction"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-modal attention
        
        Args:
            query: Query features from modality A [B, N, D]
            key: Key features from modality B [B, M, D]
            value: Value features from modality B [B, M, D]
            mask: Optional attention mask
        
        Returns:
            Attended features [B, N, D]
        """
        B, N, D = query.shape
        _, M, _ = key.shape
        
        # Project and reshape
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining modalities"""
    
    def __init__(self, dim: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        
        # Gates for each modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim * num_modalities, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(dim * num_modalities, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply gated fusion
        
        Args:
            features: List of feature tensors [B, N, D]
        
        Returns:
            Fused features [B, N, D]
        """
        # Concatenate all features
        concat_features = torch.cat(features, dim=-1)
        
        # Compute gates
        gates = []
        for i, gate in enumerate(self.gates):
            g = gate(concat_features)
            gates.append(g)
        
        # Normalize gates
        gates = torch.cat(gates, dim=-1)
        gates = F.softmax(gates, dim=-1)
        
        # Apply gates
        weighted_features = []
        for i, feat in enumerate(features):
            weighted = feat * gates[:, :, i:i+1]
            weighted_features.append(weighted)
        
        # Sum weighted features
        fused = sum(weighted_features)
        
        # Additional fusion layer
        fused = self.fusion(concat_features) + fused
        
        return fused


class MultimodalTransformerBlock(nn.Module):
    """Transformer block for multimodal fusion"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = norm_layer(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        
        # Cross-modal attention
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossModalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        # MLP
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features [B, N, D]
            context: Optional context features for cross-attention [B, M, D]
        
        Returns:
            Output features [B, N, D]
        """
        # Self-attention
        x = x + self.drop_path(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        
        # Cross-modal attention if context provided
        if context is not None:
            x = x + self.drop_path(self.cross_attn(self.norm2(x), context, context))
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x


class MultimodalFusion(nn.Module):
    """
    Main multimodal fusion module
    
    Supports multiple fusion strategies:
    - Early fusion: Concatenate and process
    - Late fusion: Process separately then combine
    - Cross-attention fusion: Interactive processing
    - Hierarchical fusion: Multi-level combination
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        fusion_type: str = "cross_attention",  # "early", "late", "cross_attention", "gated"
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        
        # Modality-specific projection layers
        self.modality_projections = nn.ModuleDict()
        
        # Positional encoding for different modalities
        self.modality_embeddings = nn.ParameterDict()
        
        if fusion_type == "cross_attention":
            # Cross-attention fusion blocks
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.fusion_blocks = nn.ModuleList([
                MultimodalTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ])
        
        elif fusion_type == "gated":
            # Gated fusion (will be initialized dynamically)
            self.gated_fusion = None
        
        elif fusion_type == "early":
            # Early fusion: simple concatenation and projection
            self.early_fusion = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(drop_rate),
                nn.Linear(embed_dim * 2, embed_dim),
            )
        
        elif fusion_type == "late":
            # Late fusion: weighted combination
            self.late_fusion_weights = None  # Will be initialized dynamically
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def add_modality(self, name: str, input_dim: int):
        """Add a new modality to the fusion module"""
        if input_dim != self.embed_dim:
            self.modality_projections[name] = nn.Linear(input_dim, self.embed_dim)
        
        # Add modality-specific positional encoding
        self.modality_embeddings[name] = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Fuse multimodal features
        
        Args:
            features: Dictionary of features from different modalities
                     Each tensor should be [B, N, D] where N can vary
        
        Returns:
            Fused features [B, N, D]
        """
        # Project features to common dimension if needed
        projected_features = {}
        for name, feat in features.items():
            if name in self.modality_projections:
                feat = self.modality_projections[name](feat)
            
            # Add modality embedding if available
            if name in self.modality_embeddings:
                feat = feat + self.modality_embeddings[name]
            
            projected_features[name] = feat
        
        if self.fusion_type == "cross_attention":
            return self._cross_attention_fusion(projected_features)
        elif self.fusion_type == "gated":
            return self._gated_fusion(projected_features)
        elif self.fusion_type == "early":
            return self._early_fusion(projected_features)
        elif self.fusion_type == "late":
            return self._late_fusion(projected_features)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
    
    def _cross_attention_fusion(
        self,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Cross-attention based fusion"""
        # Use first modality as primary, others as context
        modality_names = list(features.keys())
        primary = features[modality_names[0]]
        
        # If only one modality, return it
        if len(modality_names) == 1:
            return self.norm(primary)
        
        # Concatenate other modalities as context
        context_features = []
        for name in modality_names[1:]:
            context_features.append(features[name])
        
        if context_features:
            context = torch.cat(context_features, dim=1)
        else:
            context = None
        
        # Apply fusion blocks
        x = primary
        for block in self.fusion_blocks:
            x = block(x, context)
        
        return self.norm(x)
    
    def _gated_fusion(
        self,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Gated fusion"""
        # Initialize gated fusion if needed
        if self.gated_fusion is None:
            self.gated_fusion = GatedFusion(self.embed_dim, len(features))
        
        # Convert to list and ensure same sequence length
        feat_list = list(features.values())
        
        # Pad to same length if needed
        max_len = max(f.shape[1] for f in feat_list)
        padded_features = []
        for feat in feat_list:
            if feat.shape[1] < max_len:
                padding = torch.zeros(
                    feat.shape[0], max_len - feat.shape[1], feat.shape[2],
                    device=feat.device
                )
                feat = torch.cat([feat, padding], dim=1)
            padded_features.append(feat)
        
        return self.norm(self.gated_fusion(padded_features))
    
    def _early_fusion(
        self,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Early fusion by concatenation"""
        # Average pool features to same sequence length
        feat_list = list(features.values())
        
        # Use mean pooling to get single vector per modality
        pooled_features = []
        for feat in feat_list:
            pooled = feat.mean(dim=1, keepdim=True)  # [B, 1, D]
            pooled_features.append(pooled)
        
        # Concatenate and process
        concatenated = torch.cat(pooled_features, dim=1)  # [B, M, D]
        fused = concatenated.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        return self.norm(self.early_fusion(fused))
    
    def _late_fusion(
        self,
        features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Late fusion with learned weights"""
        # Initialize weights if needed
        if self.late_fusion_weights is None:
            num_modalities = len(features)
            self.late_fusion_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        
        # Weighted combination
        feat_list = list(features.values())
        weights = F.softmax(self.late_fusion_weights, dim=0)
        
        # Average pool to same size and combine
        fused = None
        for i, feat in enumerate(feat_list):
            pooled = feat.mean(dim=1, keepdim=True)
            weighted = pooled * weights[i]
            
            if fused is None:
                fused = weighted
            else:
                fused = fused + weighted
        
        return self.norm(fused)