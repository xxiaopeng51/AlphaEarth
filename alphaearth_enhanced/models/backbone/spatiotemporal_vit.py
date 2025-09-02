"""
Spatiotemporal Vision Transformer for Earth Observation
Inspired by Prithvi's architecture with 3D positional encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for spatiotemporal data"""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 4,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.num_patches_per_frame = (img_size // patch_size) ** 2
        self.num_patches = self.num_patches_per_frame * num_frames
        
        # Use 3D convolution for patch embedding
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, C, H, W]
        Returns:
            Patch embeddings [B, T*N, D] where N is patches per frame
        """
        B, T, C, H, W = x.shape
        
        # Reshape for 3D convolution
        x = x.transpose(1, 2)  # [B, C, T, H, W]
        x = self.proj(x)  # [B, D, T, H', W']
        
        # Flatten patches
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        x = self.norm(x)
        
        return x


class SpatioTemporalAttention(nn.Module):
    """Multi-head attention with spatiotemporal awareness"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_spatial_only: bool = False,
        use_temporal_only: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_spatial_only = use_spatial_only
        self.use_temporal_only = use_temporal_only
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        num_frames: Optional[int] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply spatiotemporal masking if specified
        if num_frames and (self.use_spatial_only or self.use_temporal_only):
            attn = self._apply_spatiotemporal_mask(attn, num_frames, N)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x
    
    def _apply_spatiotemporal_mask(
        self,
        attn: torch.Tensor,
        num_frames: int,
        num_patches: int,
    ) -> torch.Tensor:
        """Apply masking for spatial-only or temporal-only attention"""
        patches_per_frame = num_patches // num_frames
        
        if self.use_spatial_only:
            # Only attend within the same frame
            mask = torch.zeros_like(attn)
            for t in range(num_frames):
                start = t * patches_per_frame
                end = (t + 1) * patches_per_frame
                mask[:, :, start:end, start:end] = 1
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        elif self.use_temporal_only:
            # Only attend to same spatial position across time
            mask = torch.zeros_like(attn)
            for i in range(patches_per_frame):
                indices = [t * patches_per_frame + i for t in range(num_frames)]
                for idx1 in indices:
                    for idx2 in indices:
                        mask[:, :, idx1, idx2] = 1
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        return attn


class TransformerBlock(nn.Module):
    """Transformer block with spatiotemporal attention"""
    
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
        use_spatial_only: bool = False,
        use_temporal_only: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatioTemporalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_spatial_only=use_spatial_only,
            use_temporal_only=use_temporal_only,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        num_frames: Optional[int] = None,
    ) -> torch.Tensor:
        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x), num_frames))
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class SpatioTemporalViT(nn.Module):
    """
    Spatiotemporal Vision Transformer for Earth Observation
    
    Key features:
    - 3D patch embedding for temporal sequences
    - Spatiotemporal attention mechanisms
    - Support for multi-spectral bands
    - Flexible positional encoding
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        use_cls_token: bool = True,
        use_temporal_encoding: bool = True,
        use_spatial_encoding: bool = True,
        spatial_temporal_pattern: Optional[str] = None,  # "alternating" or "factorized"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.use_cls_token = use_cls_token
        self.use_temporal_encoding = use_temporal_encoding
        self.use_spatial_encoding = use_spatial_encoding
        self.spatial_temporal_pattern = spatial_temporal_pattern
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Class token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_patches += 1
        
        # Positional encodings
        if use_spatial_encoding:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches_per_frame, embed_dim)
            )
        
        if use_temporal_encoding:
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, num_frames, embed_dim)
            )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Determine attention pattern
            use_spatial_only = False
            use_temporal_only = False
            
            if spatial_temporal_pattern == "alternating":
                use_spatial_only = (i % 2 == 0)
                use_temporal_only = (i % 2 == 1)
            elif spatial_temporal_pattern == "factorized":
                use_spatial_only = (i < depth // 2)
                use_temporal_only = (i >= depth // 2)
            
            self.blocks.append(
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_spatial_only=use_spatial_only,
                    use_temporal_only=use_temporal_only,
                )
            )
        
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        if use_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
        if use_spatial_encoding:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
        if use_temporal_encoding:
            trunc_normal_(self.pos_embed_temporal, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def add_positional_encoding(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Add spatiotemporal positional encodings"""
        B, N, D = x.shape
        
        if self.use_cls_token:
            # Separate cls token and patch tokens
            cls_tokens = x[:, :1]
            x = x[:, 1:]
            N = N - 1
        
        patches_per_frame = N // num_frames
        
        # Add spatial positional encoding
        if self.use_spatial_encoding:
            spatial_pos = repeat(
                self.pos_embed_spatial[:, :patches_per_frame],
                '1 n d -> 1 (t n) d',
                t=num_frames
            )
            x = x + spatial_pos
        
        # Add temporal positional encoding
        if self.use_temporal_encoding:
            temporal_pos = repeat(
                self.pos_embed_temporal[:, :num_frames],
                '1 t d -> 1 (t n) d',
                n=patches_per_frame
            )
            x = x + temporal_pos
        
        if self.use_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
        
        return x
    
    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through transformer layers
        
        Args:
            x: Input tensor [B, T, C, H, W]
        
        Returns:
            Feature tensor [B, N, D]
        """
        B, T, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.add_positional_encoding(x, T)
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x, num_frames=T)
        
        x = self.norm(x)
        
        return x
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Full forward pass
        
        Args:
            x: Input tensor [B, T, C, H, W]
            return_features: Whether to return features instead of class predictions
        
        Returns:
            Class predictions or features
        """
        x = self.forward_features(x)
        
        if return_features:
            return x
        
        # Global average pooling or use cls token
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        
        x = self.head(x)
        
        return x