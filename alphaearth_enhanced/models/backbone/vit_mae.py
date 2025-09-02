"""
Vision Transformer with Masked Autoencoder (MAE) capabilities
Inspired by MAE and adapted for Earth observation data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from .spatiotemporal_vit import PatchEmbed3D, TransformerBlock


class VisionTransformerMAE(nn.Module):
    """
    Vision Transformer with Masked Autoencoder for self-supervised learning
    
    Key features:
    - Random masking of patches for pretraining
    - Efficient encoder-decoder architecture
    - Support for multi-spectral satellite imagery
    - Temporal consistency for video/time-series data
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # Encoder
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False,  # Fixed sin-cos embedding
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,  # Fixed sin-cos embedding
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Prediction head
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_chans * num_frames,
            bias=True,
        )
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights"""
        # Initialize patch embedding like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize positional embeddings with sin-cos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        
        # Initialize other parameters
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        
        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform per-sample random masking by per-sample shuffling.
        
        Args:
            x: Input tensor [N, L, D]
            mask_ratio: Percentage of patches to mask
        
        Returns:
            x_masked: Masked input
            mask: Binary mask
            ids_restore: Indices to restore original order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encoder forward pass with masking
        
        Args:
            x: Input tensor [B, T, C, H, W] or [B, C, H, W]
            mask_ratio: Percentage of patches to mask
        
        Returns:
            latent: Encoded features
            mask: Binary mask
            ids_restore: Indices to restore original order
        """
        # Handle both image and video inputs
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # Add time dimension
        
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional encoding
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decoder forward pass
        
        Args:
            x: Encoded features [N, L+1, D]
            ids_restore: Indices to restore original order
        
        Returns:
            pred: Predicted pixel values
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1,
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No cls token
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append cls token
        
        # Add positional encoding
        x = x + self.decoder_pos_embed
        
        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward_mae(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MAE pretraining
        
        Args:
            imgs: Input images [B, T, C, H, W] or [B, C, H, W]
            mask_ratio: Percentage of patches to mask
        
        Returns:
            loss: Reconstruction loss
            pred: Predicted pixel values
            mask: Binary mask
        """
        # Encode with masking
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        
        # Decode
        pred = self.forward_decoder(latent, ids_restore)
        
        # Calculate loss
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask
    
    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss
        
        Args:
            imgs: Original images
            pred: Predicted pixel values
            mask: Binary mask (0 is keep, 1 is remove)
        
        Returns:
            loss: Mean squared error loss
        """
        if len(imgs.shape) == 4:
            imgs = imgs.unsqueeze(1)
        
        target = patchify(imgs, self.patch_size)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels
        
        # Apply mask - only calculate loss on removed patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Standard forward pass for feature extraction
        
        Args:
            x: Input tensor [B, T, C, H, W] or [B, C, H, W]
            mask_ratio: If provided, perform MAE pretraining
        
        Returns:
            Features or MAE loss
        """
        if mask_ratio is not None:
            return self.forward_mae(x, mask_ratio)
        
        # Standard forward without masking
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token and positional encoding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> np.ndarray:
    """
    Generate 2D sin-cos positional embeddings
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Grid size (assuming square grid)
        cls_token: Whether to include cls token
    
    Returns:
        pos_embed: Positional embeddings [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: np.ndarray,
) -> np.ndarray:
    """Generate positional embeddings from grid"""
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for each position axis
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: np.ndarray,
) -> np.ndarray:
    """Generate 1D sin-cos positional embeddings"""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert images to patches
    
    Args:
        imgs: Input images [B, T, C, H, W] or [B, C, H, W]
        patch_size: Size of each patch
    
    Returns:
        patches: Flattened patches
    """
    if len(imgs.shape) == 5:
        B, T, C, H, W = imgs.shape
        assert H == W and H % patch_size == 0
        
        h = w = H // patch_size
        x = imgs.reshape(B, T, C, h, patch_size, w, patch_size)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        x = x.reshape(B, T * h * w, patch_size ** 2 * C)
    else:
        B, C, H, W = imgs.shape
        assert H == W and H % patch_size == 0
        
        h = w = H // patch_size
        x = imgs.reshape(B, C, h, patch_size, w, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, h * w, patch_size ** 2 * C)
    
    return x


def unpatchify(x: torch.Tensor, patch_size: int, num_frames: int = 1) -> torch.Tensor:
    """
    Convert patches back to images
    
    Args:
        x: Patches [B, L, patch_size**2 * C]
        patch_size: Size of each patch
        num_frames: Number of frames
    
    Returns:
        imgs: Reconstructed images
    """
    B, L, _ = x.shape
    h = w = int((L / num_frames) ** 0.5)
    C = x.shape[-1] // (patch_size ** 2)
    
    if num_frames > 1:
        x = x.reshape(B, num_frames, h, w, C, patch_size, patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        imgs = x.reshape(B, num_frames, C, h * patch_size, w * patch_size)
    else:
        x = x.reshape(B, h, w, C, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        imgs = x.reshape(B, C, h * patch_size, w * patch_size)
    
    return imgs