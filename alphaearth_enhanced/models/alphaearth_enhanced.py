"""
AlphaEarth Enhanced: Global-Scale Multimodal Earth Observation Foundation Model

This model combines insights from:
- Google's AlphaEarth Foundations
- Clay Foundation Model
- SatCLIP (Microsoft)
- Prithvi (NASA/IBM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from einops import rearrange, repeat

from .backbone import SpatioTemporalViT
from .encoders import (
    OpticalEncoder,
    SAREncoder,
    ThermalEncoder,
    TextEncoder,
    MetadataEncoder
)
from .fusion import MultimodalFusion
from .heads import TaskHead


class AlphaEarthEnhanced(nn.Module):
    """
    Main model class for AlphaEarth Enhanced.
    
    Features:
    - Multimodal input support (optical, SAR, thermal, text, metadata)
    - Spatiotemporal modeling with 3D positional encoding
    - Contrastive learning for image-text alignment
    - Scalable architecture following scaling laws
    - Support for various downstream tasks
    """
    
    def __init__(
        self,
        # Vision backbone parameters
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 4,
        num_bands_optical: int = 13,  # Sentinel-2 bands
        num_bands_sar: int = 2,  # Sentinel-1 VV, VH
        num_bands_thermal: int = 2,  # Landsat thermal bands
        
        # Model architecture parameters
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        
        # Multimodal fusion parameters
        fusion_depth: int = 4,
        fusion_heads: int = 8,
        
        # Text encoder parameters
        text_encoder: str = "clip",  # or "bert", "roberta"
        max_text_length: int = 77,
        
        # Training parameters
        use_mae: bool = True,
        mask_ratio: float = 0.75,
        use_contrastive: bool = True,
        temperature: float = 0.07,
        
        # Downstream task parameters
        num_classes: Optional[int] = None,
        task_heads: Optional[List[str]] = None,
        
        # Scaling parameters
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.use_mae = use_mae
        self.mask_ratio = mask_ratio
        self.use_contrastive = use_contrastive
        self.temperature = temperature
        
        # Initialize encoders for different modalities
        self.optical_encoder = OpticalEncoder(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=num_bands_optical,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        self.sar_encoder = SAREncoder(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            in_chans=num_bands_sar,
            embed_dim=embed_dim,
            depth=depth // 2,  # Smaller depth for SAR
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        
        self.thermal_encoder = ThermalEncoder(
            img_size=img_size,
            patch_size=patch_size * 2,  # Larger patches for lower resolution thermal
            num_frames=num_frames,
            in_chans=num_bands_thermal,
            embed_dim=embed_dim,
            depth=depth // 2,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        
        self.text_encoder = TextEncoder(
            encoder_type=text_encoder,
            max_length=max_text_length,
            embed_dim=embed_dim,
        )
        
        self.metadata_encoder = MetadataEncoder(
            embed_dim=embed_dim,
        )
        
        # Multimodal fusion module
        self.fusion = MultimodalFusion(
            embed_dim=embed_dim,
            depth=fusion_depth,
            num_heads=fusion_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        
        # Projection heads for contrastive learning
        if use_contrastive:
            self.image_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
            self.text_projection = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        if task_heads:
            for task in task_heads:
                if task == "classification" and num_classes:
                    self.task_heads[task] = nn.Linear(embed_dim, num_classes)
                elif task == "segmentation":
                    self.task_heads[task] = SegmentationHead(
                        embed_dim, num_classes or 10, img_size, patch_size
                    )
                elif task == "change_detection":
                    self.task_heads[task] = ChangeDetectionHead(embed_dim)
                # Add more task heads as needed
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_mae(
        self,
        optical: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for masked autoencoder pretraining"""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Encode with masking
        latent, mask, ids_restore = self.optical_encoder.forward_mae(
            optical, mask_ratio
        )
        
        # Decode
        pred = self.optical_encoder.decode(latent, ids_restore)
        
        # Calculate reconstruction loss
        loss = self.mae_loss(optical, pred, mask)
        
        return loss, pred, mask
    
    def forward_contrastive(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Forward pass for contrastive learning"""
        # Encode images (can be optical, SAR, or thermal)
        image_features = self.optical_encoder(images)
        image_features = self.image_projection(image_features.mean(dim=1))
        
        # Encode texts
        text_features = self.text_encoder(texts)
        text_features = self.text_projection(text_features)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Calculate contrastive loss
        logits = image_features @ text_features.T / self.temperature
        labels = torch.arange(len(images), device=images.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    def forward(
        self,
        optical: Optional[torch.Tensor] = None,
        sar: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
        task: Optional[str] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            optical: Optical imagery tensor [B, T, C, H, W]
            sar: SAR imagery tensor [B, T, C, H, W]
            thermal: Thermal imagery tensor [B, T, C, H, W]
            text: Text descriptions [B, L] or list of strings
            metadata: Dictionary containing metadata (coordinates, time, etc.)
            task: Specific task to perform
            return_features: Whether to return intermediate features
        
        Returns:
            Model outputs based on the task
        """
        features = {}
        
        # Encode each modality if provided
        if optical is not None:
            features['optical'] = self.optical_encoder(optical)
        
        if sar is not None:
            features['sar'] = self.sar_encoder(sar)
        
        if thermal is not None:
            features['thermal'] = self.thermal_encoder(thermal)
        
        if text is not None:
            features['text'] = self.text_encoder(text)
        
        if metadata is not None:
            features['metadata'] = self.metadata_encoder(metadata)
        
        # Fuse multimodal features
        if len(features) > 1:
            fused_features = self.fusion(features)
        else:
            # If only one modality, use it directly
            fused_features = list(features.values())[0]
        
        # Apply task-specific head if specified
        if task and task in self.task_heads:
            output = self.task_heads[task](fused_features)
        else:
            output = fused_features
        
        if return_features:
            return {
                'output': output,
                'features': features,
                'fused_features': fused_features,
            }
        
        return output
    
    def mae_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate masked autoencoder loss.
        
        Args:
            imgs: Original images
            pred: Predicted pixel values
            mask: Binary mask indicating which patches were masked
        
        Returns:
            MAE loss value
        """
        target = imgs
        if len(target.shape) == 5:  # [B, T, C, H, W]
            target = rearrange(target, 'b t c h w -> b (t h w) c')
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over channels
        
        # Apply mask - only calculate loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = False):
        """Load pretrained weights"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle potential mismatches in state dict keys
        model_state = self.state_dict()
        filtered_state = {
            k: v for k, v in state_dict.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        
        self.load_state_dict(filtered_state, strict=strict)
        
        print(f"Loaded {len(filtered_state)}/{len(model_state)} parameters")
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class SegmentationHead(nn.Module):
    """Segmentation head for dense prediction tasks"""
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        img_size: int,
        patch_size: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim // 4, num_classes, kernel_size=2, stride=2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation"""
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        
        # Reshape to image format
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Decode to full resolution
        x = self.decoder(x)
        
        # Interpolate to original image size if needed
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear')
        
        return x


class ChangeDetectionHead(nn.Module):
    """Head for change detection between two time points"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2),  # Binary change/no-change
        )
    
    def forward(self, features_t1: torch.Tensor, features_t2: torch.Tensor) -> torch.Tensor:
        """Forward pass for change detection"""
        # Concatenate features from two time points
        combined = torch.cat([features_t1, features_t2], dim=-1)
        return self.projection(combined)