"""
Multi-modal loss functions for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MultiModalLoss(nn.Module):
    """
    Multi-modal loss function that combines losses from different modalities.
    """
    
    def __init__(
        self,
        modalities: List[str] = ["optical", "radar", "meteorological", "text"],
        loss_weights: List[float] = None,
        main_loss_type: str = "cross_entropy",
        auxiliary_loss_type: str = "mse",
        contrastive_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.modalities = modalities
        self.loss_weights = loss_weights or [1.0] * len(modalities)
        self.main_loss_type = main_loss_type
        self.auxiliary_loss_type = auxiliary_loss_type
        self.contrastive_weight = contrastive_weight
        
        # Initialize loss functions
        self.main_loss = self._get_loss_function(main_loss_type)
        self.auxiliary_loss = self._get_loss_function(auxiliary_loss_type)
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        
    def _get_loss_function(self, loss_type: str) -> nn.Module:
        """Get loss function by type."""
        if loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        elif loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            
        Returns:
            Dictionary containing loss values
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Main task loss
        if 'task_outputs' in outputs and 'labels' in targets:
            task_outputs = outputs['task_outputs']
            labels = targets['labels']
            
            for task_name, task_output in task_outputs.items():
                if task_name in labels:
                    task_loss = self.main_loss(task_output, labels[task_name])
                    loss_dict[f'{task_name}_loss'] = task_loss
                    total_loss += task_loss
        
        # Modality-specific losses
        for i, modality in enumerate(self.modalities):
            if f'{modality}_features' in outputs and f'{modality}_targets' in targets:
                modality_loss = self.auxiliary_loss(
                    outputs[f'{modality}_features'],
                    targets[f'{modality}_targets']
                )
                weighted_loss = self.loss_weights[i] * modality_loss
                loss_dict[f'{modality}_loss'] = modality_loss
                loss_dict[f'{modality}_weighted_loss'] = weighted_loss
                total_loss += weighted_loss
        
        # Contrastive loss
        if 'contrastive_features' in outputs and self.contrastive_weight > 0:
            contrastive_loss = self.contrastive_loss(
                outputs['contrastive_features'],
                targets.get('contrastive_targets', None)
            )
            weighted_contrastive_loss = self.contrastive_weight * contrastive_loss
            loss_dict['contrastive_loss'] = contrastive_loss
            loss_dict['weighted_contrastive_loss'] = weighted_contrastive_loss
            total_loss += weighted_contrastive_loss
        
        # Regularization losses
        if 'regularization_loss' in outputs:
            reg_loss = outputs['regularization_loss']
            loss_dict['regularization_loss'] = reg_loss
            total_loss += reg_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class MaskedAutoencoderLoss(nn.Module):
    """
    Masked autoencoder loss for self-supervised learning.
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.75,
        loss_type: str = "mse",
        **kwargs
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through masked autoencoder loss.
        
        Args:
            reconstructed: Reconstructed features
            original: Original features
            mask: Mask indicating which features to reconstruct
            
        Returns:
            Masked reconstruction loss
        """
        # Apply mask
        masked_reconstructed = reconstructed * mask
        masked_original = original * mask
        
        # Compute loss only on masked regions
        loss = self.loss_fn(masked_reconstructed, masked_original)
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for time series data.
    """
    
    def __init__(
        self,
        consistency_weight: float = 0.1,
        temporal_window: int = 7,
        **kwargs
    ):
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.temporal_window = temporal_window
        
    def forward(
        self,
        features: torch.Tensor,
        temporal_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through temporal consistency loss.
        
        Args:
            features: Temporal features (B, T, D)
            temporal_coords: Temporal coordinates (B, T, 1)
            
        Returns:
            Temporal consistency loss
        """
        batch_size, temporal_len, feature_dim = features.shape
        
        if temporal_len < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Compute temporal differences
        temporal_diff = features[:, 1:] - features[:, :-1]
        
        # Compute temporal coordinate differences
        coord_diff = temporal_coords[:, 1:] - temporal_coords[:, :-1]
        
        # Normalize by temporal distance
        normalized_diff = temporal_diff / (coord_diff + 1e-8)
        
        # Compute consistency loss (smoothness)
        consistency_loss = torch.mean(torch.norm(normalized_diff, p=2, dim=-1))
        
        return self.consistency_weight * consistency_loss


class SpatialConsistencyLoss(nn.Module):
    """
    Spatial consistency loss for spatial data.
    """
    
    def __init__(
        self,
        consistency_weight: float = 0.1,
        spatial_window: int = 3,
        **kwargs
    ):
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.spatial_window = spatial_window
        
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through spatial consistency loss.
        
        Args:
            features: Spatial features (B, H, W, D)
            spatial_coords: Spatial coordinates (B, H, W, 2)
            
        Returns:
            Spatial consistency loss
        """
        batch_size, height, width, feature_dim = features.shape
        
        # Compute spatial gradients
        grad_x = features[:, :, 1:] - features[:, :, :-1]
        grad_y = features[:, 1:, :] - features[:, :-1, :]
        
        # Compute spatial coordinate differences
        coord_diff_x = spatial_coords[:, :, 1:] - spatial_coords[:, :, :-1]
        coord_diff_y = spatial_coords[:, 1:, :] - spatial_coords[:, :-1, :]
        
        # Normalize by spatial distance
        normalized_grad_x = grad_x / (torch.norm(coord_diff_x, p=2, dim=-1, keepdim=True) + 1e-8)
        normalized_grad_y = grad_y / (torch.norm(coord_diff_y, p=2, dim=-1, keepdim=True) + 1e-8)
        
        # Compute consistency loss (smoothness)
        consistency_loss = torch.mean(torch.norm(normalized_grad_x, p=2, dim=-1)) + \
                          torch.mean(torch.norm(normalized_grad_y, p=2, dim=-1))
        
        return self.consistency_weight * consistency_loss


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for multi-resolution features.
    """
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        scale_weights: List[float] = None,
        loss_type: str = "mse",
        **kwargs
    ):
        super().__init__()
        
        self.scales = scales
        self.scale_weights = scale_weights or [1.0] * len(scales)
        self.loss_type = loss_type
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through multi-scale loss.
        
        Args:
            predictions: Multi-scale predictions
            targets: Multi-scale targets
            
        Returns:
            Multi-scale loss
        """
        total_loss = 0.0
        
        for i, scale in enumerate(self.scales):
            scale_key = f'scale_{scale}'
            
            if scale_key in predictions and scale_key in targets:
                scale_loss = self.loss_fn(predictions[scale_key], targets[scale_key])
                weighted_loss = self.scale_weights[i] * scale_loss
                total_loss += weighted_loss
        
        return total_loss