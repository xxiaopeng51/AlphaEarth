"""
Spatial-temporal loss functions for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SpatialTemporalLoss(nn.Module):
    """
    Spatial-temporal loss for capturing spatiotemporal patterns.
    """
    
    def __init__(
        self,
        spatial_weight: float = 1.0,
        temporal_weight: float = 1.0,
        consistency_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.consistency_weight = consistency_weight
        
        # Spatial loss
        self.spatial_loss = SpatialConsistencyLoss()
        
        # Temporal loss
        self.temporal_loss = TemporalConsistencyLoss()
        
        # Consistency loss
        self.consistency_loss = SpatiotemporalConsistencyLoss()
    
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: torch.Tensor,
        temporal_coords: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through spatial-temporal loss.
        
        Args:
            features: Input features (B, T, H, W, D)
            spatial_coords: Spatial coordinates (B, T, H, W, 2)
            temporal_coords: Temporal coordinates (B, T, 1)
            targets: Optional target values
            
        Returns:
            Dictionary containing loss values
        """
        loss_dict = {}
        
        # Spatial consistency loss
        spatial_loss = self.spatial_loss(features, spatial_coords)
        loss_dict['spatial_loss'] = spatial_loss
        
        # Temporal consistency loss
        temporal_loss = self.temporal_loss(features, temporal_coords)
        loss_dict['temporal_loss'] = temporal_loss
        
        # Spatiotemporal consistency loss
        consistency_loss = self.consistency_loss(features, spatial_coords, temporal_coords)
        loss_dict['consistency_loss'] = consistency_loss
        
        # Total loss
        total_loss = (self.spatial_weight * spatial_loss +
                     self.temporal_weight * temporal_loss +
                     self.consistency_weight * consistency_loss)
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


class SpatialConsistencyLoss(nn.Module):
    """
    Spatial consistency loss for spatial smoothness.
    """
    
    def __init__(
        self,
        smoothness_weight: float = 1.0,
        edge_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.edge_weight = edge_weight
    
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through spatial consistency loss.
        
        Args:
            features: Spatial features (B, T, H, W, D)
            spatial_coords: Spatial coordinates (B, T, H, W, 2)
            
        Returns:
            Spatial consistency loss
        """
        batch_size, temporal_len, height, width, feature_dim = features.shape
        
        # Compute spatial gradients
        grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]
        grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]
        
        # Compute spatial coordinate differences
        coord_diff_x = spatial_coords[:, :, :, 1:] - spatial_coords[:, :, :, :-1]
        coord_diff_y = spatial_coords[:, :, 1:, :] - spatial_coords[:, :, :-1, :]
        
        # Normalize by spatial distance
        coord_dist_x = torch.norm(coord_diff_x, p=2, dim=-1, keepdim=True) + 1e-8
        coord_dist_y = torch.norm(coord_diff_y, p=2, dim=-1, keepdim=True) + 1e-8
        
        normalized_grad_x = grad_x / coord_dist_x
        normalized_grad_y = grad_y / coord_dist_y
        
        # Compute smoothness loss
        smoothness_loss = torch.mean(torch.norm(normalized_grad_x, p=2, dim=-1)) + \
                         torch.mean(torch.norm(normalized_grad_y, p=2, dim=-1))
        
        # Compute edge-preserving loss
        edge_loss = self._compute_edge_loss(features, spatial_coords)
        
        total_loss = self.smoothness_weight * smoothness_loss + self.edge_weight * edge_loss
        
        return total_loss
    
    def _compute_edge_loss(
        self,
        features: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge-preserving loss.
        
        Args:
            features: Spatial features (B, T, H, W, D)
            spatial_coords: Spatial coordinates (B, T, H, W, 2)
            
        Returns:
            Edge loss
        """
        # Compute spatial gradients
        grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]
        grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]
        
        # Compute coordinate gradients
        coord_grad_x = spatial_coords[:, :, :, 1:] - spatial_coords[:, :, :, :-1]
        coord_grad_y = spatial_coords[:, :, 1:, :] - spatial_coords[:, :, :-1, :]
        
        # Compute edge strength
        edge_strength_x = torch.norm(coord_grad_x, p=2, dim=-1, keepdim=True)
        edge_strength_y = torch.norm(coord_grad_y, p=2, dim=-1, keepdim=True)
        
        # Compute feature gradients
        feature_grad_x = torch.norm(grad_x, p=2, dim=-1, keepdim=True)
        feature_grad_y = torch.norm(grad_y, p=2, dim=-1, keepdim=True)
        
        # Edge-preserving loss (features should be smooth where coordinates are smooth)
        edge_loss = torch.mean(torch.abs(feature_grad_x - edge_strength_x)) + \
                   torch.mean(torch.abs(feature_grad_y - edge_strength_y))
        
        return edge_loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for temporal smoothness.
    """
    
    def __init__(
        self,
        smoothness_weight: float = 1.0,
        periodicity_weight: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.periodicity_weight = periodicity_weight
    
    def forward(
        self,
        features: torch.Tensor,
        temporal_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through temporal consistency loss.
        
        Args:
            features: Temporal features (B, T, H, W, D)
            temporal_coords: Temporal coordinates (B, T, 1)
            
        Returns:
            Temporal consistency loss
        """
        # Compute temporal gradients
        temporal_diff = features[:, 1:] - features[:, :-1]
        
        # Compute temporal coordinate differences
        coord_diff = temporal_coords[:, 1:] - temporal_coords[:, :-1]
        
        # Normalize by temporal distance
        coord_dist = torch.abs(coord_diff) + 1e-8
        normalized_diff = temporal_diff / coord_dist.unsqueeze(-1).unsqueeze(-1)
        
        # Compute smoothness loss
        smoothness_loss = torch.mean(torch.norm(normalized_diff, p=2, dim=-1))
        
        # Compute periodicity loss
        periodicity_loss = self._compute_periodicity_loss(features, temporal_coords)
        
        total_loss = self.smoothness_weight * smoothness_loss + self.periodicity_weight * periodicity_loss
        
        return total_loss
    
    def _compute_periodicity_loss(
        self,
        features: torch.Tensor,
        temporal_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute periodicity loss for seasonal patterns.
        
        Args:
            features: Temporal features (B, T, H, W, D)
            temporal_coords: Temporal coordinates (B, T, 1)
            
        Returns:
            Periodicity loss
        """
        batch_size, temporal_len, height, width, feature_dim = features.shape
        
        # Assume yearly periodicity (365 days)
        yearly_period = 365
        
        # Find pairs of features that are one year apart
        periodicity_loss = 0.0
        count = 0
        
        for i in range(temporal_len):
            for j in range(i + 1, temporal_len):
                time_diff = torch.abs(temporal_coords[:, j] - temporal_coords[:, i])
                
                # Check if time difference is close to yearly period
                if torch.any(torch.abs(time_diff - yearly_period) < 7):  # Within 7 days
                    feature_diff = features[:, j] - features[:, i]
                    periodicity_loss += torch.mean(torch.norm(feature_diff, p=2, dim=-1))
                    count += 1
        
        if count > 0:
            periodicity_loss /= count
        
        return periodicity_loss


class SpatiotemporalConsistencyLoss(nn.Module):
    """
    Spatiotemporal consistency loss for joint spatial-temporal patterns.
    """
    
    def __init__(
        self,
        consistency_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        features: torch.Tensor,
        spatial_coords: torch.Tensor,
        temporal_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through spatiotemporal consistency loss.
        
        Args:
            features: Spatiotemporal features (B, T, H, W, D)
            spatial_coords: Spatial coordinates (B, T, H, W, 2)
            temporal_coords: Temporal coordinates (B, T, 1)
            
        Returns:
            Spatiotemporal consistency loss
        """
        batch_size, temporal_len, height, width, feature_dim = features.shape
        
        # Compute spatiotemporal gradients
        spatial_grad_x = features[:, :, :, 1:] - features[:, :, :, :-1]
        spatial_grad_y = features[:, :, 1:, :] - features[:, :, :-1, :]
        temporal_grad = features[:, 1:] - features[:, :-1]
        
        # Compute coordinate gradients
        coord_grad_x = spatial_coords[:, :, :, 1:] - spatial_coords[:, :, :, :-1]
        coord_grad_y = spatial_coords[:, :, 1:, :] - spatial_coords[:, :, :-1, :]
        coord_grad_t = temporal_coords[:, 1:] - temporal_coords[:, :-1]
        
        # Normalize by coordinate distances
        coord_dist_x = torch.norm(coord_grad_x, p=2, dim=-1, keepdim=True) + 1e-8
        coord_dist_y = torch.norm(coord_grad_y, p=2, dim=-1, keepdim=True) + 1e-8
        coord_dist_t = torch.abs(coord_grad_t) + 1e-8
        
        normalized_grad_x = spatial_grad_x / coord_dist_x
        normalized_grad_y = spatial_grad_y / coord_dist_y
        normalized_grad_t = temporal_grad / coord_dist_t.unsqueeze(-1).unsqueeze(-1)
        
        # Compute consistency loss
        consistency_loss = (torch.mean(torch.norm(normalized_grad_x, p=2, dim=-1)) +
                           torch.mean(torch.norm(normalized_grad_y, p=2, dim=-1)) +
                           torch.mean(torch.norm(normalized_grad_t, p=2, dim=-1)))
        
        return self.consistency_weight * consistency_loss


class MultiScaleSpatialTemporalLoss(nn.Module):
    """
    Multi-scale spatial-temporal loss for multi-resolution features.
    """
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        scale_weights: List[float] = None,
        **kwargs
    ):
        super().__init__()
        
        self.scales = scales
        self.scale_weights = scale_weights or [1.0] * len(scales)
        
        # Create loss functions for each scale
        self.scale_losses = nn.ModuleList([
            SpatialTemporalLoss(**kwargs)
            for _ in scales
        ])
    
    def forward(
        self,
        multi_scale_features: Dict[str, torch.Tensor],
        spatial_coords: torch.Tensor,
        temporal_coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-scale spatial-temporal loss.
        
        Args:
            multi_scale_features: Multi-scale features dictionary
            spatial_coords: Spatial coordinates
            temporal_coords: Temporal coordinates
            
        Returns:
            Dictionary containing loss values for each scale
        """
        loss_dict = {}
        total_loss = 0.0
        
        for i, scale in enumerate(self.scales):
            scale_key = f'scale_{scale}'
            
            if scale_key in multi_scale_features:
                scale_features = multi_scale_features[scale_key]
                scale_loss = self.scale_losses[i](scale_features, spatial_coords, temporal_coords)
                
                # Weight the loss
                weighted_loss = self.scale_weights[i] * scale_loss['total_loss']
                
                loss_dict[f'{scale_key}_loss'] = scale_loss
                loss_dict[f'{scale_key}_weighted_loss'] = weighted_loss
                
                total_loss += weighted_loss
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict