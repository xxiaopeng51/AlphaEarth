"""
AlphaEarth Foundations - Enhanced Global Multimodal Foundation Model

This module implements the main AlphaEarth Foundations model that combines
multi-modal encoders, fusion modules, and task-specific heads for global-scale
earth observation and analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

from .encoders import (
    OpticalEncoder,
    RadarEncoder,
    MeteorologicalEncoder,
    TextEncoder
)
from .fusion import (
    CrossAttentionFusion,
    SpatialTemporalPrecision,
    MultiModalFusion
)
from .heads import (
    ClassificationHead,
    RegressionHead,
    SegmentationHead
)


class AlphaEarthFoundations(nn.Module):
    """
    AlphaEarth Foundations - Enhanced Global Multimodal Foundation Model
    
    This model integrates multiple modalities (optical, radar, meteorological, text)
    with advanced fusion mechanisms and spatial-temporal precision modeling for
    global-scale earth observation tasks.
    """
    
    def __init__(
        self,
        # Encoder configurations
        optical_config: Dict = None,
        radar_config: Dict = None,
        meteorological_config: Dict = None,
        text_config: Dict = None,
        
        # Fusion configurations
        fusion_config: Dict = None,
        stp_config: Dict = None,
        
        # Model configurations
        hidden_dim: int = 1024,
        num_modalities: int = 4,
        dropout: float = 0.1,
        
        # Task configurations
        task_heads: Dict = None,
        
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.dropout = dropout
        
        # Default configurations
        optical_config = optical_config or {
            'model_name': 'vit_large_patch16_224',
            'pretrained': True,
            'output_dim': hidden_dim
        }
        
        radar_config = radar_config or {
            'model_name': 'vit_base_patch16_224',
            'pretrained': True,
            'output_dim': hidden_dim
        }
        
        meteorological_config = meteorological_config or {
            'input_dim': 128,
            'hidden_dim': hidden_dim,
            'output_dim': hidden_dim
        }
        
        text_config = text_config or {
            'model_name': 'bert-base-uncased',
            'pretrained': True,
            'output_dim': hidden_dim
        }
        
        fusion_config = fusion_config or {
            'hidden_dim': hidden_dim,
            'num_layers': 4,
            'num_heads': 16,
            'modalities': ['optical', 'radar', 'meteorological', 'text']
        }
        
        stp_config = stp_config or {
            'spatial_attention_layers': 3,
            'temporal_attention_layers': 2,
            'resolution_attention_layers': 2,
            'hidden_dim': hidden_dim,
            'num_heads': 16
        }
        
        # Initialize encoders
        self.optical_encoder = OpticalEncoder(**optical_config)
        self.radar_encoder = RadarEncoder(**radar_config)
        self.meteorological_encoder = MeteorologicalEncoder(**meteorological_config)
        self.text_encoder = TextEncoder(**text_config)
        
        # Initialize fusion modules
        self.cross_attention_fusion = CrossAttentionFusion(**fusion_config)
        self.stp_module = SpatialTemporalPrecision(**stp_config)
        
        # Multi-modal fusion
        self.multimodal_fusion = MultiModalFusion(
            hidden_dim=hidden_dim,
            num_modalities=num_modalities,
            dropout=dropout
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        if task_heads:
            for task_name, head_config in task_heads.items():
                if head_config['type'] == 'classification':
                    self.task_heads[task_name] = ClassificationHead(
                        input_dim=hidden_dim,
                        num_classes=head_config['num_classes']
                    )
                elif head_config['type'] == 'regression':
                    self.task_heads[task_name] = RegressionHead(
                        input_dim=hidden_dim,
                        output_dim=head_config['output_dim']
                    )
                elif head_config['type'] == 'segmentation':
                    self.task_heads[task_name] = SegmentationHead(
                        input_dim=hidden_dim,
                        num_classes=head_config['num_classes']
                    )
        
        # Global feature projection
        self.global_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Contrastive learning components
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(
        self,
        # Input data
        optical_data: Optional[torch.Tensor] = None,
        radar_data: Optional[torch.Tensor] = None,
        meteorological_data: Optional[torch.Tensor] = None,
        text_data: Optional[Dict] = None,
        
        # Spatial-temporal information
        spatial_coords: Optional[torch.Tensor] = None,
        temporal_coords: Optional[torch.Tensor] = None,
        resolution_info: Optional[torch.Tensor] = None,
        
        # Task information
        task: Optional[str] = None,
        return_features: bool = False,
        return_contrastive: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through AlphaEarth Foundations model.
        
        Args:
            optical_data: Optical imagery data (B, C, H, W)
            radar_data: Radar imagery data (B, C, H, W)
            meteorological_data: Meteorological data (B, T, V)
            text_data: Text data dictionary with input_ids, attention_mask, etc.
            spatial_coords: Spatial coordinates (B, L, 2)
            temporal_coords: Temporal coordinates (B, L, 1)
            resolution_info: Resolution information (B, L, 1)
            task: Task name for task-specific head
            return_features: Whether to return intermediate features
            return_contrastive: Whether to return contrastive features
            
        Returns:
            Dictionary containing model outputs and features
        """
        batch_size = self._get_batch_size(optical_data, radar_data, meteorological_data, text_data)
        
        # Encode each modality
        modality_features = {}
        modality_global_features = {}
        
        if optical_data is not None:
            optical_output = self.optical_encoder(optical_data, return_features=True)
            modality_features['optical'] = optical_output['features']
            modality_global_features['optical'] = optical_output['global_features']
        
        if radar_data is not None:
            radar_output = self.radar_encoder(radar_data, return_features=True)
            modality_features['radar'] = radar_output['features']
            modality_global_features['radar'] = radar_output['global_features']
        
        if meteorological_data is not None:
            met_output = self.meteorological_encoder(meteorological_data, return_features=True)
            modality_features['meteorological'] = met_output['features']
            modality_global_features['meteorological'] = met_output['global_features']
        
        if text_data is not None:
            text_output = self.text_encoder(
                input_ids=text_data['input_ids'],
                attention_mask=text_data.get('attention_mask'),
                return_features=True
            )
            modality_features['text'] = text_output['features']
            modality_global_features['text'] = text_output['global_features']
        
        # Cross-attention fusion
        fusion_output = self.cross_attention_fusion(
            modality_features,
            attention_masks=None  # Can be added if needed
        )
        
        # Spatial-Temporal Precision processing
        stp_output = self.stp_module(
            features=fusion_output['fused_features'],
            spatial_coords=spatial_coords,
            temporal_coords=temporal_coords,
            resolution_info=resolution_info
        )
        
        # Multi-modal fusion
        multimodal_output = self.multimodal_fusion(
            modality_global_features,
            stp_output['fused_features']
        )
        
        # Global feature projection
        global_features = self.global_projection(multimodal_output['global_features'])
        
        # Task-specific outputs
        task_outputs = {}
        if task and task in self.task_heads:
            task_outputs[task] = self.task_heads[task](global_features)
        
        # Prepare output
        output = {
            'global_features': global_features,
            'multimodal_features': multimodal_output['multimodal_features'],
            'stp_features': stp_output['fused_features'],
            'task_outputs': task_outputs
        }
        
        if return_features:
            output.update({
                'modality_features': modality_features,
                'modality_global_features': modality_global_features,
                'fusion_features': fusion_output['fused_features'],
                'spatial_features': stp_output['spatial_features'],
                'temporal_features': stp_output['temporal_features'],
                'resolution_features': stp_output['resolution_features']
            })
        
        if return_contrastive:
            contrastive_features = self.contrastive_projection(global_features)
            output['contrastive_features'] = contrastive_features
        
        return output
    
    def _get_batch_size(
        self,
        optical_data: Optional[torch.Tensor],
        radar_data: Optional[torch.Tensor],
        meteorological_data: Optional[torch.Tensor],
        text_data: Optional[Dict]
    ) -> int:
        """Get batch size from available data."""
        if optical_data is not None:
            return optical_data.shape[0]
        elif radar_data is not None:
            return radar_data.shape[0]
        elif meteorological_data is not None:
            return meteorological_data.shape[0]
        elif text_data is not None and 'input_ids' in text_data:
            return text_data['input_ids'].shape[0]
        else:
            raise ValueError("No valid input data provided")
    
    def compute_contrastive_loss(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of features.
        
        Args:
            features1: First set of features (B, D)
            features2: Second set of features (B, D)
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Contrastive loss
        """
        # Project features
        proj1 = self.contrastive_projection(features1)
        proj2 = self.contrastive_projection(features2)
        
        # Normalize features
        proj1 = F.normalize(proj1, p=2, dim=-1)
        proj2 = F.normalize(proj2, p=2, dim=-1)
        
        # Compute similarities
        similarities = torch.mm(proj1, proj2.T) / self.temperature
        
        if labels is not None:
            # Supervised contrastive learning
            batch_size = features1.shape[0]
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float()
            
            # Remove diagonal (self-similarity)
            mask = mask - torch.eye(batch_size, device=mask.device)
            
            # Compute loss
            exp_sim = torch.exp(similarities)
            log_prob = similarities - torch.log(exp_sim.sum(dim=1, keepdim=True))
            
            # Mask out negative pairs
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = -mean_log_prob_pos.mean()
        else:
            # Self-supervised contrastive learning
            batch_size = features1.shape[0]
            labels = torch.arange(batch_size, device=features1.device)
            
            loss = F.cross_entropy(similarities, labels)
        
        return loss
    
    def encode_multimodal(
        self,
        optical_data: Optional[torch.Tensor] = None,
        radar_data: Optional[torch.Tensor] = None,
        meteorological_data: Optional[torch.Tensor] = None,
        text_data: Optional[Dict] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        temporal_coords: Optional[torch.Tensor] = None,
        resolution_info: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode multi-modal data and return global features.
        
        Args:
            optical_data: Optical imagery data
            radar_data: Radar imagery data
            meteorological_data: Meteorological data
            text_data: Text data
            spatial_coords: Spatial coordinates
            temporal_coords: Temporal coordinates
            resolution_info: Resolution information
            
        Returns:
            Global multi-modal features (B, D)
        """
        with torch.no_grad():
            output = self.forward(
                optical_data=optical_data,
                radar_data=radar_data,
                meteorological_data=meteorological_data,
                text_data=text_data,
                spatial_coords=spatial_coords,
                temporal_coords=temporal_coords,
                resolution_info=resolution_info,
                return_features=False
            )
        
        return output['global_features']
    
    def predict_task(
        self,
        task: str,
        optical_data: Optional[torch.Tensor] = None,
        radar_data: Optional[torch.Tensor] = None,
        meteorological_data: Optional[torch.Tensor] = None,
        text_data: Optional[Dict] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        temporal_coords: Optional[torch.Tensor] = None,
        resolution_info: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Make predictions for a specific task.
        
        Args:
            task: Task name
            optical_data: Optical imagery data
            radar_data: Radar imagery data
            meteorological_data: Meteorological data
            text_data: Text data
            spatial_coords: Spatial coordinates
            temporal_coords: Temporal coordinates
            resolution_info: Resolution information
            
        Returns:
            Task predictions
        """
        if task not in self.task_heads:
            raise ValueError(f"Task '{task}' not found in task heads")
        
        output = self.forward(
            optical_data=optical_data,
            radar_data=radar_data,
            meteorological_data=meteorological_data,
            text_data=text_data,
            spatial_coords=spatial_coords,
            temporal_coords=temporal_coords,
            resolution_info=resolution_info,
            task=task
        )
        
        return output['task_outputs'][task]
    
    def get_attention_weights(
        self,
        optical_data: Optional[torch.Tensor] = None,
        radar_data: Optional[torch.Tensor] = None,
        meteorological_data: Optional[torch.Tensor] = None,
        text_data: Optional[Dict] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        temporal_coords: Optional[torch.Tensor] = None,
        resolution_info: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from the model for interpretability.
        
        Args:
            optical_data: Optical imagery data
            radar_data: Radar imagery data
            meteorological_data: Meteorological data
            text_data: Text data
            spatial_coords: Spatial coordinates
            temporal_coords: Temporal coordinates
            resolution_info: Resolution information
            
        Returns:
            Dictionary containing attention weights
        """
        output = self.forward(
            optical_data=optical_data,
            radar_data=radar_data,
            meteorological_data=meteorological_data,
            text_data=text_data,
            spatial_coords=spatial_coords,
            temporal_coords=temporal_coords,
            resolution_info=resolution_info,
            return_features=True
        )
        
        # Extract attention weights from fusion module
        attention_weights = {}
        
        # This would need to be implemented in the fusion modules
        # to return attention weights during forward pass
        
        return attention_weights


class AlphaEarthFoundationsWithScaling(nn.Module):
    """
    AlphaEarth Foundations with scaling law optimizations.
    
    This version includes optimizations for large-scale training and inference,
    including gradient checkpointing, mixed precision, and efficient attention.
    """
    
    def __init__(
        self,
        base_model: AlphaEarthFoundations,
        use_gradient_checkpointing: bool = True,
        use_flash_attention: bool = True,
        use_mixed_precision: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        self.use_mixed_precision = use_mixed_precision
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Enable flash attention if available
        if use_flash_attention:
            self._enable_flash_attention()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        # This would enable gradient checkpointing in the fusion modules
        # Implementation depends on the specific modules
        pass
    
    def _enable_flash_attention(self):
        """Enable flash attention for efficiency."""
        # This would replace standard attention with flash attention
        # Implementation depends on availability of flash attention
        pass
    
    def forward(self, *args, **kwargs):
        """Forward pass with scaling optimizations."""
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                return self.base_model(*args, **kwargs)
        else:
            return self.base_model(*args, **kwargs)