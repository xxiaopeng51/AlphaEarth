"""
Evaluation metrics for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    r2_score, jaccard_score
)


class ClassificationMetrics:
    """
    Classification metrics for evaluation.
    """
    
    def __init__(self, num_classes: int = None):
        self.num_classes = num_classes
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Dictionary containing classification metrics
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Get predicted classes
        pred_classes = np.argmax(pred_np, axis=1)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(target_np, pred_classes),
            'precision': precision_score(target_np, pred_classes, average='weighted', zero_division=0),
            'recall': recall_score(target_np, pred_classes, average='weighted', zero_division=0),
            'f1_score': f1_score(target_np, pred_classes, average='weighted', zero_division=0)
        }
        
        # Compute per-class metrics if num_classes is specified
        if self.num_classes is not None and self.num_classes <= 10:
            precision_per_class = precision_score(target_np, pred_classes, average=None, zero_division=0)
            recall_per_class = recall_score(target_np, pred_classes, average=None, zero_division=0)
            f1_per_class = f1_score(target_np, pred_classes, average=None, zero_division=0)
            
            for i in range(self.num_classes):
                metrics[f'precision_class_{i}'] = precision_per_class[i] if i < len(precision_per_class) else 0.0
                metrics[f'recall_class_{i}'] = recall_per_class[i] if i < len(recall_per_class) else 0.0
                metrics[f'f1_class_{i}'] = f1_per_class[i] if i < len(f1_per_class) else 0.0
        
        # Compute AUC for binary classification
        if self.num_classes == 2:
            try:
                metrics['auc'] = roc_auc_score(target_np, pred_np[:, 1])
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics


class RegressionMetrics:
    """
    Regression metrics for evaluation.
    """
    
    def __init__(self):
        pass
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute regression metrics.
        
        Args:
            predictions: Model predictions (B, output_dim)
            targets: Ground truth values (B, output_dim)
            
        Returns:
            Dictionary containing regression metrics
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Compute metrics
        metrics = {
            'mse': mean_squared_error(target_np, pred_np),
            'rmse': np.sqrt(mean_squared_error(target_np, pred_np)),
            'mae': mean_absolute_error(target_np, pred_np),
            'r2': r2_score(target_np, pred_np)
        }
        
        # Compute relative metrics
        if np.mean(np.abs(target_np)) > 1e-8:
            metrics['mape'] = np.mean(np.abs((target_np - pred_np) / target_np)) * 100
            metrics['smape'] = np.mean(2 * np.abs(target_np - pred_np) / (np.abs(target_np) + np.abs(pred_np))) * 100
        
        return metrics


class SegmentationMetrics:
    """
    Segmentation metrics for evaluation.
    """
    
    def __init__(self, num_classes: int = None):
        self.num_classes = num_classes
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute segmentation metrics.
        
        Args:
            predictions: Model predictions (B, num_classes, H, W)
            targets: Ground truth labels (B, H, W)
            
        Returns:
            Dictionary containing segmentation metrics
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Get predicted classes
        pred_classes = np.argmax(pred_np, axis=1)
        
        # Flatten for metric computation
        pred_flat = pred_classes.flatten()
        target_flat = target_np.flatten()
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(target_flat, pred_flat),
            'precision': precision_score(target_flat, pred_flat, average='weighted', zero_division=0),
            'recall': recall_score(target_flat, pred_flat, average='weighted', zero_division=0),
            'f1_score': f1_score(target_flat, pred_flat, average='weighted', zero_division=0)
        }
        
        # Compute IoU (Jaccard score)
        if self.num_classes is not None:
            iou_scores = jaccard_score(target_flat, pred_flat, average=None, zero_division=0)
            metrics['mean_iou'] = np.mean(iou_scores)
            
            # Per-class IoU
            for i in range(self.num_classes):
                if i < len(iou_scores):
                    metrics[f'iou_class_{i}'] = iou_scores[i]
                else:
                    metrics[f'iou_class_{i}'] = 0.0
        
        return metrics


class MultiModalMetrics:
    """
    Multi-modal metrics for evaluation.
    """
    
    def __init__(self):
        self.classification_metrics = ClassificationMetrics()
        self.regression_metrics = RegressionMetrics()
        self.segmentation_metrics = SegmentationMetrics()
    
    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        task_types: Dict[str, str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute multi-modal metrics.
        
        Args:
            predictions: Dictionary of predictions for each modality
            targets: Dictionary of targets for each modality
            task_types: Dictionary of task types for each modality
            
        Returns:
            Dictionary containing metrics for each modality
        """
        all_metrics = {}
        
        for modality, pred in predictions.items():
            if modality in targets and modality in task_types:
                target = targets[modality]
                task_type = task_types[modality]
                
                if task_type == 'classification':
                    metrics = self.classification_metrics.compute(pred, target)
                elif task_type == 'regression':
                    metrics = self.regression_metrics.compute(pred, target)
                elif task_type == 'segmentation':
                    metrics = self.segmentation_metrics.compute(pred, target)
                else:
                    metrics = {}
                
                all_metrics[modality] = metrics
        
        return all_metrics


class ContrastiveMetrics:
    """
    Contrastive learning metrics for evaluation.
    """
    
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature
    
    def compute(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute contrastive learning metrics.
        
        Args:
            features1: First set of features (B, D)
            features2: Second set of features (B, D)
            labels: Optional labels for supervised contrastive learning
            
        Returns:
            Dictionary containing contrastive metrics
        """
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features1, features2.T) / self.temperature
        
        # Compute metrics
        metrics = {}
        
        if labels is not None:
            # Supervised contrastive learning metrics
            labels = labels.cpu().numpy()
            pred_labels = torch.argmax(similarity_matrix, dim=1).cpu().numpy()
            
            metrics['accuracy'] = accuracy_score(labels, pred_labels)
            metrics['precision'] = precision_score(labels, pred_labels, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(labels, pred_labels, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(labels, pred_labels, average='weighted', zero_division=0)
        else:
            # Self-supervised contrastive learning metrics
            batch_size = features1.shape[0]
            labels = torch.arange(batch_size, device=features1.device)
            
            metrics['accuracy'] = accuracy_score(labels.cpu().numpy(), torch.argmax(similarity_matrix, dim=1).cpu().numpy())
        
        # Compute alignment and uniformity metrics
        metrics['alignment'] = self._compute_alignment(features1, features2)
        metrics['uniformity'] = self._compute_uniformity(features1)
        
        return metrics
    
    def _compute_alignment(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute alignment metric."""
        # Compute pairwise distances between positive pairs
        distances = torch.norm(features1 - features2, p=2, dim=1)
        return torch.mean(distances).item()
    
    def _compute_uniformity(self, features: torch.Tensor) -> float:
        """Compute uniformity metric."""
        # Compute pairwise distances between all features
        pairwise_distances = torch.pdist(features, p=2)
        return torch.mean(pairwise_distances).item()


class SpatialTemporalMetrics:
    """
    Spatial-temporal metrics for evaluation.
    """
    
    def __init__(self):
        pass
    
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        spatial_coords: torch.Tensor,
        temporal_coords: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute spatial-temporal metrics.
        
        Args:
            predictions: Model predictions (B, T, H, W, D)
            targets: Ground truth values (B, T, H, W, D)
            spatial_coords: Spatial coordinates (B, T, H, W, 2)
            temporal_coords: Temporal coordinates (B, T, 1)
            
        Returns:
            Dictionary containing spatial-temporal metrics
        """
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()
        
        # Compute basic regression metrics
        metrics = {
            'mse': mean_squared_error(target_np, pred_np),
            'rmse': np.sqrt(mean_squared_error(target_np, pred_np)),
            'mae': mean_absolute_error(target_np, pred_np),
            'r2': r2_score(target_np, pred_np)
        }
        
        # Compute spatial consistency
        spatial_consistency = self._compute_spatial_consistency(predictions, spatial_coords)
        metrics['spatial_consistency'] = spatial_consistency
        
        # Compute temporal consistency
        temporal_consistency = self._compute_temporal_consistency(predictions, temporal_coords)
        metrics['temporal_consistency'] = temporal_consistency
        
        return metrics
    
    def _compute_spatial_consistency(self, predictions: torch.Tensor, spatial_coords: torch.Tensor) -> float:
        """Compute spatial consistency metric."""
        # Compute spatial gradients
        grad_x = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]
        grad_y = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        
        # Compute spatial coordinate differences
        coord_diff_x = spatial_coords[:, :, :, 1:] - spatial_coords[:, :, :, :-1]
        coord_diff_y = spatial_coords[:, :, 1:, :] - spatial_coords[:, :, :-1, :]
        
        # Normalize by spatial distance
        coord_dist_x = torch.norm(coord_diff_x, p=2, dim=-1, keepdim=True) + 1e-8
        coord_dist_y = torch.norm(coord_diff_y, p=2, dim=-1, keepdim=True) + 1e-8
        
        normalized_grad_x = grad_x / coord_dist_x
        normalized_grad_y = grad_y / coord_dist_y
        
        # Compute consistency (lower is better)
        consistency = torch.mean(torch.norm(normalized_grad_x, p=2, dim=-1)) + \
                     torch.mean(torch.norm(normalized_grad_y, p=2, dim=-1))
        
        return consistency.item()
    
    def _compute_temporal_consistency(self, predictions: torch.Tensor, temporal_coords: torch.Tensor) -> float:
        """Compute temporal consistency metric."""
        # Compute temporal gradients
        temporal_diff = predictions[:, 1:] - predictions[:, :-1]
        
        # Compute temporal coordinate differences
        coord_diff = temporal_coords[:, 1:] - temporal_coords[:, :-1]
        
        # Normalize by temporal distance
        coord_dist = torch.abs(coord_diff) + 1e-8
        normalized_diff = temporal_diff / coord_dist.unsqueeze(-1).unsqueeze(-1)
        
        # Compute consistency (lower is better)
        consistency = torch.mean(torch.norm(normalized_diff, p=2, dim=-1))
        
        return consistency.item()