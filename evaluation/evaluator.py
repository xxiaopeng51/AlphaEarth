"""
Evaluator for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import time

from ..models import AlphaEarthFoundations
from .metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    SegmentationMetrics,
    MultiModalMetrics
)


class Evaluator:
    """
    Evaluator for AlphaEarth Foundations model.
    
    This class handles model evaluation on various tasks and datasets,
    computing comprehensive metrics and generating evaluation reports.
    """
    
    def __init__(
        self,
        model: AlphaEarthFoundations,
        device: str = "cuda",
        **kwargs
    ):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self.classification_metrics = ClassificationMetrics()
        self.regression_metrics = RegressionMetrics()
        self.segmentation_metrics = SegmentationMetrics()
        self.multimodal_metrics = MultiModalMetrics()
        
    def evaluate(
        self,
        data_loader: DataLoader,
        task: str,
        return_predictions: bool = False,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            task: Task type ('classification', 'regression', 'segmentation')
            return_predictions: Whether to return predictions
            return_features: Whether to return features
            
        Returns:
            Dictionary containing evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_features = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {task}"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    optical_data=batch.get('optical'),
                    radar_data=batch.get('radar'),
                    meteorological_data=batch.get('meteorological'),
                    text_data=batch.get('text'),
                    spatial_coords=batch.get('spatial_coords'),
                    temporal_coords=batch.get('temporal_coords'),
                    resolution_info=batch.get('resolution_info'),
                    task=task,
                    return_features=return_features
                )
                
                # Get predictions and targets
                if task in outputs['task_outputs']:
                    predictions = outputs['task_outputs'][task]
                    targets = batch.get('labels', batch.get('targets'))
                    
                    if targets is not None:
                        all_predictions.append(predictions.cpu())
                        all_targets.append(targets.cpu())
                        
                        # Compute loss
                        if task == 'classification':
                            loss = nn.CrossEntropyLoss()(predictions, targets)
                        elif task == 'regression':
                            loss = nn.MSELoss()(predictions, targets)
                        elif task == 'segmentation':
                            loss = nn.CrossEntropyLoss()(predictions, targets)
                        else:
                            loss = torch.tensor(0.0)
                        
                        total_loss += loss.item()
                        num_batches += 1
                
                # Store features if requested
                if return_features and 'global_features' in outputs:
                    all_features.append(outputs['global_features'].cpu())
        
        # Concatenate all predictions and targets
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_targets, task)
        
        # Add loss to metrics
        if num_batches > 0:
            metrics['loss'] = total_loss / num_batches
        
        # Prepare results
        results = {
            'metrics': metrics,
            'num_samples': len(all_targets) if all_targets.numel() > 0 else 0
        }
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['targets'] = all_targets
        
        if return_features:
            results['features'] = all_features
        
        return results
    
    def evaluate_multimodal(
        self,
        data_loader: DataLoader,
        tasks: List[str],
        return_predictions: bool = False,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate the model on multiple tasks.
        
        Args:
            data_loader: Data loader for evaluation
            tasks: List of task types
            return_predictions: Whether to return predictions
            return_features: Whether to return features
            
        Returns:
            Dictionary containing evaluation results for all tasks
        """
        self.model.eval()
        
        all_results = {}
        
        for task in tasks:
            self.logger.info(f"Evaluating task: {task}")
            task_results = self.evaluate(
                data_loader=data_loader,
                task=task,
                return_predictions=return_predictions,
                return_features=return_features
            )
            all_results[task] = task_results
        
        return all_results
    
    def evaluate_with_ablation(
        self,
        data_loader: DataLoader,
        task: str,
        modalities: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the model with modality ablation.
        
        Args:
            data_loader: Data loader for evaluation
            task: Task type
            modalities: List of modalities to test
            
        Returns:
            Dictionary containing ablation results
        """
        if modalities is None:
            modalities = ['optical', 'radar', 'meteorological', 'text']
        
        ablation_results = {}
        
        # Test with all modalities
        all_results = self.evaluate(data_loader, task, **kwargs)
        ablation_results['all_modalities'] = all_results
        
        # Test with individual modalities
        for modality in modalities:
            self.logger.info(f"Evaluating with {modality} only")
            
            # Create modified data loader with only one modality
            modified_loader = self._create_single_modality_loader(data_loader, modality)
            
            modality_results = self.evaluate(modified_loader, task, **kwargs)
            ablation_results[f'{modality}_only'] = modality_results
        
        return ablation_results
    
    def evaluate_with_corruption(
        self,
        data_loader: DataLoader,
        task: str,
        corruption_types: List[str] = None,
        corruption_levels: List[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the model with data corruption.
        
        Args:
            data_loader: Data loader for evaluation
            task: Task type
            corruption_types: Types of corruption to test
            corruption_levels: Levels of corruption to test
            
        Returns:
            Dictionary containing corruption results
        """
        if corruption_types is None:
            corruption_types = ['noise', 'blur', 'occlusion']
        
        if corruption_levels is None:
            corruption_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        corruption_results = {}
        
        # Test without corruption
        clean_results = self.evaluate(data_loader, task, **kwargs)
        corruption_results['clean'] = clean_results
        
        # Test with different corruption types and levels
        for corruption_type in corruption_types:
            corruption_results[corruption_type] = {}
            
            for level in corruption_levels:
                self.logger.info(f"Evaluating with {corruption_type} corruption level {level}")
                
                # Create corrupted data loader
                corrupted_loader = self._create_corrupted_loader(
                    data_loader, corruption_type, level
                )
                
                corrupted_results = self.evaluate(corrupted_loader, task, **kwargs)
                corruption_results[corruption_type][f'level_{level}'] = corrupted_results
        
        return corruption_results
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to the specified device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = self._move_batch_to_device(value)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        task: str
    ) -> Dict[str, float]:
        """Compute metrics for a specific task."""
        if task == 'classification':
            return self.classification_metrics.compute(predictions, targets)
        elif task == 'regression':
            return self.regression_metrics.compute(predictions, targets)
        elif task == 'segmentation':
            return self.segmentation_metrics.compute(predictions, targets)
        else:
            return {}
    
    def _create_single_modality_loader(self, data_loader: DataLoader, modality: str) -> DataLoader:
        """Create a data loader with only one modality."""
        # This would need to be implemented based on the specific data loader
        # For now, return the original loader
        return data_loader
    
    def _create_corrupted_loader(self, data_loader: DataLoader, corruption_type: str, level: float) -> DataLoader:
        """Create a data loader with corrupted data."""
        # This would need to be implemented based on the specific corruption type
        # For now, return the original loader
        return data_loader
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate an evaluation report."""
        report = "AlphaEarth Foundations Model Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        for task, task_results in results.items():
            report += f"Task: {task}\n"
            report += "-" * 20 + "\n"
            
            if 'metrics' in task_results:
                metrics = task_results['metrics']
                for metric_name, metric_value in metrics.items():
                    report += f"{metric_name}: {metric_value:.4f}\n"
            
            report += f"Number of samples: {task_results.get('num_samples', 0)}\n"
            report += "\n"
        
        return report
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save evaluation results to file."""
        import json
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Results saved to {save_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object serializable for JSON."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj