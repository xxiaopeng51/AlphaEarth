"""
Data utility functions for AlphaEarth Foundations model.
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multi-modal data.
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched data dictionary
    """
    # Initialize batched data
    batched_data = {}
    
    # Get all keys from the first sample
    sample_keys = batch[0].keys()
    
    for key in sample_keys:
        if key == 'sample_id' or key == 'metadata':
            # Handle non-tensor data
            batched_data[key] = [sample[key] for sample in batch]
        else:
            # Handle tensor data
            tensor_list = []
            for sample in batch:
                if key in sample and sample[key] is not None:
                    tensor_list.append(sample[key])
            
            if tensor_list:
                # Stack tensors
                if isinstance(tensor_list[0], torch.Tensor):
                    batched_data[key] = torch.stack(tensor_list, dim=0)
                else:
                    batched_data[key] = tensor_list
    
    return batched_data


def get_data_splits(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Dataset to split
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Calculate sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    return train_dataset, val_dataset, test_dataset


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    collate_fn: Optional[callable] = None
) -> DataLoader:
    """
    Create a DataLoader with default settings.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        collate_fn: Custom collate function
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )


def normalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Normalize tensor with mean and standard deviation.
    
    Args:
        tensor: Input tensor
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Normalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Denormalize tensor with mean and standard deviation.
    
    Args:
        tensor: Normalized tensor
        mean: Mean values for each channel
        std: Standard deviation values for each channel
        
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    return tensor * std + mean


def compute_dataset_statistics(dataset, num_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute statistics for a dataset.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to use for statistics (None for all)
        
    Returns:
        Dictionary containing dataset statistics
    """
    if num_samples is None:
        num_samples = len(dataset)
    
    # Sample data
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Collect statistics
    statistics = {
        'total_samples': len(dataset),
        'analyzed_samples': len(sample_indices),
        'modalities': set(),
        'tensor_shapes': {},
        'value_ranges': {}
    }
    
    for idx in sample_indices:
        sample = dataset[idx]
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                statistics['modalities'].add(key)
                
                if key not in statistics['tensor_shapes']:
                    statistics['tensor_shapes'][key] = []
                statistics['tensor_shapes'][key].append(value.shape)
                
                if key not in statistics['value_ranges']:
                    statistics['value_ranges'][key] = {'min': float('inf'), 'max': float('-inf')}
                
                statistics['value_ranges'][key]['min'] = min(
                    statistics['value_ranges'][key]['min'],
                    value.min().item()
                )
                statistics['value_ranges'][key]['max'] = max(
                    statistics['value_ranges'][key]['max'],
                    value.max().item()
                )
    
    # Convert sets to lists for JSON serialization
    statistics['modalities'] = list(statistics['modalities'])
    
    return statistics


def create_data_augmentation(config: Dict) -> Dict[str, Any]:
    """
    Create data augmentation configuration.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Augmentation configuration dictionary
    """
    augmentation = {}
    
    # Optical data augmentation
    if 'optical' in config:
        optical_config = config['optical']
        augmentation['optical'] = {
            'horizontal_flip': optical_config.get('horizontal_flip', 0.5),
            'vertical_flip': optical_config.get('vertical_flip', 0.5),
            'rotation': optical_config.get('rotation', 0.3),
            'brightness_contrast': optical_config.get('brightness_contrast', 0.2),
            'noise': optical_config.get('noise', 0.1)
        }
    
    # Radar data augmentation
    if 'radar' in config:
        radar_config = config['radar']
        augmentation['radar'] = {
            'horizontal_flip': radar_config.get('horizontal_flip', 0.5),
            'vertical_flip': radar_config.get('vertical_flip', 0.5),
            'rotation': radar_config.get('rotation', 0.3),
            'noise': radar_config.get('noise', 0.1)
        }
    
    # Meteorological data augmentation
    if 'meteorological' in config:
        met_config = config['meteorological']
        augmentation['meteorological'] = {
            'noise': met_config.get('noise', 0.05),
            'temporal_shift': met_config.get('temporal_shift', 0.1)
        }
    
    return augmentation


def validate_data_consistency(dataset, num_samples: int = 100) -> Dict[str, Any]:
    """
    Validate data consistency in a dataset.
    
    Args:
        dataset: Dataset to validate
        num_samples: Number of samples to check
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'total_samples': len(dataset),
        'checked_samples': min(num_samples, len(dataset)),
        'errors': [],
        'warnings': [],
        'modality_consistency': {}
    }
    
    # Sample data for validation
    sample_indices = np.random.choice(len(dataset), validation_results['checked_samples'], replace=False)
    
    for idx in sample_indices:
        try:
            sample = dataset[idx]
            
            # Check for required keys
            required_keys = ['sample_id', 'metadata']
            for key in required_keys:
                if key not in sample:
                    validation_results['errors'].append(f"Sample {idx}: Missing required key '{key}'")
            
            # Check tensor consistency
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    if key not in validation_results['modality_consistency']:
                        validation_results['modality_consistency'][key] = {
                            'shapes': set(),
                            'dtypes': set(),
                            'value_ranges': {'min': float('inf'), 'max': float('-inf')}
                        }
                    
                    # Check shape consistency
                    validation_results['modality_consistency'][key]['shapes'].add(value.shape)
                    
                    # Check dtype consistency
                    validation_results['modality_consistency'][key]['dtypes'].add(value.dtype)
                    
                    # Check value ranges
                    validation_results['modality_consistency'][key]['value_ranges']['min'] = min(
                        validation_results['modality_consistency'][key]['value_ranges']['min'],
                        value.min().item()
                    )
                    validation_results['modality_consistency'][key]['value_ranges']['max'] = max(
                        validation_results['modality_consistency'][key]['value_ranges']['max'],
                        value.max().item()
                    )
        
        except Exception as e:
            validation_results['errors'].append(f"Sample {idx}: Error loading sample - {str(e)}")
    
    # Convert sets to lists for JSON serialization
    for modality in validation_results['modality_consistency']:
        validation_results['modality_consistency'][modality]['shapes'] = list(
            validation_results['modality_consistency'][modality]['shapes']
        )
        validation_results['modality_consistency'][modality]['dtypes'] = list(
            validation_results['modality_consistency'][modality]['dtypes']
        )
    
    return validation_results


def create_data_loader_config(
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False
) -> Dict[str, Any]:
    """
    Create data loader configuration.
    
    Args:
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Data loader configuration dictionary
    """
    return {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'shuffle': shuffle,
        'drop_last': drop_last
    }