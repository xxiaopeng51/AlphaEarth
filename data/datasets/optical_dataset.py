"""
Optical satellite imagery dataset for AlphaEarth Foundations model.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
import rasterio
from PIL import Image
import torchvision.transforms as transforms


class OpticalDataset(Dataset):
    """
    Optical satellite imagery dataset.
    
    This dataset loads and processes optical satellite imagery from sources
    like Sentinel-2, Landsat, etc.
    """
    
    def __init__(
        self,
        data_root: str,
        samples: List[Dict],
        transforms: Optional[Dict] = None,
        bands: List[str] = None,
        image_size: int = 224,
        **kwargs
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.samples = samples
        self.transforms = transforms or {}
        self.bands = bands or ["B02", "B03", "B04", "B08", "B11", "B12"]
        self.image_size = image_size
        
        # Initialize transforms
        self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup data transforms."""
        self.transform_list = []
        
        # Geometric transforms
        if self.transforms.get('horizontal_flip', 0) > 0:
            self.transform_list.append(
                transforms.RandomHorizontalFlip(p=self.transforms['horizontal_flip'])
            )
        
        if self.transforms.get('vertical_flip', 0) > 0:
            self.transform_list.append(
                transforms.RandomVerticalFlip(p=self.transforms['vertical_flip'])
            )
        
        if self.transforms.get('rotation', 0) > 0:
            self.transform_list.append(
                transforms.RandomRotation(degrees=self.transforms['rotation'])
            )
        
        # Photometric transforms
        if self.transforms.get('brightness_contrast', 0) > 0:
            self.transform_list.append(
                transforms.ColorJitter(
                    brightness=self.transforms['brightness_contrast'],
                    contrast=self.transforms['brightness_contrast']
                )
            )
        
        # Resize
        self.transform_list.append(
            transforms.Resize((self.image_size, self.image_size))
        )
        
        # Convert to tensor
        self.transform_list.append(transforms.ToTensor())
        
        # Normalize
        if 'normalize' in self.transforms:
            normalize_params = self.transforms['normalize']
            self.transform_list.append(
                transforms.Normalize(
                    mean=normalize_params.get('mean', [0.5] * len(self.bands)),
                    std=normalize_params.get('std', [0.5] * len(self.bands))
                )
            )
        
        self.transform = transforms.Compose(self.transform_list)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing optical imagery data
        """
        sample = self.samples[idx]
        metadata = sample['metadata']
        
        # Load optical imagery
        optical_path = self.data_root / metadata['optical_path']
        optical_data = self._load_optical_data(optical_path)
        
        # Apply transforms
        if self.transform:
            optical_data = self.transform(optical_data)
        
        return {
            'optical': optical_data,
            'sample_id': sample['id'],
            'metadata': metadata
        }
    
    def _load_optical_data(self, path: Path) -> np.ndarray:
        """
        Load optical satellite imagery data.
        
        Args:
            path: Path to optical data file
            
        Returns:
            Optical imagery data as numpy array
        """
        if path.suffix.lower() in ['.tif', '.tiff']:
            # Load GeoTIFF
            with rasterio.open(path) as src:
                data = src.read()
                # Reorder from (C, H, W) to (H, W, C)
                data = np.transpose(data, (1, 2, 0))
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Load image
            image = Image.open(path)
            data = np.array(image)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Select bands if specified
        if self.bands and data.shape[-1] > len(self.bands):
            # Assume bands are in order
            data = data[:, :, :len(self.bands)]
        
        # Normalize to [0, 1]
        if data.dtype == np.uint8:
            data = data.astype(np.float32) / 255.0
        elif data.dtype == np.uint16:
            data = data.astype(np.float32) / 65535.0
        
        return data
    
    def get_info(self) -> Dict:
        """Get dataset information."""
        return {
            'num_samples': len(self.samples),
            'bands': self.bands,
            'image_size': self.image_size,
            'transforms': self.transforms
        }


class OpticalDatasetWithTemporal(OpticalDataset):
    """
    Optical dataset with temporal information.
    """
    
    def __init__(
        self,
        data_root: str,
        samples: List[Dict],
        temporal_window: int = 7,
        **kwargs
    ):
        self.temporal_window = temporal_window
        super().__init__(data_root, samples, **kwargs)
        
        # Group samples by location for temporal sequences
        self.temporal_groups = self._create_temporal_groups()
    
    def _create_temporal_groups(self) -> Dict[str, List[int]]:
        """Create temporal groups by location."""
        temporal_groups = {}
        
        for idx, sample in enumerate(self.samples):
            location = sample['metadata'].get('location', 'unknown')
            if location not in temporal_groups:
                temporal_groups[location] = []
            temporal_groups[location].append(idx)
        
        # Sort by timestamp if available
        for location in temporal_groups:
            indices = temporal_groups[location]
            if 'timestamp' in self.samples[indices[0]]['metadata']:
                indices.sort(key=lambda x: self.samples[x]['metadata']['timestamp'])
        
        return temporal_groups
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a temporal sequence from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing temporal optical data
        """
        sample = self.samples[idx]
        metadata = sample['metadata']
        location = metadata.get('location', 'unknown')
        
        # Get temporal sequence
        if location in self.temporal_groups:
            location_indices = self.temporal_groups[location]
            sample_idx = location_indices.index(idx)
            
            # Get temporal window
            start_idx = max(0, sample_idx - self.temporal_window // 2)
            end_idx = min(len(location_indices), start_idx + self.temporal_window)
            
            if end_idx - start_idx < self.temporal_window:
                start_idx = max(0, end_idx - self.temporal_window)
            
            temporal_indices = location_indices[start_idx:end_idx]
            
            # Load temporal data
            temporal_data = []
            for temp_idx in temporal_indices:
                temp_sample = self.samples[temp_idx]
                temp_path = self.data_root / temp_sample['metadata']['optical_path']
                temp_data = self._load_optical_data(temp_path)
                
                if self.transform:
                    temp_data = self.transform(temp_data)
                
                temporal_data.append(temp_data)
            
            # Pad if necessary
            while len(temporal_data) < self.temporal_window:
                temporal_data.append(temporal_data[-1])
            
            temporal_data = torch.stack(temporal_data, dim=0)
            
            return {
                'optical': temporal_data,
                'sample_id': sample['id'],
                'metadata': metadata,
                'temporal_length': len(temporal_indices)
            }
        else:
            # Fallback to single image
            return super().__getitem__(idx)


class OpticalDatasetWithSpatial(OpticalDataset):
    """
    Optical dataset with spatial information.
    """
    
    def __init__(
        self,
        data_root: str,
        samples: List[Dict],
        spatial_resolution: float = 0.1,
        **kwargs
    ):
        self.spatial_resolution = spatial_resolution
        super().__init__(data_root, samples, **kwargs)
        
        # Create spatial grid
        self.spatial_grid = self._create_spatial_grid()
    
    def _create_spatial_grid(self) -> Dict[Tuple[float, float], List[int]]:
        """Create spatial grid for spatial analysis."""
        spatial_grid = {}
        
        for idx, sample in enumerate(self.samples):
            metadata = sample['metadata']
            lat = metadata.get('latitude', 0.0)
            lon = metadata.get('longitude', 0.0)
            
            # Quantize coordinates to grid
            grid_lat = round(lat / self.spatial_resolution) * self.spatial_resolution
            grid_lon = round(lon / self.spatial_resolution) * self.spatial_resolution
            
            grid_key = (grid_lat, grid_lon)
            if grid_key not in spatial_grid:
                spatial_grid[grid_key] = []
            spatial_grid[grid_key].append(idx)
        
        return spatial_grid
    
    def get_spatial_neighbors(self, lat: float, lon: float, window: int = 3) -> List[int]:
        """Get spatial neighbors for a given location."""
        grid_lat = round(lat / self.spatial_resolution) * self.spatial_resolution
        grid_lon = round(lon / self.spatial_resolution) * self.spatial_resolution
        
        neighbors = []
        for dlat in range(-window, window + 1):
            for dlon in range(-window, window + 1):
                neighbor_key = (
                    grid_lat + dlat * self.spatial_resolution,
                    grid_lon + dlon * self.spatial_resolution
                )
                if neighbor_key in self.spatial_grid:
                    neighbors.extend(self.spatial_grid[neighbor_key])
        
        return neighbors
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with spatial context.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data and spatial context
        """
        # Get base sample
        sample_data = super().__getitem__(idx)
        
        # Add spatial context
        metadata = sample_data['metadata']
        lat = metadata.get('latitude', 0.0)
        lon = metadata.get('longitude', 0.0)
        
        spatial_neighbors = self.get_spatial_neighbors(lat, lon)
        sample_data['spatial_neighbors'] = spatial_neighbors
        sample_data['spatial_coords'] = torch.tensor([lat, lon], dtype=torch.float32)
        
        return sample_data