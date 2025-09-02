"""
Multi-modal dataset for AlphaEarth Foundations model.

This module implements a unified dataset that combines multiple modalities
(optical, radar, meteorological, text) for training and evaluation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import os
import json
from pathlib import Path


class MultiModalDataset(Dataset):
    """
    Multi-modal dataset that combines different data modalities.
    
    This dataset loads and combines optical imagery, radar data, meteorological
    data, and text annotations for multi-modal training and evaluation.
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        modalities: List[str] = ["optical", "radar", "meteorological", "text"],
        transforms: Optional[Dict] = None,
        max_samples: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.modalities = modalities
        self.transforms = transforms or {}
        self.max_samples = max_samples
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_file)
        
        # Filter samples based on available modalities
        self.samples = self._filter_samples()
        
        # Initialize modality-specific datasets
        self.modality_datasets = self._initialize_modality_datasets()
        
    def _load_metadata(self, metadata_file: str) -> pd.DataFrame:
        """Load dataset metadata from file."""
        if metadata_file.endswith('.csv'):
            metadata = pd.read_csv(metadata_file)
        elif metadata_file.endswith('.json'):
            with open(metadata_file, 'r') as f:
                metadata = pd.DataFrame(json.load(f))
        else:
            raise ValueError(f"Unsupported metadata file format: {metadata_file}")
        
        return metadata
    
    def _filter_samples(self) -> List[Dict]:
        """Filter samples based on available modalities."""
        samples = []
        
        for idx, row in self.metadata.iterrows():
            sample = {
                'id': row.get('id', idx),
                'metadata': row.to_dict()
            }
            
            # Check if all required modalities are available
            available_modalities = []
            for modality in self.modalities:
                if self._check_modality_availability(row, modality):
                    available_modalities.append(modality)
            
            if len(available_modalities) == len(self.modalities):
                sample['available_modalities'] = available_modalities
                samples.append(sample)
            
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        return samples
    
    def _check_modality_availability(self, row: pd.Series, modality: str) -> bool:
        """Check if a modality is available for a given sample."""
        if modality == "optical":
            return 'optical_path' in row and pd.notna(row['optical_path'])
        elif modality == "radar":
            return 'radar_path' in row and pd.notna(row['radar_path'])
        elif modality == "meteorological":
            return 'meteorological_path' in row and pd.notna(row['meteorological_path'])
        elif modality == "text":
            return 'text_path' in row and pd.notna(row['text_path'])
        else:
            return False
    
    def _initialize_modality_datasets(self) -> Dict:
        """Initialize modality-specific datasets."""
        modality_datasets = {}
        
        for modality in self.modalities:
            if modality == "optical":
                modality_datasets[modality] = OpticalDataset(
                    data_root=self.data_root,
                    samples=self.samples,
                    transforms=self.transforms.get(modality)
                )
            elif modality == "radar":
                modality_datasets[modality] = RadarDataset(
                    data_root=self.data_root,
                    samples=self.samples,
                    transforms=self.transforms.get(modality)
                )
            elif modality == "meteorological":
                modality_datasets[modality] = MeteorologicalDataset(
                    data_root=self.data_root,
                    samples=self.samples,
                    transforms=self.transforms.get(modality)
                )
            elif modality == "text":
                modality_datasets[modality] = TextDataset(
                    data_root=self.data_root,
                    samples=self.samples,
                    transforms=self.transforms.get(modality)
                )
        
        return modality_datasets
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing data from all modalities
        """
        sample = self.samples[idx]
        
        # Load data from each modality
        data = {
            'sample_id': sample['id'],
            'metadata': sample['metadata']
        }
        
        for modality in self.modalities:
            if modality in self.modality_datasets:
                modality_data = self.modality_datasets[modality][idx]
                data[modality] = modality_data
        
        return data
    
    def get_sample_metadata(self, idx: int) -> Dict:
        """Get metadata for a specific sample."""
        return self.samples[idx]['metadata']
    
    def get_modality_info(self, modality: str) -> Dict:
        """Get information about a specific modality."""
        if modality in self.modality_datasets:
            return self.modality_datasets[modality].get_info()
        else:
            return {}


class MultiModalDatasetWithTemporal(MultiModalDataset):
    """
    Multi-modal dataset with temporal information for time series analysis.
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        temporal_window: int = 7,
        temporal_stride: int = 1,
        **kwargs
    ):
        self.temporal_window = temporal_window
        self.temporal_stride = temporal_stride
        
        super().__init__(data_root, metadata_file, **kwargs)
        
        # Group samples by temporal sequences
        self.temporal_sequences = self._create_temporal_sequences()
    
    def _create_temporal_sequences(self) -> List[List[int]]:
        """Create temporal sequences from samples."""
        sequences = []
        
        # Group samples by location and create temporal sequences
        location_groups = {}
        for idx, sample in enumerate(self.samples):
            location = sample['metadata'].get('location', 'unknown')
            if location not in location_groups:
                location_groups[location] = []
            location_groups[location].append(idx)
        
        # Create sequences for each location
        for location, indices in location_groups.items():
            # Sort by timestamp if available
            if 'timestamp' in self.samples[indices[0]]['metadata']:
                indices.sort(key=lambda x: self.samples[x]['metadata']['timestamp'])
            
            # Create sliding window sequences
            for i in range(0, len(indices) - self.temporal_window + 1, self.temporal_stride):
                sequence = indices[i:i + self.temporal_window]
                if len(sequence) == self.temporal_window:
                    sequences.append(sequence)
        
        return sequences
    
    def __len__(self) -> int:
        """Return the number of temporal sequences."""
        return len(self.temporal_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a temporal sequence from the dataset.
        
        Args:
            idx: Sequence index
            
        Returns:
            Dictionary containing temporal data from all modalities
        """
        sequence_indices = self.temporal_sequences[idx]
        
        # Load temporal data from each modality
        temporal_data = {
            'sequence_id': idx,
            'sequence_length': len(sequence_indices)
        }
        
        for modality in self.modalities:
            if modality in self.modality_datasets:
                modality_temporal_data = []
                for seq_idx in sequence_indices:
                    modality_data = self.modality_datasets[modality][seq_idx]
                    modality_temporal_data.append(modality_data)
                
                # Stack temporal data
                if modality_temporal_data:
                    temporal_data[modality] = torch.stack(modality_temporal_data, dim=0)
        
        return temporal_data


class MultiModalDatasetWithSpatial(MultiModalDataset):
    """
    Multi-modal dataset with spatial information for spatial analysis.
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str,
        spatial_resolution: float = 0.1,  # degrees
        spatial_window: int = 5,  # 5x5 grid
        **kwargs
    ):
        self.spatial_resolution = spatial_resolution
        self.spatial_window = spatial_window
        
        super().__init__(data_root, metadata_file, **kwargs)
        
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
    
    def get_spatial_neighbors(self, lat: float, lon: float) -> List[int]:
        """Get spatial neighbors for a given location."""
        grid_lat = round(lat / self.spatial_resolution) * self.spatial_resolution
        grid_lon = round(lon / self.spatial_resolution) * self.spatial_resolution
        
        neighbors = []
        for dlat in range(-self.spatial_window, self.spatial_window + 1):
            for dlon in range(-self.spatial_window, self.spatial_window + 1):
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