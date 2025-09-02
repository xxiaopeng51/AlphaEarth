"""
Data processing modules for AlphaEarth Foundations model.
"""

from .datasets import (
    MultiModalDataset,
    OpticalDataset,
    RadarDataset,
    MeteorologicalDataset,
    TextDataset
)

from .transforms import (
    OpticalTransforms,
    RadarTransforms,
    MeteorologicalTransforms,
    TextTransforms
)

from .utils import (
    DataLoader,
    collate_fn,
    get_data_splits
)

__all__ = [
    "MultiModalDataset",
    "OpticalDataset", 
    "RadarDataset",
    "MeteorologicalDataset",
    "TextDataset",
    "OpticalTransforms",
    "RadarTransforms",
    "MeteorologicalTransforms",
    "TextTransforms",
    "DataLoader",
    "collate_fn",
    "get_data_splits"
]