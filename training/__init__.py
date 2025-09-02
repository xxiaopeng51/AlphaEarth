"""
Training framework for AlphaEarth Foundations model.
"""

from .trainer import Trainer
from .losses import (
    MultiModalLoss,
    ContrastiveLoss,
    SpatialTemporalLoss
)
from .optimizers import (
    get_optimizer,
    get_scheduler
)

__all__ = [
    "Trainer",
    "MultiModalLoss",
    "ContrastiveLoss", 
    "SpatialTemporalLoss",
    "get_optimizer",
    "get_scheduler"
]