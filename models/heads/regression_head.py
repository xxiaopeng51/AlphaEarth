"""
Regression head for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class RegressionHead(nn.Module):
    """
    Regression head for continuous value prediction tasks.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression layer
        self.regressor = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through regression head.
        
        Args:
            features: Input features (B, D)
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary containing:
                - 'predictions': Regression predictions (B, output_dim)
                - 'uncertainty': Uncertainty estimates (B, output_dim) if return_uncertainty=True
        """
        # Hidden layers
        hidden_features = self.hidden_layers(features)
        
        # Regression
        predictions = self.regressor(hidden_features)
        
        result = {
            'predictions': predictions
        }
        
        if return_uncertainty:
            # Simple uncertainty estimation using dropout
            uncertainty = self._estimate_uncertainty(features)
            result['uncertainty'] = uncertainty
        
        return result
    
    def _estimate_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate prediction uncertainty using Monte Carlo dropout.
        
        Args:
            features: Input features (B, D)
            
        Returns:
            Uncertainty estimates (B, output_dim)
        """
        # Enable dropout for uncertainty estimation
        self.train()
        
        # Multiple forward passes
        predictions = []
        for _ in range(10):  # Number of Monte Carlo samples
            pred = self.forward(features)['predictions']
            predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, B, output_dim)
        
        # Compute uncertainty as standard deviation
        uncertainty = torch.std(predictions, dim=0)
        
        # Disable dropout
        self.eval()
        
        return uncertainty


class MultiOutputRegressionHead(nn.Module):
    """
    Multi-output regression head for predicting multiple continuous values.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dims: List[int] = [1, 1, 1],
        hidden_dim: int = 512,
        dropout: float = 0.1,
        shared_layers: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.hidden_dim = hidden_dim
        self.shared_layers = shared_layers
        
        if shared_layers:
            # Shared hidden layers
            self.shared_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
            # Task-specific heads
            self.task_heads = nn.ModuleList([
                nn.Linear(hidden_dim // 2, output_dim)
                for output_dim in output_dims
            ])
        else:
            # Separate networks for each output
            self.task_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
                for output_dim in output_dims
            ])
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-output regression head.
        
        Args:
            features: Input features (B, D)
            return_individual: Whether to return individual task predictions
            
        Returns:
            Dictionary containing predictions for each task
        """
        if self.shared_layers:
            # Shared feature extraction
            shared_features = self.shared_layers(features)
            
            # Task-specific predictions
            predictions = {}
            for i, head in enumerate(self.task_heads):
                predictions[f'task_{i}'] = head(shared_features)
        else:
            # Separate predictions
            predictions = {}
            for i, network in enumerate(self.task_networks):
                predictions[f'task_{i}'] = network(features)
        
        # Combine all predictions
        all_predictions = torch.cat(list(predictions.values()), dim=-1)
        
        result = {
            'predictions': all_predictions,
            'individual_predictions': predictions
        }
        
        return result


class QuantileRegressionHead(nn.Module):
    """
    Quantile regression head for uncertainty estimation.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 1,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        hidden_dim: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantiles = quantiles
        self.hidden_dim = hidden_dim
        
        # Hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Quantile-specific heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, output_dim)
            for _ in quantiles
        ])
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through quantile regression head.
        
        Args:
            features: Input features (B, D)
            
        Returns:
            Dictionary containing quantile predictions
        """
        # Hidden layers
        hidden_features = self.hidden_layers(features)
        
        # Quantile predictions
        quantile_predictions = {}
        for i, (quantile, head) in enumerate(zip(self.quantiles, self.quantile_heads)):
            quantile_predictions[f'q{quantile}'] = head(hidden_features)
        
        # Median prediction (main prediction)
        median_prediction = quantile_predictions['q0.5']
        
        # Uncertainty estimation
        lower_bound = quantile_predictions['q0.1']
        upper_bound = quantile_predictions['q0.9']
        uncertainty = (upper_bound - lower_bound) / 2
        
        return {
            'predictions': median_prediction,
            'quantile_predictions': quantile_predictions,
            'uncertainty': uncertainty,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }