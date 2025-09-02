"""
Classification head for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ClassificationHead(nn.Module):
    """
    Classification head for multi-class and multi-label classification tasks.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 1000,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
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
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
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
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through classification head.
        
        Args:
            features: Input features (B, D)
            return_logits: Whether to return raw logits
            
        Returns:
            Dictionary containing:
                - 'logits': Raw logits (B, num_classes)
                - 'probabilities': Softmax probabilities (B, num_classes)
                - 'predictions': Predicted class indices (B,)
        """
        # Hidden layers
        hidden_features = self.hidden_layers(features)
        
        # Classification
        logits = self.classifier(hidden_features)
        
        # Probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Predictions
        predictions = torch.argmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }
        
        if return_logits:
            result['raw_logits'] = logits
        
        return result


class MultiLabelClassificationHead(nn.Module):
    """
    Multi-label classification head for tasks with multiple labels per sample.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 1000,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        threshold: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        
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
        
        # Classification layer
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
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
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-label classification head.
        
        Args:
            features: Input features (B, D)
            return_logits: Whether to return raw logits
            
        Returns:
            Dictionary containing:
                - 'logits': Raw logits (B, num_classes)
                - 'probabilities': Sigmoid probabilities (B, num_classes)
                - 'predictions': Binary predictions (B, num_classes)
        """
        # Hidden layers
        hidden_features = self.hidden_layers(features)
        
        # Classification
        logits = self.classifier(hidden_features)
        
        # Probabilities (sigmoid for multi-label)
        probabilities = torch.sigmoid(logits)
        
        # Predictions (binary based on threshold)
        predictions = (probabilities > self.threshold).float()
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }
        
        if return_logits:
            result['raw_logits'] = logits
        
        return result


class HierarchicalClassificationHead(nn.Module):
    """
    Hierarchical classification head for hierarchical label structures.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hierarchy_levels: List[int] = [10, 50, 200],
        hidden_dim: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hierarchy_levels = hierarchy_levels
        self.hidden_dim = hidden_dim
        
        # Shared hidden layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Level-specific classification heads
        self.level_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            for num_classes in hierarchy_levels
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
        return_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical classification head.
        
        Args:
            features: Input features (B, D)
            return_logits: Whether to return raw logits
            
        Returns:
            Dictionary containing predictions for each hierarchy level
        """
        # Shared features
        shared_features = self.shared_layers(features)
        
        # Level-specific predictions
        level_outputs = {}
        for level_idx, head in enumerate(self.level_heads):
            level_logits = head(shared_features)
            level_probabilities = F.softmax(level_logits, dim=-1)
            level_predictions = torch.argmax(level_logits, dim=-1)
            
            level_outputs[f'level_{level_idx}'] = {
                'logits': level_logits,
                'probabilities': level_probabilities,
                'predictions': level_predictions
            }
        
        return level_outputs


class ContrastiveClassificationHead(nn.Module):
    """
    Classification head with contrastive learning capabilities.
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        num_classes: int = 1000,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Feature projection for contrastive learning
        self.contrastive_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Class prototypes for contrastive learning
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, hidden_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize class prototypes
        nn.init.xavier_uniform_(self.class_prototypes)
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_contrastive: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through contrastive classification head.
        
        Args:
            features: Input features (B, D)
            labels: Optional labels for contrastive learning
            return_contrastive: Whether to return contrastive features
            
        Returns:
            Dictionary containing classification and contrastive outputs
        """
        # Classification
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': predictions
        }
        
        if return_contrastive:
            # Contrastive features
            contrastive_features = self.contrastive_projection(features)
            contrastive_features = F.normalize(contrastive_features, p=2, dim=-1)
            
            # Compute similarities with class prototypes
            class_prototypes_norm = F.normalize(self.class_prototypes, p=2, dim=-1)
            similarities = torch.mm(contrastive_features, class_prototypes_norm.T) / self.temperature
            
            result.update({
                'contrastive_features': contrastive_features,
                'prototype_similarities': similarities
            })
            
            # Compute contrastive loss if labels are provided
            if labels is not None:
                contrastive_loss = F.cross_entropy(similarities, labels)
                result['contrastive_loss'] = contrastive_loss
        
        return result