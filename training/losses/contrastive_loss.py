"""
Contrastive loss functions for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
        **kwargs
    ):
        super().__init__()
        
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    
    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through contrastive loss.
        
        Args:
            features: Input features (B, D)
            labels: Optional labels for supervised contrastive learning
            mask: Optional mask for contrastive learning
            
        Returns:
            Contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            # Supervised contrastive learning
            return self._supervised_contrastive_loss(features, labels, mask)
        elif labels is not None:
            # Supervised contrastive learning without mask
            return self._supervised_contrastive_loss(features, labels)
        else:
            # Self-supervised contrastive learning
            return self._self_supervised_contrastive_loss(features)
    
    def _supervised_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Supervised contrastive loss.
        
        Args:
            features: Input features (B, D)
            labels: Labels (B,)
            mask: Optional mask (B, B)
            
        Returns:
            Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        if mask is None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=device)
        
        # Compute log probabilities
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Compute mean log probability of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # Compute loss
        loss = -mean_log_prob_pos.mean()
        
        return loss
    
    def _self_supervised_contrastive_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Self-supervised contrastive loss.
        
        Args:
            features: Input features (B, D)
            
        Returns:
            Self-supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels (positive pairs are at diagonal)
        labels = torch.arange(batch_size, device=device)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for contrastive learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        **kwargs
    ):
        super().__init__()
        
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through InfoNCE loss.
        
        Args:
            query: Query features (B, D)
            positive: Positive features (B, D)
            negatives: Negative features (B, N, D)
            
        Returns:
            InfoNCE loss
        """
        # Normalize features
        query = F.normalize(query, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)
        
        # Compute positive similarity
        pos_sim = torch.sum(query * positive, dim=1, keepdim=True) / self.temperature
        
        # Compute negative similarities
        neg_sim = torch.bmm(negatives, query.unsqueeze(-1)).squeeze(-1) / self.temperature
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # Labels (positive is at index 0)
        labels = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for contrastive learning.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance_metric: str = "euclidean",
        **kwargs
    ):
        super().__init__()
        
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through triplet loss.
        
        Args:
            anchor: Anchor features (B, D)
            positive: Positive features (B, D)
            negative: Negative features (B, D)
            
        Returns:
            Triplet loss
        """
        if self.distance_metric == "euclidean":
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        # Compute triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class SimCLRLoss(nn.Module):
    """
    SimCLR loss for contrastive learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        **kwargs
    ):
        super().__init__()
        
        self.temperature = temperature
    
    def forward(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through SimCLR loss.
        
        Args:
            features1: First augmented features (B, D)
            features2: Second augmented features (B, D)
            
        Returns:
            SimCLR loss
        """
        batch_size = features1.shape[0]
        device = features1.device
        
        # Normalize features
        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        
        # Concatenate features
        features = torch.cat([features1, features2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels (positive pairs are at diagonal)
        labels = torch.arange(batch_size, device=device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, device=device)
        similarity_matrix = similarity_matrix - mask * 1e9
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class MoCoLoss(nn.Module):
    """
    MoCo loss for contrastive learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        queue_size: int = 65536,
        **kwargs
    ):
        super().__init__()
        
        self.temperature = temperature
        self.queue_size = queue_size
        
        # Initialize queue
        self.register_buffer("queue", torch.randn(queue_size, 128))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Normalize queue
        self.queue = F.normalize(self.queue, p=2, dim=1)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through MoCo loss.
        
        Args:
            query: Query features (B, D)
            key: Key features (B, D)
            
        Returns:
            MoCo loss
        """
        batch_size = query.shape[0]
        device = query.device
        
        # Normalize features
        query = F.normalize(query, p=2, dim=1)
        key = F.normalize(key, p=2, dim=1)
        
        # Compute positive similarities
        pos_sim = torch.sum(query * key, dim=1, keepdim=True) / self.temperature
        
        # Compute negative similarities with queue
        neg_sim = torch.mm(query, self.queue.T) / self.temperature
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # Labels (positive is at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._update_queue(key)
        
        return loss
    
    def _update_queue(self, key: torch.Tensor):
        """Update the queue with new key features."""
        batch_size = key.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace queue features
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = key
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Handle wraparound
            remaining = self.queue_size - ptr
            self.queue[ptr:] = key[:remaining]
            self.queue[:batch_size - remaining] = key[remaining:]
            ptr = batch_size - remaining
        
        self.queue_ptr[0] = ptr