"""
Contrastive Learning Module inspired by SatCLIP
Implements CLIP-style contrastive learning for Earth observation data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class ContrastiveLearning(nn.Module):
    """
    Contrastive learning module for image-text alignment
    Inspired by SatCLIP with location-aware embeddings
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        temperature: float = 0.07,
        use_location_encoding: bool = True,
        use_temporal_encoding: bool = True,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_location_encoding = use_location_encoding
        self.use_temporal_encoding = use_temporal_encoding
        
        # Temperature parameter
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        # Projection heads
        self.image_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Location encoding
        if use_location_encoding:
            self.location_encoder = LocationEncoder(embed_dim)
        
        # Temporal encoding
        if use_temporal_encoding:
            self.temporal_encoder = TemporalEncoder(embed_dim)
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        locations: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        return_similarity: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for contrastive learning
        
        Args:
            image_features: Image features [B, D]
            text_features: Text features [B, D]
            locations: Geographic coordinates [B, 2]
            timestamps: Temporal information [B, 1]
            return_similarity: Return similarity matrix
        
        Returns:
            loss: Contrastive loss
            metrics: Dictionary of metrics
        """
        batch_size = image_features.shape[0]
        
        # Add location encoding if provided
        if self.use_location_encoding and locations is not None:
            location_features = self.location_encoder(locations)
            image_features = image_features + location_features
            text_features = text_features + location_features
        
        # Add temporal encoding if provided
        if self.use_temporal_encoding and timestamps is not None:
            temporal_features = self.temporal_encoder(timestamps)
            image_features = image_features + temporal_features
            text_features = text_features + temporal_features
        
        # Project features
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        if hasattr(self, 'temperature') and isinstance(self.temperature, nn.Parameter):
            temp = self.temperature.exp()
        else:
            temp = self.temperature
        
        logits = image_features @ text_features.T / temp
        
        # Compute contrastive loss
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        # Compute metrics
        with torch.no_grad():
            # Top-1 accuracy
            pred_i2t = logits.argmax(dim=-1)
            pred_t2i = logits.T.argmax(dim=-1)
            
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()
            
            # Top-5 accuracy
            top5_i2t = logits.topk(5, dim=-1)[1]
            top5_t2i = logits.T.topk(5, dim=-1)[1]
            
            acc5_i2t = (top5_i2t == labels.unsqueeze(1)).any(dim=1).float().mean()
            acc5_t2i = (top5_t2i == labels.unsqueeze(1)).any(dim=1).float().mean()
        
        metrics = {
            'loss': loss.item(),
            'acc_i2t': acc_i2t.item(),
            'acc_t2i': acc_t2i.item(),
            'acc5_i2t': acc5_i2t.item(),
            'acc5_t2i': acc5_t2i.item(),
            'temperature': temp.item() if isinstance(temp, torch.Tensor) else temp,
        }
        
        if return_similarity:
            return loss, metrics, logits
        
        return loss, metrics


class CLIPLoss(nn.Module):
    """
    CLIP loss with additional regularization terms
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negatives: bool = True,
        margin: float = 0.2,
    ):
        super().__init__()
        
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.margin = margin
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute CLIP loss with optional hard negatives
        
        Args:
            image_features: Normalized image features [B, D]
            text_features: Normalized text features [B, D]
            hard_negatives: Hard negative samples [B, K, D]
        
        Returns:
            loss: CLIP loss value
        """
        batch_size = image_features.shape[0]
        
        # Standard CLIP loss
        logits = image_features @ text_features.T / self.temperature
        labels = torch.arange(batch_size, device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        loss = (loss_i2t + loss_t2i) / 2
        
        # Add hard negative mining if enabled
        if self.use_hard_negatives and hard_negatives is not None:
            # Compute similarities with hard negatives
            hard_neg_sim = image_features @ hard_negatives.transpose(-2, -1)
            
            # Triplet loss with margin
            positive_sim = (image_features * text_features).sum(dim=-1, keepdim=True)
            triplet_loss = F.relu(hard_neg_sim - positive_sim + self.margin).mean()
            
            loss = loss + 0.1 * triplet_loss
        
        return loss


class LocationEncoder(nn.Module):
    """
    Location-aware encoding for geographic coordinates
    Following SatCLIP's approach
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Sinusoidal encoding for coordinates
        self.coord_encoder = CoordinateEncoding(embed_dim // 2)
        
        # Country/region embedding (optional)
        self.region_embed = nn.Embedding(200, embed_dim // 2)  # ~200 countries
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(
        self,
        coordinates: torch.Tensor,
        region_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode location information
        
        Args:
            coordinates: [B, 2] latitude, longitude
            region_ids: [B] optional region/country IDs
        
        Returns:
            Location features [B, embed_dim]
        """
        # Encode coordinates
        coord_features = self.coord_encoder(coordinates)
        
        # Add region embedding if provided
        if region_ids is not None:
            region_features = self.region_embed(region_ids)
        else:
            region_features = torch.zeros(
                coordinates.shape[0], self.region_embed.embedding_dim,
                device=coordinates.device
            )
        
        # Combine features
        combined = torch.cat([coord_features, region_features], dim=-1)
        
        return self.fusion(combined)


class TemporalEncoder(nn.Module):
    """
    Temporal encoding for time information
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Cyclic encoding for time
        self.hour_encoder = CyclicEncoding(24, embed_dim // 4)
        self.day_encoder = CyclicEncoding(31, embed_dim // 4)
        self.month_encoder = CyclicEncoding(12, embed_dim // 4)
        self.year_encoder = nn.Linear(1, embed_dim // 4)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal information
        
        Args:
            timestamps: [B, 4] hour, day, month, year
        
        Returns:
            Temporal features [B, embed_dim]
        """
        hour = timestamps[:, 0].long()
        day = timestamps[:, 1].long()
        month = timestamps[:, 2].long()
        year = timestamps[:, 3:4]
        
        # Encode each component
        hour_feat = self.hour_encoder(hour)
        day_feat = self.day_encoder(day)
        month_feat = self.month_encoder(month)
        year_feat = self.year_encoder((year - 2020) / 10)  # Normalize years
        
        # Combine
        combined = torch.cat([hour_feat, day_feat, month_feat, year_feat], dim=-1)
        
        return self.fusion(combined)


class CoordinateEncoding(nn.Module):
    """
    Sinusoidal encoding for geographic coordinates
    """
    
    def __init__(self, embed_dim: int, max_freq: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_freq = max_freq
        
        # Frequency bands
        freqs = torch.exp(
            torch.linspace(0, np.log(max_freq), embed_dim // 4)
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encode coordinates using sinusoidal functions
        
        Args:
            coords: [B, 2] latitude, longitude in degrees
        
        Returns:
            Encoded coordinates [B, embed_dim]
        """
        # Normalize to [-1, 1]
        lat = coords[:, 0:1] / 90.0
        lon = coords[:, 1:2] / 180.0
        
        # Apply sinusoidal encoding
        lat_enc = torch.cat([
            torch.sin(lat * self.freqs),
            torch.cos(lat * self.freqs)
        ], dim=-1)
        
        lon_enc = torch.cat([
            torch.sin(lon * self.freqs),
            torch.cos(lon * self.freqs)
        ], dim=-1)
        
        return torch.cat([lat_enc, lon_enc], dim=-1)


class CyclicEncoding(nn.Module):
    """
    Cyclic encoding for periodic features (hour, day, month)
    """
    
    def __init__(self, period: int, embed_dim: int):
        super().__init__()
        self.period = period
        self.embed_dim = embed_dim
        
        # Learnable embeddings
        self.embed = nn.Embedding(period, embed_dim)
        
        # Cyclic projection
        self.cyclic_proj = nn.Linear(2, embed_dim // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode cyclic features
        
        Args:
            x: [B] integer values in [0, period)
        
        Returns:
            Encoded features [B, embed_dim]
        """
        # Discrete embedding
        discrete = self.embed(x)
        
        # Continuous cyclic encoding
        angle = 2 * np.pi * x.float() / self.period
        cyclic = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        cyclic_feat = self.cyclic_proj(cyclic)
        
        # Combine discrete and continuous
        return torch.cat([discrete[:, :self.embed_dim//2], cyclic_feat], dim=-1)