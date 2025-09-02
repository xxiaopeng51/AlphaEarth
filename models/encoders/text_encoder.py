"""
Text encoder for AlphaEarth Foundations model.

This module implements the text encoder that processes textual descriptions,
metadata, and annotations related to satellite imagery and geospatial data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
import numpy as np


class TextEncoder(nn.Module):
    """
    Text encoder for processing textual descriptions and metadata.
    
    This encoder processes text descriptions, metadata, and annotations
    related to satellite imagery and geospatial data, extracting semantic
    features for multi-modal fusion.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 768,
        max_length: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        
        # Load pre-trained text model
        self.backbone = AutoModel.from_pretrained(
            model_name if pretrained else model_name,
            **kwargs
        )
        
        # Get backbone output dimension
        backbone_dim = self.backbone.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Dropout(dropout),
            nn.Linear(backbone_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Specialized embeddings for geospatial text
        self.geospatial_embeddings = nn.Embedding(1000, backbone_dim)  # For location names, etc.
        self.temporal_embeddings = nn.Embedding(1000, backbone_dim)    # For temporal references
        
        # Text preprocessing layers
        self.text_preprocessing = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.LayerNorm(backbone_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        geospatial_ids: Optional[torch.Tensor] = None,
        temporal_ids: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Input token IDs (B, L)
            attention_mask: Attention mask (B, L)
            token_type_ids: Token type IDs (B, L)
            geospatial_ids: Geospatial entity IDs (B, L)
            temporal_ids: Temporal entity IDs (B, L)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - 'features': Encoded features (B, L, D)
                - 'global_features': Global pooled features (B, D)
                - 'pooled_features': Pooled features (B, D)
        """
        # Get backbone features
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get sequence and pooled features
        sequence_features = outputs.last_hidden_state  # (B, L, D)
        pooled_features = outputs.pooler_output        # (B, D)
        
        # Add specialized embeddings
        if geospatial_ids is not None:
            geospatial_emb = self.geospatial_embeddings(geospatial_ids)
            sequence_features = sequence_features + geospatial_emb
        
        if temporal_ids is not None:
            temporal_emb = self.temporal_embeddings(temporal_ids)
            sequence_features = sequence_features + temporal_emb
        
        # Apply text preprocessing
        processed_features = self.text_preprocessing(sequence_features)
        
        # Output projection
        projected_features = self.output_projection(processed_features)
        
        # Global features (mean pooling with attention mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(projected_features)
            masked_features = projected_features.masked_fill(~mask_expanded.bool(), 0)
            global_features = masked_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            global_features = projected_features.mean(dim=1)
        
        result = {
            'features': projected_features,
            'global_features': global_features,
            'pooled_features': self.output_projection(pooled_features)
        }
        
        if return_features:
            result['sequence_features'] = sequence_features
            result['attention_weights'] = outputs.attentions if hasattr(outputs, 'attentions') else None
        
        return result
    
    def encode_geospatial_text(
        self,
        text: List[str],
        locations: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode geospatial text with location and temporal information.
        
        Args:
            text: List of text descriptions
            locations: List of location names
            timestamps: List of timestamps
            
        Returns:
            Dictionary containing encoded features
        """
        # Tokenize text
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        encoded = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create geospatial and temporal IDs
        geospatial_ids = None
        temporal_ids = None
        
        if locations is not None:
            # Simple location ID mapping (can be enhanced with proper geocoding)
            location_vocab = {loc: idx for idx, loc in enumerate(set(locations))}
            geospatial_ids = torch.tensor([location_vocab.get(loc, 0) for loc in locations])
        
        if timestamps is not None:
            # Simple temporal ID mapping (can be enhanced with proper temporal encoding)
            temporal_vocab = {ts: idx for idx, ts in enumerate(set(timestamps))}
            temporal_ids = torch.tensor([temporal_vocab.get(ts, 0) for ts in timestamps])
        
        # Encode
        return self.forward(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            geospatial_ids=geospatial_ids,
            temporal_ids=temporal_ids
        )
    
    def compute_text_similarity(
        self,
        text1: torch.Tensor,
        text2: torch.Tensor,
        similarity_type: str = "cosine"
    ) -> torch.Tensor:
        """
        Compute similarity between two text encodings.
        
        Args:
            text1: First text features (B, D)
            text2: Second text features (B, D)
            similarity_type: Type of similarity ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity scores (B,)
        """
        if similarity_type == "cosine":
            # Cosine similarity
            text1_norm = F.normalize(text1, p=2, dim=-1)
            text2_norm = F.normalize(text2, p=2, dim=-1)
            similarity = torch.sum(text1_norm * text2_norm, dim=-1)
        elif similarity_type == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = torch.norm(text1 - text2, p=2, dim=-1)
            similarity = 1.0 / (1.0 + distance)
        elif similarity_type == "dot":
            # Dot product similarity
            similarity = torch.sum(text1 * text2, dim=-1)
        else:
            raise ValueError(f"Unsupported similarity type: {similarity_type}")
        
        return similarity


class TextEncoderWithContrastive(nn.Module):
    """
    Text encoder with contrastive learning capabilities.
    """
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        temperature: float = 0.07,
        queue_size: int = 65536
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.temperature = temperature
        self.queue_size = queue_size
        
        # Momentum encoder for contrastive learning
        self.momentum_encoder = TextEncoder(
            model_name=text_encoder.model_name,
            pretrained=False,
            freeze_backbone=False,
            output_dim=text_encoder.output_dim
        )
        
        # Copy initial weights
        self.momentum_encoder.load_state_dict(text_encoder.state_dict())
        
        # Feature queue for contrastive learning
        self.register_buffer("queue", torch.randn(text_encoder.output_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Momentum update parameter
        self.momentum = 0.999
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing contrastive features
        """
        # Encode with main encoder
        main_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Encode with momentum encoder
        with torch.no_grad():
            momentum_features = self.momentum_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Update momentum encoder
        self._update_momentum_encoder()
        
        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(
            main_features['global_features'],
            momentum_features['global_features']
        )
        
        return {
            **main_features,
            'momentum_features': momentum_features['global_features'],
            'contrastive_loss': contrastive_loss
        }
    
    def _update_momentum_encoder(self):
        """Update momentum encoder with exponential moving average."""
        for param_q, param_k in zip(
            self.text_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)
    
    def _compute_contrastive_loss(
        self,
        features_q: torch.Tensor,
        features_k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss using momentum features.
        
        Args:
            features_q: Query features (B, D)
            features_k: Key features (B, D)
            
        Returns:
            Contrastive loss
        """
        batch_size = features_q.shape[0]
        
        # Normalize features
        features_q = F.normalize(features_q, p=2, dim=-1)
        features_k = F.normalize(features_k, p=2, dim=-1)
        
        # Compute positive similarities
        positive_sim = torch.sum(features_q * features_k, dim=-1) / self.temperature
        
        # Compute negative similarities with queue
        queue_features = F.normalize(self.queue.clone().detach(), p=2, dim=0)
        negative_sim = torch.mm(features_q, queue_features) / self.temperature
        
        # Combine positive and negative similarities
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        
        # Labels (positive pairs are at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long, device=features_q.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue
        self._update_queue(features_k)
        
        return loss
    
    def _update_queue(self, features_k: torch.Tensor):
        """Update the feature queue with new key features."""
        batch_size = features_k.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace queue features
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = features_k.T
            ptr = (ptr + batch_size) % self.queue_size
        else:
            # Handle wraparound
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = features_k[:remaining].T
            self.queue[:, :batch_size - remaining] = features_k[remaining:].T
            ptr = batch_size - remaining
        
        self.queue_ptr[0] = ptr