"""
Text Encoder for Natural Language Descriptions
Supports CLIP, BERT, and RoBERTa encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    AutoModel,
    AutoTokenizer,
)


class TextEncoder(nn.Module):
    """
    Text encoder for processing natural language descriptions
    
    Features:
    - Multiple pretrained model support (CLIP, BERT, RoBERTa)
    - Location-aware text encoding
    - Temporal context encoding
    - Multi-language support
    """
    
    def __init__(
        self,
        encoder_type: str = "clip",  # "clip", "bert", "roberta", "custom"
        model_name: Optional[str] = None,
        max_length: int = 77,
        embed_dim: int = 768,
        freeze_encoder: bool = False,
        use_location_encoding: bool = True,
        use_temporal_encoding: bool = True,
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.use_location_encoding = use_location_encoding
        self.use_temporal_encoding = use_temporal_encoding
        
        # Initialize text encoder and tokenizer
        if encoder_type == "clip":
            model_name = model_name or "openai/clip-vit-base-patch32"
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.encoder = CLIPTextModel.from_pretrained(model_name)
            hidden_dim = self.encoder.config.hidden_size
            
        elif encoder_type == "bert":
            model_name = model_name or "bert-base-uncased"
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.encoder = BertModel.from_pretrained(model_name)
            hidden_dim = self.encoder.config.hidden_size
            
        elif encoder_type == "roberta":
            model_name = model_name or "roberta-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.encoder = RobertaModel.from_pretrained(model_name)
            hidden_dim = self.encoder.config.hidden_size
            
        else:
            # Custom encoder
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_dim = self.encoder.config.hidden_size
        
        # Projection layer if dimensions don't match
        if hidden_dim != embed_dim:
            self.projection = nn.Linear(hidden_dim, embed_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Location encoding module
        if use_location_encoding:
            self.location_encoder = LocationTextEncoder(embed_dim)
        
        # Temporal encoding module
        if use_temporal_encoding:
            self.temporal_encoder = TemporalTextEncoder(embed_dim)
    
    def tokenize(
        self,
        texts: Union[List[str], torch.Tensor],
    ) -> dict:
        """
        Tokenize input texts
        
        Args:
            texts: List of text strings or tensor of token IDs
        
        Returns:
            Tokenized inputs
        """
        if isinstance(texts, torch.Tensor):
            # Already tokenized
            return {"input_ids": texts}
        
        # Tokenize text strings
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return encoded
    
    def forward(
        self,
        texts: Union[List[str], torch.Tensor],
        locations: Optional[torch.Tensor] = None,  # [B, 2] lat, lon
        timestamps: Optional[torch.Tensor] = None,  # [B, 1] unix timestamp
        return_pooled: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of text encoder
        
        Args:
            texts: Input texts (list of strings or token tensor)
            locations: Optional geographic coordinates
            timestamps: Optional temporal information
            return_pooled: Return pooled representation or sequence
        
        Returns:
            Encoded text features
        """
        # Tokenize if needed
        if isinstance(texts, list):
            tokens = self.tokenize(texts)
            # Move to same device as model
            tokens = {k: v.to(next(self.encoder.parameters()).device) 
                     for k, v in tokens.items()}
        else:
            tokens = {"input_ids": texts}
        
        # Encode text
        if self.encoder_type == "clip":
            outputs = self.encoder(**tokens)
            features = outputs.last_hidden_state
        else:
            outputs = self.encoder(**tokens)
            features = outputs.last_hidden_state
        
        # Pool if requested
        if return_pooled:
            if self.encoder_type == "clip":
                # Use EOS token representation
                features = features[torch.arange(features.shape[0]), 
                                   tokens["input_ids"].argmax(dim=-1)]
            else:
                # Use CLS token or mean pooling
                features = features[:, 0, :]  # CLS token
        
        # Project to target dimension
        features = self.projection(features)
        
        # Add location encoding if provided
        if self.use_location_encoding and locations is not None:
            location_features = self.location_encoder(locations)
            if len(features.shape) == 3:
                location_features = location_features.unsqueeze(1)
            features = features + location_features
        
        # Add temporal encoding if provided
        if self.use_temporal_encoding and timestamps is not None:
            temporal_features = self.temporal_encoder(timestamps)
            if len(features.shape) == 3:
                temporal_features = temporal_features.unsqueeze(1)
            features = features + temporal_features
        
        return features


class LocationTextEncoder(nn.Module):
    """Encode geographic location information"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Sinusoidal encoding for lat/lon
        self.lat_encoder = nn.Sequential(
            SinusoidalEncoding(embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )
        
        self.lon_encoder = nn.Sequential(
            SinusoidalEncoding(embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, locations: torch.Tensor) -> torch.Tensor:
        """
        Encode location information
        
        Args:
            locations: [B, 2] tensor with latitude and longitude
        
        Returns:
            Location features [B, embed_dim]
        """
        lat = locations[:, 0:1]
        lon = locations[:, 1:2]
        
        lat_features = self.lat_encoder(lat)
        lon_features = self.lon_encoder(lon)
        
        combined = torch.cat([lat_features, lon_features], dim=-1)
        return self.fusion(combined)


class TemporalTextEncoder(nn.Module):
    """Encode temporal information"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Encode different time scales
        self.hour_encoder = nn.Embedding(24, embed_dim // 4)
        self.day_encoder = nn.Embedding(31, embed_dim // 4)
        self.month_encoder = nn.Embedding(12, embed_dim // 4)
        self.season_encoder = nn.Embedding(4, embed_dim // 4)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal information
        
        Args:
            timestamps: [B, 1] tensor with unix timestamps or [B, 4] with hour, day, month, season
        
        Returns:
            Temporal features [B, embed_dim]
        """
        if timestamps.shape[1] == 1:
            # Convert unix timestamp to components
            # This is simplified - in practice, use proper datetime conversion
            hours = (timestamps % 86400 // 3600).long()
            days = (timestamps % 2592000 // 86400).long()
            months = (timestamps % 31536000 // 2592000).long()
            seasons = (months // 3).long()
        else:
            hours = timestamps[:, 0].long()
            days = timestamps[:, 1].long()
            months = timestamps[:, 2].long()
            seasons = timestamps[:, 3].long()
        
        hour_features = self.hour_encoder(hours)
        day_features = self.day_encoder(days)
        month_features = self.month_encoder(months)
        season_features = self.season_encoder(seasons)
        
        combined = torch.cat([
            hour_features, day_features, 
            month_features, season_features
        ], dim=-1)
        
        return self.fusion(combined)


class SinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding for continuous values"""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal encoding
        
        Args:
            x: Input values [B, 1]
        
        Returns:
            Encoded features [B, dim]
        """
        half_dim = self.dim // 2
        embeddings = torch.zeros(x.shape[0], self.dim, device=x.device)
        
        positions = x
        div_term = torch.exp(
            torch.arange(0, half_dim, device=x.device) * 
            -(torch.log(torch.tensor(self.max_period)) / half_dim)
        )
        
        embeddings[:, 0::2] = torch.sin(positions * div_term)
        embeddings[:, 1::2] = torch.cos(positions * div_term)
        
        return embeddings