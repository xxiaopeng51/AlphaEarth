"""
Metadata Encoder for Auxiliary Information
Handles geographic, temporal, and sensor metadata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np


class MetadataEncoder(nn.Module):
    """
    Encoder for various metadata associated with Earth observation data
    
    Features:
    - Geographic coordinates encoding
    - Temporal information encoding
    - Sensor parameters encoding
    - Weather conditions encoding
    - Elevation and terrain encoding
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        
        # Geographic encoding
        self.geo_encoder = GeographicEncoder(embed_dim // 4)
        
        # Temporal encoding
        self.temporal_encoder = TemporalMetadataEncoder(embed_dim // 4)
        
        # Sensor metadata encoding
        self.sensor_encoder = SensorMetadataEncoder(embed_dim // 4)
        
        # Environmental conditions encoding
        self.env_encoder = EnvironmentalEncoder(embed_dim // 4)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(
        self,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode metadata information
        
        Args:
            metadata: Dictionary containing various metadata fields:
                - 'coordinates': [B, 2] latitude, longitude
                - 'elevation': [B, 1] elevation in meters
                - 'time': [B, 4] year, month, day, hour
                - 'sensor': [B, N] sensor parameters
                - 'weather': [B, M] weather conditions
                - 'sun_angles': [B, 2] sun azimuth, sun elevation
                - 'view_angles': [B, 2] view azimuth, view zenith
        
        Returns:
            Encoded metadata features [B, embed_dim]
        """
        features = []
        
        # Geographic features
        if 'coordinates' in metadata:
            geo_features = self.geo_encoder(
                metadata['coordinates'],
                metadata.get('elevation', None)
            )
            features.append(geo_features)
        
        # Temporal features
        if 'time' in metadata:
            temporal_features = self.temporal_encoder(metadata['time'])
            features.append(temporal_features)
        
        # Sensor features
        sensor_data = {}
        for key in ['sensor', 'sun_angles', 'view_angles']:
            if key in metadata:
                sensor_data[key] = metadata[key]
        if sensor_data:
            sensor_features = self.sensor_encoder(sensor_data)
            features.append(sensor_features)
        
        # Environmental features
        if 'weather' in metadata:
            env_features = self.env_encoder(metadata['weather'])
            features.append(env_features)
        
        # Concatenate and fuse all features
        if features:
            combined = torch.cat(features, dim=-1)
            
            # Pad if necessary
            if combined.shape[-1] < self.embed_dim:
                padding = torch.zeros(
                    combined.shape[0],
                    self.embed_dim - combined.shape[-1],
                    device=combined.device
                )
                combined = torch.cat([combined, padding], dim=-1)
            
            return self.fusion(combined)
        
        # Return zeros if no metadata provided
        batch_size = next(iter(metadata.values())).shape[0] if metadata else 1
        return torch.zeros(batch_size, self.embed_dim, device=combined.device)


class GeographicEncoder(nn.Module):
    """Encode geographic location and elevation"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Sinusoidal encoding for coordinates
        self.coord_encoder = SinusoidalPositionalEncoding(embed_dim // 2)
        
        # Elevation encoding
        self.elevation_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
        )
        
        self.fusion = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        coordinates: torch.Tensor,
        elevation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode geographic information
        
        Args:
            coordinates: [B, 2] latitude, longitude
            elevation: [B, 1] elevation in meters
        
        Returns:
            Geographic features [B, embed_dim]
        """
        # Encode coordinates
        coord_features = self.coord_encoder(coordinates)
        
        # Encode elevation if provided
        if elevation is not None:
            # Normalize elevation (assuming range -500 to 9000 meters)
            elevation_norm = (elevation + 500) / 9500
            elev_features = self.elevation_encoder(elevation_norm)
        else:
            elev_features = torch.zeros_like(coord_features)
        
        # Combine features
        combined = torch.cat([coord_features, elev_features], dim=-1)
        return self.fusion(combined)


class TemporalMetadataEncoder(nn.Module):
    """Encode temporal metadata"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Embeddings for discrete time components
        self.year_encoder = nn.Linear(1, embed_dim // 4)
        self.month_encoder = nn.Embedding(12, embed_dim // 4)
        self.day_encoder = nn.Embedding(31, embed_dim // 4)
        self.hour_encoder = nn.Embedding(24, embed_dim // 4)
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
    
    def forward(self, time_info: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal information
        
        Args:
            time_info: [B, 4] year, month, day, hour
        
        Returns:
            Temporal features [B, embed_dim]
        """
        year = time_info[:, 0:1]
        month = time_info[:, 1].long()
        day = time_info[:, 2].long()
        hour = time_info[:, 3].long()
        
        # Normalize year (assuming 2000-2030 range)
        year_norm = (year - 2000) / 30
        
        year_features = self.year_encoder(year_norm)
        month_features = self.month_encoder(month)
        day_features = self.day_encoder(day)
        hour_features = self.hour_encoder(hour)
        
        combined = torch.cat([
            year_features, month_features,
            day_features, hour_features
        ], dim=-1)
        
        return self.fusion(combined)


class SensorMetadataEncoder(nn.Module):
    """Encode sensor-specific metadata"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Sun angle encoding
        self.sun_angle_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )
        
        # View angle encoding
        self.view_angle_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
        )
        
        self.fusion = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, sensor_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode sensor metadata
        
        Args:
            sensor_data: Dictionary with sensor parameters
        
        Returns:
            Sensor features [B, embed_dim]
        """
        features = []
        
        if 'sun_angles' in sensor_data:
            # Normalize angles to [-1, 1]
            sun_angles_norm = sensor_data['sun_angles'] / 180.0
            sun_features = self.sun_angle_encoder(sun_angles_norm)
            features.append(sun_features)
        
        if 'view_angles' in sensor_data:
            view_angles_norm = sensor_data['view_angles'] / 180.0
            view_features = self.view_angle_encoder(view_angles_norm)
            features.append(view_features)
        
        if features:
            combined = torch.cat(features, dim=-1)
            # Pad if necessary
            if combined.shape[-1] < self.fusion.in_features:
                padding = torch.zeros(
                    combined.shape[0],
                    self.fusion.in_features - combined.shape[-1],
                    device=combined.device
                )
                combined = torch.cat([combined, padding], dim=-1)
            return self.fusion(combined)
        
        return torch.zeros(1, self.fusion.out_features)


class EnvironmentalEncoder(nn.Module):
    """Encode environmental conditions"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        # Weather conditions encoding
        # Assuming weather vector contains: temperature, humidity, pressure, wind_speed, cloud_cover
        self.weather_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        """
        Encode weather conditions
        
        Args:
            weather: [B, 5] weather parameters
        
        Returns:
            Weather features [B, embed_dim]
        """
        # Normalize weather parameters
        # Temperature: -50 to 50°C -> [0, 1]
        # Humidity: 0 to 100% -> [0, 1]
        # Pressure: 900 to 1100 hPa -> [0, 1]
        # Wind speed: 0 to 50 m/s -> [0, 1]
        # Cloud cover: 0 to 100% -> [0, 1]
        
        weather_norm = weather.clone()
        weather_norm[:, 0] = (weather[:, 0] + 50) / 100  # Temperature
        weather_norm[:, 1] = weather[:, 1] / 100  # Humidity
        weather_norm[:, 2] = (weather[:, 2] - 900) / 200  # Pressure
        weather_norm[:, 3] = weather[:, 3] / 50  # Wind speed
        weather_norm[:, 4] = weather[:, 4] / 100  # Cloud cover
        
        return self.weather_encoder(weather_norm)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding for continuous coordinates"""
    
    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
    
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal encoding to coordinates
        
        Args:
            coordinates: [B, 2] latitude, longitude
        
        Returns:
            Encoded coordinates [B, embed_dim]
        """
        B = coordinates.shape[0]
        half_dim = self.embed_dim // 4
        
        # Normalize coordinates
        lat = coordinates[:, 0:1] / 90.0  # Latitude: -90 to 90
        lon = coordinates[:, 1:2] / 180.0  # Longitude: -180 to 180
        
        # Create frequency bands
        freqs = torch.exp(
            torch.arange(0, half_dim, device=coordinates.device) *
            -(np.log(self.max_period) / half_dim)
        )
        
        # Apply sinusoidal encoding
        lat_enc = torch.zeros(B, half_dim * 2, device=coordinates.device)
        lat_enc[:, 0::2] = torch.sin(lat * freqs)
        lat_enc[:, 1::2] = torch.cos(lat * freqs)
        
        lon_enc = torch.zeros(B, half_dim * 2, device=coordinates.device)
        lon_enc[:, 0::2] = torch.sin(lon * freqs)
        lon_enc[:, 1::2] = torch.cos(lon * freqs)
        
        return torch.cat([lat_enc, lon_enc], dim=-1)