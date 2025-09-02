"""
Multi-Modal Earth Dataset

支持多种地球观测数据源的数据集类，包括：
- Sentinel-2 (光学)
- Sentinel-1 (SAR)
- Landsat (光学)
- MODIS (环境)
- GEDI (激光雷达)
- 文本描述
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import random
from datetime import datetime, timedelta

from .data_sources import (
    SentinelDataSource, LandsatDataSource, SARDataSource,
    HyperspectralDataSource, EnvironmentalDataSource
)
from .transforms import MultiModalTransforms


class MultiModalEarthDataset(Dataset):
    """
    多模态地球观测数据集
    
    支持从多种数据源加载和组合数据，包括时间序列和空间对齐。
    """
    
    def __init__(
        self,
        data_config: Dict[str, Any],
        split: str = "train",
        num_samples: int = 10000,
        patch_size: int = 224,
        time_window: int = 16,
        temporal_resolution: str = "monthly",  # "daily", "weekly", "monthly"
        spatial_resolution: float = 10.0,  # 米
        enable_augmentation: bool = True,
        cache_data: bool = False,
        synthetic_mode: bool = True  # 用于演示，生成合成数据
    ):
        """
        Args:
            data_config: 数据配置
                {
                    "optical": {"source": "sentinel2", "bands": ["B02", "B03", "B04", "B08", "B11"]},
                    "sar": {"source": "sentinel1", "polarizations": ["VV", "VH"]},
                    "environmental": {"source": "era5", "variables": ["temperature", "precipitation"]},
                    "text": {"source": "osm", "include_descriptions": True}
                }
            split: 数据分割 ("train", "val", "test")
            num_samples: 样本数量
            patch_size: 空间patch大小
            time_window: 时间窗口长度
            temporal_resolution: 时间分辨率
            spatial_resolution: 空间分辨率 (米)
            enable_augmentation: 是否启用数据增强
            cache_data: 是否缓存数据
            synthetic_mode: 是否使用合成数据 (用于演示)
        """
        super().__init__()
        
        self.data_config = data_config
        self.split = split
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.time_window = time_window
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.enable_augmentation = enable_augmentation
        self.cache_data = cache_data
        self.synthetic_mode = synthetic_mode
        
        # 初始化数据源
        self.data_sources = self._initialize_data_sources()
        
        # 初始化变换
        self.transforms = MultiModalTransforms(
            patch_size=patch_size,
            enable_augmentation=enable_augmentation and (split == "train")
        )
        
        # 生成样本索引
        self.sample_indices = self._generate_sample_indices()
        
        # 缓存
        self.data_cache = {} if cache_data else None
    
    def _initialize_data_sources(self) -> Dict[str, Any]:
        """初始化各种数据源"""
        sources = {}
        
        for modality, config in self.data_config.items():
            source_type = config.get("source", "")
            
            if modality == "optical":
                if source_type == "sentinel2":
                    sources[modality] = SentinelDataSource(
                        bands=config.get("bands", ["B02", "B03", "B04", "B08", "B11"]),
                        resolution=self.spatial_resolution
                    )
                elif source_type == "landsat":
                    sources[modality] = LandsatDataSource(
                        bands=config.get("bands", ["B2", "B3", "B4", "B5", "B6", "B7"]),
                        resolution=self.spatial_resolution
                    )
                    
            elif modality == "sar":
                sources[modality] = SARDataSource(
                    polarizations=config.get("polarizations", ["VV", "VH"]),
                    resolution=self.spatial_resolution
                )
                
            elif modality == "hyperspectral":
                sources[modality] = HyperspectralDataSource(
                    num_bands=config.get("num_bands", 242),
                    resolution=config.get("resolution", 30.0)
                )
                
            elif modality == "environmental":
                sources[modality] = EnvironmentalDataSource(
                    variables=config.get("variables", ["temperature", "precipitation", "humidity"]),
                    resolution=config.get("resolution", 1000.0)
                )
        
        return sources
    
    def _generate_sample_indices(self) -> List[Dict[str, Any]]:
        """生成样本索引"""
        indices = []
        
        for i in range(self.num_samples):
            # 随机地理位置 (全球覆盖)
            lat = np.random.uniform(-60, 70)  # 避免极地区域
            lon = np.random.uniform(-180, 180)
            
            # 随机时间范围
            start_date = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 365*3))
            end_date = start_date + timedelta(days=365)  # 1年时间窗口
            
            # 随机有效期
            valid_start = start_date + timedelta(days=np.random.randint(0, 180))
            valid_end = valid_start + timedelta(days=np.random.randint(30, 365))
            
            sample_info = {
                "index": i,
                "coordinates": [lat, lon],
                "time_range": [start_date, end_date],
                "valid_period": [valid_start, valid_end],
                "region": self._get_region_name(lat, lon)
            }
            indices.append(sample_info)
        
        return indices
    
    def _get_region_name(self, lat: float, lon: float) -> str:
        """根据坐标获取区域名称"""
        if lat > 50:
            return "arctic"
        elif lat > 23.5:
            return "temperate_north"
        elif lat > -23.5:
            return "tropical"
        elif lat > -50:
            return "temperate_south"
        else:
            return "antarctic"
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        if self.cache_data and idx in self.data_cache:
            return self.data_cache[idx]
        
        sample_info = self.sample_indices[idx]
        
        if self.synthetic_mode:
            # 生成合成数据用于演示
            sample = self._generate_synthetic_sample(sample_info)
        else:
            # 从真实数据源加载
            sample = self._load_real_sample(sample_info)
        
        # 应用变换
        sample = self.transforms(sample)
        
        # 缓存样本
        if self.cache_data:
            self.data_cache[idx] = sample
        
        return sample
    
    def _generate_synthetic_sample(self, sample_info: Dict[str, Any]) -> Dict[str, Any]:
        """生成合成样本用于演示和测试"""
        coordinates = sample_info["coordinates"]
        time_range = sample_info["time_range"]
        valid_period = sample_info["valid_period"]
        
        # 生成时间序列
        timestamps = self._generate_timestamps(time_range, self.time_window)
        
        # 生成多模态数据
        multimodal_data = {}
        
        for modality, source in self.data_sources.items():
            if modality == "optical":
                # 光学数据: 模拟季节性变化和地表类型
                data = self._generate_optical_data(coordinates, timestamps)
            elif modality == "sar":
                # SAR数据: 模拟后散射系数
                data = self._generate_sar_data(coordinates, timestamps)
            elif modality == "hyperspectral":
                # 高光谱数据
                data = self._generate_hyperspectral_data(coordinates, timestamps)
            elif modality == "environmental":
                # 环境数据: 模拟气象变量
                data = self._generate_environmental_data(coordinates, timestamps)
            else:
                # 默认随机数据
                channels = getattr(source, 'num_channels', 3)
                data = torch.randn(self.time_window, self.patch_size, self.patch_size, channels)
            
            multimodal_data[modality] = data
        
        # 生成文本描述
        text_description = self._generate_text_description(coordinates, sample_info["region"])
        
        return {
            "multimodal_data": multimodal_data,
            "timestamps": timestamps,
            "coordinates": torch.tensor(coordinates, dtype=torch.float32),
            "valid_period": torch.tensor([
                valid_period[0].timestamp() * 1000,  # 转换为毫秒
                valid_period[1].timestamp() * 1000
            ], dtype=torch.float32),
            "text_description": text_description,
            "region": sample_info["region"],
            "sample_id": sample_info["index"]
        }
    
    def _generate_timestamps(self, time_range: List[datetime], num_frames: int) -> torch.Tensor:
        """生成时间戳序列"""
        start_ts = time_range[0].timestamp() * 1000  # 毫秒
        end_ts = time_range[1].timestamp() * 1000
        
        # 根据时间分辨率生成间隔
        if self.temporal_resolution == "daily":
            interval = 24 * 3600 * 1000  # 1天
        elif self.temporal_resolution == "weekly":
            interval = 7 * 24 * 3600 * 1000  # 1周
        elif self.temporal_resolution == "monthly":
            interval = 30 * 24 * 3600 * 1000  # 1月
        else:
            interval = (end_ts - start_ts) / num_frames
        
        # 生成规律时间序列 + 小幅随机扰动
        base_timestamps = np.arange(start_ts, end_ts, interval)[:num_frames]
        noise = np.random.normal(0, interval * 0.1, len(base_timestamps))
        timestamps = base_timestamps + noise
        
        # 填充到指定长度
        if len(timestamps) < num_frames:
            last_ts = timestamps[-1] if len(timestamps) > 0 else start_ts
            padding = np.full(num_frames - len(timestamps), last_ts)
            timestamps = np.concatenate([timestamps, padding])
        
        return torch.tensor(timestamps[:num_frames], dtype=torch.float32)
    
    def _generate_optical_data(self, coordinates: List[float], timestamps: torch.Tensor) -> torch.Tensor:
        """生成模拟光学数据"""
        lat, lon = coordinates
        T = len(timestamps)
        
        # 基于地理位置和季节的光谱特征
        base_reflectance = self._get_base_reflectance(lat, lon)
        
        # 时间变化 (季节性)
        seasonal_variation = self._compute_seasonal_variation(timestamps, lat)
        
        # 生成多波段数据
        optical_data = torch.zeros(T, self.patch_size, self.patch_size, 13)  # 13个波段
        
        for t in range(T):
            # 基础反射率 + 季节变化 + 空间变化 + 噪声
            base = base_reflectance * (1 + seasonal_variation[t] * 0.3)
            
            # 空间变化 (模拟地表异质性)
            spatial_var = torch.randn(self.patch_size, self.patch_size, 1) * 0.1
            
            # 光谱变化
            spectral_profile = self._generate_spectral_profile(base)
            
            optical_data[t] = spectral_profile + spatial_var + torch.randn_like(spatial_var) * 0.05
        
        return torch.clamp(optical_data, 0, 1)
    
    def _generate_sar_data(self, coordinates: List[float], timestamps: torch.Tensor) -> torch.Tensor:
        """生成模拟SAR数据"""
        lat, lon = coordinates
        T = len(timestamps)
        
        # SAR后散射系数 (dB)
        base_sigma0 = self._get_base_backscatter(lat, lon)
        
        sar_data = torch.zeros(T, self.patch_size, self.patch_size, 4)  # VV, VH, coherence, angle
        
        for t in range(T):
            # VV极化
            vv = base_sigma0["vv"] + torch.randn(self.patch_size, self.patch_size) * 2
            # VH极化  
            vh = base_sigma0["vh"] + torch.randn(self.patch_size, self.patch_size) * 2
            # 相干性
            coherence = torch.rand(self.patch_size, self.patch_size) * 0.8 + 0.1
            # 入射角
            angle = torch.full((self.patch_size, self.patch_size), 35.0) + torch.randn(self.patch_size, self.patch_size) * 5
            
            sar_data[t] = torch.stack([vv, vh, coherence, angle], dim=-1)
        
        return sar_data
    
    def _generate_hyperspectral_data(self, coordinates: List[float], timestamps: torch.Tensor) -> torch.Tensor:
        """生成模拟高光谱数据"""
        T = len(timestamps)
        num_bands = 242  # PRISMA波段数
        
        # 基于地表类型的光谱特征
        base_spectrum = self._generate_base_spectrum(coordinates, num_bands)
        
        hyperspectral_data = torch.zeros(T, self.patch_size, self.patch_size, num_bands)
        
        for t in range(T):
            # 添加时间和空间变化
            temporal_factor = 1 + 0.1 * torch.sin(torch.tensor(t / T * 2 * np.pi))
            spatial_noise = torch.randn(self.patch_size, self.patch_size, 1) * 0.05
            
            hyperspectral_data[t] = base_spectrum * temporal_factor + spatial_noise
        
        return torch.clamp(hyperspectral_data, 0, 1)
    
    def _generate_environmental_data(self, coordinates: List[float], timestamps: torch.Tensor) -> torch.Tensor:
        """生成模拟环境数据"""
        lat, lon = coordinates
        T = len(timestamps)
        
        # 基于地理位置的气候特征
        base_climate = self._get_base_climate(lat, lon)
        
        env_data = torch.zeros(T, self.patch_size, self.patch_size, 8)
        
        for t in range(T):
            # 季节性气候变化
            seasonal_temp = base_climate["temperature"] + 10 * np.sin((t / T) * 2 * np.pi)
            seasonal_precip = base_climate["precipitation"] * (1 + 0.5 * np.sin((t / T) * 2 * np.pi + np.pi/2))
            
            # 其他环境变量
            humidity = np.random.normal(60, 15)  # 相对湿度
            wind_speed = np.random.exponential(5)  # 风速
            pressure = np.random.normal(1013, 10)  # 气压
            solar_radiation = np.random.normal(200, 50)  # 太阳辐射
            
            env_vars = torch.tensor([
                seasonal_temp, seasonal_precip, humidity, wind_speed,
                pressure, solar_radiation, lat, lon
            ], dtype=torch.float32)
            
            # 广播到空间维度
            env_data[t] = env_vars.unsqueeze(0).unsqueeze(0).expand(
                self.patch_size, self.patch_size, -1
            )
        
        return env_data
    
    def _generate_text_description(self, coordinates: List[float], region: str) -> str:
        """生成地理位置的文本描述"""
        lat, lon = coordinates
        
        # 基础地理描述模板
        templates = [
            f"Satellite imagery of {region} region at latitude {lat:.2f}, longitude {lon:.2f}",
            f"Earth observation data from {region} area, coordinates ({lat:.2f}°, {lon:.2f}°)",
            f"Remote sensing data covering {region} landscape at {lat:.2f}N, {lon:.2f}E",
        ]
        
        # 添加地表类型描述
        landcover_types = self._infer_landcover(lat, lon)
        landcover_desc = ", ".join(landcover_types)
        
        base_desc = random.choice(templates)
        full_desc = f"{base_desc}. Dominant land cover: {landcover_desc}."
        
        return full_desc
    
    def _get_base_reflectance(self, lat: float, lon: float) -> float:
        """根据地理位置获取基础反射率"""
        # 简化的地表类型映射
        if abs(lat) > 60:  # 极地
            return 0.8  # 高反射率 (冰雪)
        elif abs(lat) < 23.5:  # 热带
            return 0.15  # 低反射率 (植被)
        else:  # 温带
            return 0.3  # 中等反射率
    
    def _compute_seasonal_variation(self, timestamps: torch.Tensor, lat: float) -> torch.Tensor:
        """计算季节性变化"""
        # 将时间戳转换为年内天数
        year_progress = (timestamps - timestamps.min()) / (365 * 24 * 3600 * 1000)
        
        # 季节性变化 (北半球和南半球相位相反)
        phase_shift = 0 if lat > 0 else np.pi
        seasonal = torch.sin(year_progress * 2 * np.pi + phase_shift)
        
        return seasonal
    
    def _generate_spectral_profile(self, base_reflectance: float) -> torch.Tensor:
        """生成光谱剖面"""
        # 简化的13波段光谱剖面 (Sentinel-2 + Landsat)
        wavelengths = torch.tensor([443, 490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190, 1570, 2260])
        
        # 基于波长的反射率变化
        spectral_response = base_reflectance * (1 + 0.1 * torch.sin(wavelengths / 100))
        
        # 扩展到空间维度
        return spectral_response.unsqueeze(0).unsqueeze(0).expand(
            self.patch_size, self.patch_size, -1
        )
    
    def _get_base_backscatter(self, lat: float, lon: float) -> Dict[str, torch.Tensor]:
        """获取基础后散射系数"""
        # 基于地表类型的SAR后散射
        if abs(lat) > 60:  # 极地/冰雪
            vv_sigma0 = -15  # dB
            vh_sigma0 = -25  # dB
        elif abs(lat) < 23.5:  # 热带森林
            vv_sigma0 = -8   # dB
            vh_sigma0 = -15  # dB  
        else:  # 温带/农田
            vv_sigma0 = -12  # dB
            vh_sigma0 = -18  # dB
        
        return {
            "vv": torch.full((self.patch_size, self.patch_size), vv_sigma0),
            "vh": torch.full((self.patch_size, self.patch_size), vh_sigma0)
        }
    
    def _generate_base_spectrum(self, coordinates: List[float], num_bands: int) -> torch.Tensor:
        """生成基础光谱特征"""
        lat, lon = coordinates
        
        # 基于地理位置的光谱特征
        base_value = 0.3 + 0.2 * np.sin(lat * np.pi / 180)  # 纬度相关
        
        # 生成光谱曲线
        wavelengths = torch.linspace(400, 2500, num_bands)  # 400-2500nm
        spectrum = base_value * (1 + 0.1 * torch.sin(wavelengths / 100))
        
        return spectrum.unsqueeze(0).unsqueeze(0).expand(
            self.patch_size, self.patch_size, -1
        )
    
    def _get_base_climate(self, lat: float, lon: float) -> Dict[str, float]:
        """获取基础气候特征"""
        # 简化的气候模型
        temp_base = 25 - abs(lat) * 0.5  # 温度随纬度降低
        precip_base = 100 + 50 * np.sin(abs(lat) * np.pi / 90)  # 降水模式
        
        return {
            "temperature": temp_base,
            "precipitation": precip_base
        }
    
    def _infer_landcover(self, lat: float, lon: float) -> List[str]:
        """推断土地覆盖类型"""
        landcover_types = []
        
        if abs(lat) > 60:
            landcover_types.extend(["ice", "tundra", "sparse_vegetation"])
        elif abs(lat) < 23.5:
            landcover_types.extend(["tropical_forest", "savanna", "cropland"])
        else:
            landcover_types.extend(["temperate_forest", "grassland", "urban", "cropland"])
        
        # 随机选择1-3种类型
        return random.sample(landcover_types, random.randint(1, min(3, len(landcover_types))))
    
    def _load_real_sample(self, sample_info: Dict[str, Any]) -> Dict[str, Any]:
        """从真实数据源加载样本 (待实现)"""
        # TODO: 实现真实数据加载逻辑
        # 这里需要连接到实际的数据源 (Google Earth Engine, AWS Open Data等)
        raise NotImplementedError("Real data loading not implemented yet. Use synthetic_mode=True for now.")
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """获取样本元信息"""
        return self.sample_indices[idx]
    
    def get_modality_stats(self) -> Dict[str, Dict[str, float]]:
        """获取各模态的统计信息"""
        stats = {}
        
        # 计算数据集统计 (基于前100个样本)
        sample_size = min(100, len(self))
        
        for modality in self.data_sources.keys():
            modality_data = []
            for i in range(sample_size):
                sample = self[i]
                if modality in sample["multimodal_data"]:
                    modality_data.append(sample["multimodal_data"][modality])
            
            if modality_data:
                stacked_data = torch.stack(modality_data)
                stats[modality] = {
                    "mean": stacked_data.mean().item(),
                    "std": stacked_data.std().item(),
                    "min": stacked_data.min().item(),
                    "max": stacked_data.max().item(),
                    "shape": list(stacked_data.shape[1:])  # 排除batch维度
                }
        
        return stats