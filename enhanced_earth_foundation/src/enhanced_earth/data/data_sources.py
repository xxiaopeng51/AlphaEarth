"""
Data Sources

各种地球观测数据源的抽象和实现，支持：
- Sentinel-2 (光学)
- Sentinel-1 (SAR)  
- Landsat (光学)
- MODIS (环境)
- GEDI (激光雷达)
- ERA5-Land (气象)
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    resolution: float  # 空间分辨率 (米)
    temporal_resolution: str  # 时间分辨率
    bands_or_channels: List[str]
    data_range: Tuple[float, float]  # 数据值范围
    units: str
    description: str


class BaseDataSource(ABC):
    """数据源基类"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.name = config.name
        self.resolution = config.resolution
        self.num_channels = len(config.bands_or_channels)
    
    @abstractmethod
    def load_data(
        self,
        coordinates: Tuple[float, float],
        time_range: Tuple[datetime, datetime],
        patch_size: int
    ) -> torch.Tensor:
        """加载数据"""
        pass
    
    @abstractmethod
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """预处理数据"""
        pass


class SentinelDataSource(BaseDataSource):
    """Sentinel-2光学数据源"""
    
    def __init__(
        self,
        bands: List[str] = None,
        resolution: float = 10.0,
        cloud_mask: bool = True
    ):
        if bands is None:
            bands = ["B02", "B03", "B04", "B08", "B11", "B12"]  # 主要波段
        
        config = DataSourceConfig(
            name="sentinel2",
            resolution=resolution,
            temporal_resolution="5_days",
            bands_or_channels=bands,
            data_range=(0, 10000),  # DN值范围
            units="DN",
            description="Sentinel-2 MultiSpectral Instrument (MSI) data"
        )
        super().__init__(config)
        
        self.bands = bands
        self.cloud_mask = cloud_mask
        
        # 波段特性
        self.band_properties = {
            "B01": {"wavelength": 443, "resolution": 60},   # Coastal aerosol
            "B02": {"wavelength": 490, "resolution": 10},   # Blue
            "B03": {"wavelength": 560, "resolution": 10},   # Green
            "B04": {"wavelength": 665, "resolution": 10},   # Red
            "B05": {"wavelength": 705, "resolution": 20},   # Red edge 1
            "B06": {"wavelength": 740, "resolution": 20},   # Red edge 2
            "B07": {"wavelength": 783, "resolution": 20},   # Red edge 3
            "B08": {"wavelength": 842, "resolution": 10},   # NIR
            "B8A": {"wavelength": 865, "resolution": 20},   # Narrow NIR
            "B09": {"wavelength": 945, "resolution": 60},   # Water vapour
            "B10": {"wavelength": 1375, "resolution": 60},  # SWIR - Cirrus
            "B11": {"wavelength": 1610, "resolution": 20},  # SWIR 1
            "B12": {"wavelength": 2190, "resolution": 20}   # SWIR 2
        }
    
    def load_data(
        self,
        coordinates: Tuple[float, float],
        time_range: Tuple[datetime, datetime],
        patch_size: int
    ) -> torch.Tensor:
        """加载Sentinel-2数据 (合成版本)"""
        lat, lon = coordinates
        start_date, end_date = time_range
        
        # 计算时间序列长度
        time_span = (end_date - start_date).days
        num_images = min(time_span // 5, 50)  # 每5天一张图，最多50张
        
        # 生成模拟数据
        data = torch.zeros(num_images, patch_size, patch_size, len(self.bands))
        
        for i, band in enumerate(self.bands):
            band_props = self.band_properties[band]
            wavelength = band_props["wavelength"]
            
            # 基于波长和地理位置的反射率模拟
            base_reflectance = self._simulate_band_reflectance(wavelength, lat, lon)
            
            # 时间变化
            for t in range(num_images):
                temporal_factor = 1 + 0.2 * np.sin(t / num_images * 2 * np.pi)
                spatial_noise = torch.randn(patch_size, patch_size) * 0.1
                
                data[t, :, :, i] = base_reflectance * temporal_factor + spatial_noise
        
        return torch.clamp(data * 10000, 0, 10000)  # 转换为DN值
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Sentinel-2数据预处理"""
        # 转换为反射率
        reflectance = data / 10000.0
        
        # 裁剪到合理范围
        reflectance = torch.clamp(reflectance, 0, 1)
        
        # 可选的大气校正 (简化)
        # reflectance = self._atmospheric_correction(reflectance)
        
        return reflectance
    
    def _simulate_band_reflectance(self, wavelength: float, lat: float, lon: float) -> torch.Tensor:
        """模拟特定波段的反射率"""
        # 基于地表类型和波长的反射率模拟
        if wavelength < 500:  # 蓝光
            base = 0.1
        elif wavelength < 600:  # 绿光
            base = 0.15
        elif wavelength < 700:  # 红光
            base = 0.12
        elif wavelength < 900:  # 近红外
            base = 0.4 if abs(lat) < 50 else 0.2  # 植被区域高反射
        else:  # 短波红外
            base = 0.2
        
        # 添加地理位置相关的变化
        geo_factor = 1 + 0.1 * np.sin(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
        
        return torch.full((1, 1), base * geo_factor)


class LandsatDataSource(BaseDataSource):
    """Landsat光学数据源"""
    
    def __init__(
        self,
        bands: List[str] = None,
        resolution: float = 30.0,
        satellite: str = "landsat8"
    ):
        if bands is None:
            bands = ["B2", "B3", "B4", "B5", "B6", "B7"]  # Landsat 8/9主要波段
        
        config = DataSourceConfig(
            name=f"{satellite}",
            resolution=resolution,
            temporal_resolution="16_days",
            bands_or_channels=bands,
            data_range=(0, 65535),  # 16-bit DN值
            units="DN",
            description=f"{satellite} Operational Land Imager (OLI) data"
        )
        super().__init__(config)
        
        self.bands = bands
        self.satellite = satellite
    
    def load_data(self, coordinates, time_range, patch_size) -> torch.Tensor:
        """加载Landsat数据 (合成版本)"""
        # 类似Sentinel-2的实现，但时间分辨率不同
        lat, lon = coordinates
        time_span = (time_range[1] - time_range[0]).days
        num_images = min(time_span // 16, 25)  # 每16天一张图
        
        data = torch.rand(num_images, patch_size, patch_size, len(self.bands)) * 65535
        return data
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Landsat数据预处理"""
        # 转换为反射率
        reflectance = data / 65535.0
        return torch.clamp(reflectance, 0, 1)


class SARDataSource(BaseDataSource):
    """Sentinel-1 SAR数据源"""
    
    def __init__(
        self,
        polarizations: List[str] = None,
        resolution: float = 10.0,
        product_type: str = "GRD"
    ):
        if polarizations is None:
            polarizations = ["VV", "VH"]
        
        # 添加相干性和入射角
        channels = polarizations + ["coherence", "incidence_angle"]
        
        config = DataSourceConfig(
            name="sentinel1",
            resolution=resolution,
            temporal_resolution="12_days",
            bands_or_channels=channels,
            data_range=(-30, 10),  # dB范围
            units="dB",
            description="Sentinel-1 Synthetic Aperture Radar (SAR) data"
        )
        super().__init__(config)
        
        self.polarizations = polarizations
        self.product_type = product_type
    
    def load_data(self, coordinates, time_range, patch_size) -> torch.Tensor:
        """加载SAR数据 (合成版本)"""
        time_span = (time_range[1] - time_range[0]).days
        num_images = min(time_span // 12, 30)  # 每12天一张图
        
        data = torch.zeros(num_images, patch_size, patch_size, len(self.config.bands_or_channels))
        
        for t in range(num_images):
            # VV极化 (通常较强)
            data[t, :, :, 0] = torch.randn(patch_size, patch_size) * 3 - 10
            # VH极化 (通常较弱)
            data[t, :, :, 1] = torch.randn(patch_size, patch_size) * 3 - 18
            # 相干性 [0, 1]
            data[t, :, :, 2] = torch.rand(patch_size, patch_size) * 0.8 + 0.1
            # 入射角 [20, 45]度
            data[t, :, :, 3] = torch.randn(patch_size, patch_size) * 5 + 35
        
        return data
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """SAR数据预处理"""
        processed = data.clone()
        
        # 后散射系数转换为线性尺度
        processed[:, :, :, :2] = torch.pow(10, processed[:, :, :, :2] / 10)
        
        # 相干性和入射角保持原样
        
        return processed


class HyperspectralDataSource(BaseDataSource):
    """高光谱数据源 (PRISMA, EnMAP等)"""
    
    def __init__(
        self,
        num_bands: int = 242,
        resolution: float = 30.0,
        spectral_range: Tuple[float, float] = (400, 2500)
    ):
        bands = [f"Band_{i:03d}" for i in range(num_bands)]
        
        config = DataSourceConfig(
            name="hyperspectral",
            resolution=resolution,
            temporal_resolution="variable",
            bands_or_channels=bands,
            data_range=(0, 1),
            units="reflectance",
            description="Hyperspectral imagery data"
        )
        super().__init__(config)
        
        self.num_bands = num_bands
        self.spectral_range = spectral_range
        self.wavelengths = np.linspace(spectral_range[0], spectral_range[1], num_bands)
    
    def load_data(self, coordinates, time_range, patch_size) -> torch.Tensor:
        """加载高光谱数据 (合成版本)"""
        time_span = (time_range[1] - time_range[0]).days
        num_images = min(time_span // 30, 12)  # 每月一张图
        
        lat, lon = coordinates
        
        # 生成光谱曲线
        base_spectrum = self._generate_spectral_signature(lat, lon)
        
        data = torch.zeros(num_images, patch_size, patch_size, self.num_bands)
        
        for t in range(num_images):
            for i, wavelength in enumerate(self.wavelengths):
                # 基础光谱 + 时间变化 + 空间变化 + 噪声
                temporal_factor = 1 + 0.1 * np.sin(t / num_images * 2 * np.pi)
                spatial_var = torch.randn(patch_size, patch_size) * 0.05
                
                data[t, :, :, i] = base_spectrum[i] * temporal_factor + spatial_var
        
        return torch.clamp(data, 0, 1)
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """高光谱数据预处理"""
        # 光谱标准化
        data_norm = (data - data.mean(dim=-1, keepdim=True)) / (data.std(dim=-1, keepdim=True) + 1e-8)
        
        # 噪声去除 (简单的高斯滤波)
        # data_denoised = F.conv1d(data_norm.transpose(-1, -2), 
        #                         torch.ones(1, 1, 3, device=data.device) / 3, 
        #                         padding=1).transpose(-1, -2)
        
        return torch.clamp(data_norm * 0.1 + 0.5, 0, 1)  # 重新缩放到[0,1]
    
    def _generate_spectral_signature(self, lat: float, lon: float) -> np.ndarray:
        """生成光谱特征"""
        # 基于地理位置的典型光谱特征
        if abs(lat) > 60:  # 极地
            # 冰雪光谱
            spectrum = 0.8 - 0.3 * np.exp(-(self.wavelengths - 1600)**2 / (200**2))
        elif abs(lat) < 23.5:  # 热带
            # 植被光谱
            spectrum = 0.1 + 0.6 * np.exp(-(self.wavelengths - 800)**2 / (100**2))
        else:  # 温带
            # 混合光谱
            spectrum = 0.3 + 0.2 * np.sin(self.wavelengths / 200)
        
        return spectrum


class SARDataSource(BaseDataSource):
    """SAR数据源 (Sentinel-1, PALSAR等)"""
    
    def __init__(
        self,
        polarizations: List[str] = None,
        resolution: float = 10.0,
        product_type: str = "GRD"
    ):
        if polarizations is None:
            polarizations = ["VV", "VH"]
        
        # 添加额外的SAR产品
        channels = polarizations + ["coherence", "incidence_angle"]
        
        config = DataSourceConfig(
            name="sentinel1_sar",
            resolution=resolution,
            temporal_resolution="12_days",
            bands_or_channels=channels,
            data_range=(-30, 10),
            units="dB",
            description="Synthetic Aperture Radar data"
        )
        super().__init__(config)
        
        self.polarizations = polarizations
        self.product_type = product_type
    
    def load_data(self, coordinates, time_range, patch_size) -> torch.Tensor:
        """加载SAR数据"""
        lat, lon = coordinates
        time_span = (time_range[1] - time_range[0]).days
        num_images = min(time_span // 12, 30)
        
        data = torch.zeros(num_images, patch_size, patch_size, len(self.config.bands_or_channels))
        
        # 基于地表类型的后散射特征
        base_backscatter = self._get_surface_backscatter(lat, lon)
        
        for t in range(num_images):
            for i, pol in enumerate(self.polarizations):
                base_sigma0 = base_backscatter[pol.lower()]
                temporal_var = torch.randn(patch_size, patch_size) * 2
                data[t, :, :, i] = base_sigma0 + temporal_var
            
            # 相干性
            data[t, :, :, -2] = torch.rand(patch_size, patch_size) * 0.7 + 0.2
            # 入射角
            data[t, :, :, -1] = torch.full((patch_size, patch_size), 35.0) + torch.randn(patch_size, patch_size) * 3
        
        return data
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """SAR数据预处理"""
        processed = data.clone()
        
        # 后散射系数去斑点 (简化)
        for i in range(len(self.polarizations)):
            processed[:, :, :, i] = self._despeckle(processed[:, :, :, i])
        
        return processed
    
    def _get_surface_backscatter(self, lat: float, lon: float) -> Dict[str, float]:
        """获取地表后散射特征"""
        if abs(lat) > 60:  # 极地
            return {"vv": -15, "vh": -25}
        elif abs(lat) < 23.5:  # 热带森林
            return {"vv": -8, "vh": -15}
        else:  # 农田/草地
            return {"vv": -12, "vh": -18}
    
    def _despeckle(self, sar_image: torch.Tensor) -> torch.Tensor:
        """简单的SAR去斑点处理"""
        # 使用中值滤波去斑点
        kernel_size = 3
        padding = kernel_size // 2
        
        # 简化实现：使用平均滤波
        filtered = F.avg_pool2d(
            sar_image.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).squeeze(0).squeeze(0)
        
        return filtered


class EnvironmentalDataSource(BaseDataSource):
    """环境数据源 (ERA5-Land, DEM等)"""
    
    def __init__(
        self,
        variables: List[str] = None,
        resolution: float = 1000.0
    ):
        if variables is None:
            variables = [
                "temperature_2m", "precipitation", "relative_humidity",
                "wind_speed", "surface_pressure", "solar_radiation",
                "elevation", "slope"
            ]
        
        config = DataSourceConfig(
            name="environmental",
            resolution=resolution,
            temporal_resolution="hourly",
            bands_or_channels=variables,
            data_range=(-50, 50),  # 根据变量而定
            units="various",
            description="Environmental and meteorological data"
        )
        super().__init__(config)
        
        self.variables = variables
        
        # 变量属性
        self.variable_properties = {
            "temperature_2m": {"range": (-40, 50), "unit": "°C"},
            "precipitation": {"range": (0, 100), "unit": "mm/h"},
            "relative_humidity": {"range": (0, 100), "unit": "%"},
            "wind_speed": {"range": (0, 30), "unit": "m/s"},
            "surface_pressure": {"range": (950, 1050), "unit": "hPa"},
            "solar_radiation": {"range": (0, 1000), "unit": "W/m²"},
            "elevation": {"range": (-500, 8000), "unit": "m"},
            "slope": {"range": (0, 90), "unit": "degrees"}
        }
    
    def load_data(self, coordinates, time_range, patch_size) -> torch.Tensor:
        """加载环境数据"""
        lat, lon = coordinates
        time_span = (time_range[1] - time_range[0]).days
        num_timesteps = min(time_span * 24, 100)  # 每小时一个数据点，最多100个
        
        data = torch.zeros(num_timesteps, patch_size, patch_size, len(self.variables))
        
        for t in range(num_timesteps):
            for i, var in enumerate(self.variables):
                var_props = self.variable_properties.get(var, {"range": (0, 1)})
                var_range = var_props["range"]
                
                # 基于地理位置和时间的变量模拟
                if var == "temperature_2m":
                    base_temp = 25 - abs(lat) * 0.5  # 纬度效应
                    seasonal = 10 * np.sin(t / num_timesteps * 2 * np.pi)  # 季节变化
                    value = base_temp + seasonal
                elif var == "precipitation":
                    # 降水的随机性较大
                    value = np.random.exponential(2) if np.random.random() < 0.3 else 0
                elif var == "elevation":
                    # 地形相对稳定
                    base_elev = abs(lat) * 50  # 简化的地形模型
                    value = base_elev + np.random.normal(0, 100)
                else:
                    # 其他变量使用随机值
                    value = np.random.uniform(*var_range)
                
                # 标准化到[0, 1]
                normalized_value = (value - var_range[0]) / (var_range[1] - var_range[0])
                
                data[t, :, :, i] = torch.full((patch_size, patch_size), normalized_value)
        
        return data
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """环境数据预处理"""
        # 时间平滑
        smoothed = self._temporal_smoothing(data)
        
        # 标准化
        normalized = (smoothed - smoothed.mean(dim=(1, 2), keepdim=True)) / \
                    (smoothed.std(dim=(1, 2), keepdim=True) + 1e-8)
        
        return normalized
    
    def _temporal_smoothing(self, data: torch.Tensor, window_size: int = 3) -> torch.Tensor:
        """时间维度平滑"""
        T = data.shape[0]
        if T <= window_size:
            return data
        
        # 简单的移动平均
        smoothed = data.clone()
        for t in range(window_size // 2, T - window_size // 2):
            start_idx = t - window_size // 2
            end_idx = t + window_size // 2 + 1
            smoothed[t] = data[start_idx:end_idx].mean(dim=0)
        
        return smoothed


class LiDARDataSource(BaseDataSource):
    """激光雷达数据源 (GEDI, ICESat-2等)"""
    
    def __init__(
        self,
        resolution: float = 25.0,
        include_quality: bool = True
    ):
        channels = ["canopy_height", "ground_elevation", "canopy_cover"]
        if include_quality:
            channels.append("quality_flag")
        
        config = DataSourceConfig(
            name="lidar",
            resolution=resolution,
            temporal_resolution="irregular",
            bands_or_channels=channels,
            data_range=(0, 100),  # 高度范围 (米)
            units="meters",
            description="LiDAR elevation and canopy data"
        )
        super().__init__(config)
        
        self.include_quality = include_quality
    
    def load_data(self, coordinates, time_range, patch_size) -> torch.Tensor:
        """加载LiDAR数据"""
        lat, lon = coordinates
        
        # LiDAR数据通常稀疏，时间分辨率不规律
        num_acquisitions = np.random.randint(1, 5)  # 1-4次获取
        
        data = torch.zeros(num_acquisitions, patch_size, patch_size, len(self.config.bands_or_channels))
        
        # 基于地理位置的地形特征
        base_elevation = abs(lat) * 20  # 简化地形
        canopy_height_base = max(0, 30 - abs(lat))  # 热带地区植被更高
        
        for t in range(num_acquisitions):
            # 地面高程
            ground_elev = base_elevation + torch.randn(patch_size, patch_size) * 10
            data[t, :, :, 0] = torch.clamp(ground_elev, 0, 8000)
            
            # 冠层高度
            canopy_height = canopy_height_base + torch.randn(patch_size, patch_size) * 5
            data[t, :, :, 1] = torch.clamp(canopy_height, 0, 80)
            
            # 冠层覆盖度
            canopy_cover = torch.rand(patch_size, patch_size) * 100
            data[t, :, :, 2] = canopy_cover
            
            # 质量标志
            if self.include_quality:
                quality = torch.randint(0, 4, (patch_size, patch_size)).float()  # 0-3质量等级
                data[t, :, :, 3] = quality
        
        return data
    
    def preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """LiDAR数据预处理"""
        # 高度数据标准化
        processed = data.clone()
        
        # 地面高程标准化
        processed[:, :, :, 0] = processed[:, :, :, 0] / 8000.0
        
        # 冠层高度标准化
        processed[:, :, :, 1] = processed[:, :, :, 1] / 80.0
        
        # 覆盖度已经是百分比
        processed[:, :, :, 2] = processed[:, :, :, 2] / 100.0
        
        # 质量标志标准化
        if self.include_quality:
            processed[:, :, :, 3] = processed[:, :, :, 3] / 3.0
        
        return processed