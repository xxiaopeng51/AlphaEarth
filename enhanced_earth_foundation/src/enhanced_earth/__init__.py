"""
Enhanced Earth Foundation Model

一个融合AlphaEarth、Clay、SatCLIP、Prithvi等模型优势的
全球尺度多模态地球基础模型。

主要特性:
- 多模态数据支持 (光学、雷达、高光谱、文本等)
- 改进的Space-Time-Precision架构
- 动态嵌入和自适应融合
- 全球尺度覆盖和跨区域泛化
- 可扩展的训练框架
"""

from .models.enhanced_earth_model import EnhancedEarthFoundationModel
from .models.multimodal_encoder import MultiModalEncoder
from .models.stp_encoder import EnhancedSTPEncoder
from .models.temporal_summarizer import TemporalSummarizer
from .data.multimodal_dataset import MultiModalEarthDataset
from .data.data_loaders import create_multimodal_dataloader
from .training.trainer import EnhancedEarthTrainer

__version__ = "0.1.0"

__all__ = [
    "EnhancedEarthFoundationModel",
    "MultiModalEncoder", 
    "EnhancedSTPEncoder",
    "TemporalSummarizer",
    "MultiModalEarthDataset",
    "create_multimodal_dataloader",
    "EnhancedEarthTrainer",
]