"""数据变换和增强"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List
import random


class MultiModalTransforms:
    """多模态数据变换"""
    
    def __init__(self, patch_size: int = 224, enable_augmentation: bool = True):
        self.patch_size = patch_size
        self.enable_augmentation = enable_augmentation
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """应用变换"""
        if self.enable_augmentation:
            sample = self._apply_augmentations(sample)
        
        sample = self._normalize_data(sample)
        return sample
    
    def _apply_augmentations(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """应用数据增强"""
        # 随机翻转
        if random.random() < 0.5:
            sample = self._random_flip(sample)
        
        # 随机旋转
        if random.random() < 0.3:
            sample = self._random_rotation(sample)
        
        return sample
    
    def _random_flip(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """随机翻转"""
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        
        for modality, data in sample["multimodal_data"].items():
            if flip_h:
                data = torch.flip(data, dims=[2])  # 水平翻转
            if flip_v:
                data = torch.flip(data, dims=[1])  # 垂直翻转
            sample["multimodal_data"][modality] = data
        
        return sample
    
    def _random_rotation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """随机旋转"""
        angle = random.choice([90, 180, 270])
        k = angle // 90
        
        for modality, data in sample["multimodal_data"].items():
            # 旋转每个时间步
            T = data.shape[0]
            for t in range(T):
                sample["multimodal_data"][modality][t] = torch.rot90(data[t], k=k, dims=[0, 1])
        
        return sample
    
    def _normalize_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """数据标准化"""
        # 这里可以添加模态特定的标准化
        return sample