"""
Data Loaders

多模态数据加载器，支持复杂的批处理和对齐策略。
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict

from .multimodal_dataset import MultiModalEarthDataset


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    多模态数据的自定义collate函数
    
    处理不同模态数据的时间长度不一致问题，
    支持动态padding和mask生成。
    """
    batch_size = len(batch)
    
    # 收集所有模态名称
    all_modalities = set()
    for sample in batch:
        all_modalities.update(sample["multimodal_data"].keys())
    
    # 为每个模态找到最大时间长度
    max_time_lengths = {}
    for modality in all_modalities:
        max_len = 0
        for sample in batch:
            if modality in sample["multimodal_data"]:
                max_len = max(max_len, sample["multimodal_data"][modality].shape[0])
        max_time_lengths[modality] = max_len
    
    # 准备输出字典
    collated = {
        "multimodal_data": {},
        "timestamps": {},
        "attention_masks": {},
        "coordinates": [],
        "valid_periods": [],
        "text_descriptions": [],
        "regions": [],
        "sample_ids": []
    }
    
    # 处理每个模态
    for modality in all_modalities:
        modality_data = []
        modality_timestamps = []
        modality_masks = []
        
        max_len = max_time_lengths[modality]
        
        for sample in batch:
            if modality in sample["multimodal_data"]:
                data = sample["multimodal_data"][modality]  # (T, H, W, C)
                timestamps = sample["timestamps"]  # (T,)
                
                T, H, W, C = data.shape
                
                # Padding到最大长度
                if T < max_len:
                    # 数据padding (重复最后一帧)
                    last_frame = data[-1:].expand(max_len - T, -1, -1, -1)
                    padded_data = torch.cat([data, last_frame], dim=0)
                    
                    # 时间戳padding (重复最后一个时间戳)
                    last_timestamp = timestamps[-1:].expand(max_len - T)
                    padded_timestamps = torch.cat([timestamps, last_timestamp], dim=0)
                    
                    # 创建mask (1表示真实数据，0表示padding)
                    mask = torch.cat([
                        torch.ones(T),
                        torch.zeros(max_len - T)
                    ])
                else:
                    padded_data = data[:max_len]
                    padded_timestamps = timestamps[:max_len]
                    mask = torch.ones(max_len)
                
                modality_data.append(padded_data)
                modality_timestamps.append(padded_timestamps)
                modality_masks.append(mask)
            else:
                # 如果某个样本缺少该模态，用零填充
                # 获取参考样本的形状信息
                ref_sample = next(s for s in batch if modality in s["multimodal_data"])
                ref_shape = ref_sample["multimodal_data"][modality].shape
                H, W, C = ref_shape[1:]
                
                zero_data = torch.zeros(max_len, H, W, C)
                zero_timestamps = torch.zeros(max_len)
                zero_mask = torch.zeros(max_len)
                
                modality_data.append(zero_data)
                modality_timestamps.append(zero_timestamps)
                modality_masks.append(zero_mask)
        
        # 堆叠为batch
        collated["multimodal_data"][modality] = torch.stack(modality_data)  # (B, T, H, W, C)
        collated["timestamps"][modality] = torch.stack(modality_timestamps)  # (B, T)
        collated["attention_masks"][modality] = torch.stack(modality_masks)  # (B, T)
    
    # 处理其他字段
    for sample in batch:
        collated["coordinates"].append(sample["coordinates"])
        collated["valid_periods"].append(sample["valid_period"])
        collated["text_descriptions"].append(sample["text_description"])
        collated["regions"].append(sample["region"])
        collated["sample_ids"].append(sample["sample_id"])
    
    # 转换为tensor
    collated["coordinates"] = torch.stack(collated["coordinates"])
    collated["valid_periods"] = torch.stack(collated["valid_periods"])
    
    # 文本描述需要tokenization (这里简化处理)
    collated["text_descriptions"] = collated["text_descriptions"]  # 保持字符串列表
    
    return collated


def create_multimodal_dataloader(
    data_config: Dict[str, Any],
    split: str = "train",
    batch_size: int = 4,
    num_workers: int = 4,
    num_samples: int = 1000,
    patch_size: int = 224,
    time_window: int = 16,
    shuffle: bool = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    synthetic_mode: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    创建多模态数据加载器
    
    Args:
        data_config: 数据配置字典
        split: 数据分割
        batch_size: 批大小
        num_workers: 工作进程数
        num_samples: 样本数量
        patch_size: 空间patch大小
        time_window: 时间窗口
        shuffle: 是否打乱 (默认train=True, val/test=False)
        pin_memory: 是否pin memory
        drop_last: 是否丢弃最后不完整的batch
        synthetic_mode: 是否使用合成数据
        **dataset_kwargs: 传递给数据集的额外参数
    """
    if shuffle is None:
        shuffle = (split == "train")
    
    # 创建数据集
    dataset = MultiModalEarthDataset(
        data_config=data_config,
        split=split,
        num_samples=num_samples,
        patch_size=patch_size,
        time_window=time_window,
        synthetic_mode=synthetic_mode,
        **dataset_kwargs
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )
    
    return dataloader


class BalancedMultiModalSampler:
    """平衡的多模态采样器，确保不同区域和模态的平衡"""
    
    def __init__(
        self,
        dataset: MultiModalEarthDataset,
        samples_per_region: int = 100,
        modality_balance: Dict[str, float] = None
    ):
        self.dataset = dataset
        self.samples_per_region = samples_per_region
        self.modality_balance = modality_balance or {}
        
        # 按区域分组样本
        self.region_indices = defaultdict(list)
        for idx, sample_info in enumerate(dataset.sample_indices):
            region = sample_info["region"]
            self.region_indices[region].append(idx)
    
    def get_balanced_indices(self) -> List[int]:
        """获取平衡的样本索引"""
        balanced_indices = []
        
        for region, indices in self.region_indices.items():
            # 每个区域采样指定数量
            if len(indices) >= self.samples_per_region:
                sampled = np.random.choice(indices, self.samples_per_region, replace=False)
            else:
                sampled = np.random.choice(indices, self.samples_per_region, replace=True)
            
            balanced_indices.extend(sampled)
        
        return balanced_indices


def create_distributed_dataloader(
    data_config: Dict[str, Any],
    split: str = "train",
    batch_size: int = 4,
    world_size: int = 1,
    rank: int = 0,
    **kwargs
) -> DataLoader:
    """创建分布式训练的数据加载器"""
    from torch.utils.data.distributed import DistributedSampler
    
    dataset = MultiModalEarthDataset(
        data_config=data_config,
        split=split,
        **kwargs
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train")
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=multimodal_collate_fn,
        num_workers=kwargs.get("num_workers", 4),
        pin_memory=kwargs.get("pin_memory", True)
    )
    
    return dataloader