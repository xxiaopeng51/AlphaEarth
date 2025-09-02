"""训练回调函数"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path


class ModelCheckpoint:
    """模型检查点回调"""
    
    def __init__(
        self,
        save_dir: str,
        save_interval: int = 1000,
        save_best: bool = True,
        monitor: str = "val_loss"
    ):
        self.save_dir = Path(save_dir)
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor = monitor
        self.best_value = float('inf')
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, data: Dict[str, Any]):
        """Epoch结束时的回调"""
        epoch = data["epoch"]
        val_metrics = data["val_metrics"]
        
        # 保存最佳模型
        if self.save_best and self.monitor in val_metrics:
            current_value = val_metrics[self.monitor]
            if current_value < self.best_value:
                self.best_value = current_value
                # 这里需要访问trainer来保存模型
                # 简化实现


class EarlyStopping:
    """早停回调"""
    
    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_loss",
        min_delta: float = 1e-4
    ):
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.wait = 0
    
    def should_stop(self, current_value: float) -> bool:
        """检查是否应该早停"""
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
        
        return self.wait >= self.patience


class LearningRateScheduler:
    """学习率调度回调"""
    
    def __init__(self, scheduler, warmup_steps: int = 0):
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def on_batch_end(self, data: Dict[str, Any]):
        """Batch结束时的回调"""
        self.current_step += 1
        
        if self.current_step > self.warmup_steps:
            self.scheduler.step()