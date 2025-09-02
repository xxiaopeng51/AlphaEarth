"""
Enhanced Earth Foundation Model Trainer

支持大规模分布式训练的训练器，集成了多种优化策略：
1. 混合精度训练
2. 梯度累积
3. 学习率调度
4. 模型检查点
5. 分布式训练支持
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import wandb
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from omegaconf import DictConfig, OmegaConf

from ..models.enhanced_earth_model import EnhancedEarthFoundationModel
from ..data.data_loaders import create_multimodal_dataloader, create_distributed_dataloader
from .losses import CombinedLoss
from .metrics import EvaluationMetrics
from .callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_size: str = "base"
    embed_dim: int = 64
    
    # 训练配置
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # 优化器配置
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "linear", "step"
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # 混合精度
    use_mixed_precision: bool = True
    
    # 分布式训练
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # 数据配置
    num_workers: int = 8
    pin_memory: bool = True
    
    # 检查点和日志
    save_dir: str = "./checkpoints"
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    
    # 实验跟踪
    use_wandb: bool = True
    project_name: str = "enhanced-earth-foundation"
    experiment_name: str = "baseline"
    
    # 其他
    seed: int = 42
    resume_from: Optional[str] = None


class EnhancedEarthTrainer:
    """
    Enhanced Earth Foundation Model训练器
    
    支持大规模多模态模型的高效训练
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[EnhancedEarthFoundationModel] = None,
        data_config: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 初始化分布式训练
        if config.distributed:
            self._init_distributed()
        
        # 创建模型
        if model is None:
            model = EnhancedEarthFoundationModel(
                model_size=config.model_size,
                embed_dim=config.embed_dim
            )
        
        self.model = model.to(self.device)
        
        # 分布式包装
        if config.distributed:
            self.model = DDP(self.model, device_ids=[config.rank])
        
        # 创建数据加载器
        self.data_config = data_config or self._get_default_data_config()
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 创建优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 损失函数和评估指标
        self.criterion = CombinedLoss()
        self.metrics = EvaluationMetrics()
        
        # 混合精度
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # 回调函数
        self.callbacks = self._create_callbacks()
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 日志记录
        if config.use_wandb and (not config.distributed or config.rank == 0):
            self._init_wandb()
        
        # 恢复训练 (如果指定)
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _init_distributed(self):
        """初始化分布式训练"""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        torch.cuda.set_device(self.config.rank)
        self.device = torch.device(f"cuda:{self.config.rank}")
    
    def _get_default_data_config(self) -> Dict[str, Any]:
        """获取默认数据配置"""
        return {
            "optical": {"source": "sentinel2", "channels": 13, "resolution": 10},
            "sar": {"source": "sentinel1", "channels": 4, "resolution": 10},
            "environmental": {"source": "era5", "channels": 8, "resolution": 1000}
        }
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        if self.config.distributed:
            train_loader = create_distributed_dataloader(
                data_config=self.data_config,
                split="train",
                batch_size=self.config.batch_size,
                world_size=self.config.world_size,
                rank=self.config.rank,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                synthetic_mode=True
            )
            
            val_loader = create_distributed_dataloader(
                data_config=self.data_config,
                split="val",
                batch_size=self.config.batch_size,
                world_size=self.config.world_size,
                rank=self.config.rank,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                synthetic_mode=True
            )
        else:
            train_loader = create_multimodal_dataloader(
                data_config=self.data_config,
                split="train",
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                synthetic_mode=True
            )
            
            val_loader = create_multimodal_dataloader(
                data_config=self.data_config,
                split="val",
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                synthetic_mode=True
            )
        
        return train_loader, val_loader
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=total_steps - warmup_steps
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_steps // 3,
                gamma=0.1
            )
        else:
            return None
    
    def _create_callbacks(self) -> Dict[str, Any]:
        """创建回调函数"""
        callbacks = {}
        
        # 模型检查点
        callbacks["checkpoint"] = ModelCheckpoint(
            save_dir=self.config.save_dir,
            save_interval=self.config.save_interval,
            save_best=True,
            monitor="val_loss"
        )
        
        # 早停
        callbacks["early_stopping"] = EarlyStopping(
            patience=10,
            monitor="val_loss",
            min_delta=1e-4
        )
        
        # 学习率调度
        if self.scheduler:
            callbacks["lr_scheduler"] = LearningRateScheduler(
                scheduler=self.scheduler,
                warmup_steps=len(self.train_loader) * self.config.warmup_epochs
            )
        
        return callbacks
    
    def _init_wandb(self):
        """初始化Weights & Biases"""
        wandb.init(
            project=self.config.project_name,
            name=self.config.experiment_name,
            config=asdict(self.config)
        )
        
        # 记录模型信息
        wandb.watch(self.model, log="all", log_freq=1000)
    
    def train(self):
        """主训练循环"""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Model size: {self.model.get_model_size()}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            if epoch % (self.config.eval_interval // len(self.train_loader)) == 0:
                val_metrics = self.validate()
                
                # 更新最佳模型
                val_loss = val_metrics.get("val_loss", float('inf'))
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_best_model()
                
                # 记录指标
                self._log_metrics(train_metrics, val_metrics, epoch)
                
                # 回调函数
                self._call_callbacks("on_epoch_end", {
                    "epoch": epoch,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics
                })
                
                # 早停检查
                if self.callbacks["early_stopping"].should_stop(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # 保存检查点
            if epoch % (self.config.save_interval // len(self.train_loader)) == 0:
                self.save_checkpoint(f"epoch_{epoch}")
        
        print("Training completed!")
        
        # 最终验证
        final_metrics = self.validate()
        return final_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {}
        epoch_metrics = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 将数据移到设备
            batch = self._move_to_device(batch)
            
            # 前向传播和损失计算
            with autocast(enabled=self.config.use_mixed_precision):
                outputs = self.model(
                    multimodal_data=batch["multimodal_data"],
                    timestamps=batch["timestamps"].get("optical", batch["timestamps"][list(batch["timestamps"].keys())[0]]),
                    coordinates=batch["coordinates"],
                    valid_periods=batch["valid_periods"],
                    attention_mask=batch["attention_masks"].get("optical", None)
                )
                
                # 计算损失
                losses = self.criterion(outputs, batch)
                loss = losses["total_loss"]
                
                # 梯度累积
                loss = loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 优化器步骤
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 学习率调度
                if self.scheduler and self.global_step > self.config.warmup_epochs * len(self.train_loader):
                    self.scheduler.step()
            
            # 累积损失和指标
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())
            
            # 日志记录
            if batch_idx % self.config.log_interval == 0:
                self._log_batch_metrics(batch_idx, num_batches, losses)
        
        # 计算epoch平均值
        for key, values in epoch_losses.items():
            epoch_metrics[f"train_{key}"] = np.mean(values)
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        val_losses = {}
        val_metrics = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = self._move_to_device(batch)
                
                with autocast(enabled=self.config.use_mixed_precision):
                    outputs = self.model(
                        multimodal_data=batch["multimodal_data"],
                        timestamps=batch["timestamps"].get("optical", batch["timestamps"][list(batch["timestamps"].keys())[0]]),
                        coordinates=batch["coordinates"],
                        valid_periods=batch["valid_periods"],
                        attention_mask=batch["attention_masks"].get("optical", None)
                    )
                    
                    losses = self.criterion(outputs, batch)
                
                # 累积损失
                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item())
                
                # 计算评估指标
                batch_metrics = self.metrics.compute_metrics(outputs, batch)
                for key, value in batch_metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    val_metrics[key].append(value)
        
        # 计算平均值
        final_metrics = {}
        for key, values in val_losses.items():
            final_metrics[f"val_{key}"] = np.mean(values)
        
        for key, values in val_metrics.items():
            final_metrics[f"val_{key}"] = np.mean(values)
        
        return final_metrics
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将batch数据移到设备"""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                device_batch[key] = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _log_batch_metrics(self, batch_idx: int, num_batches: int, losses: Dict[str, torch.Tensor]):
        """记录batch指标"""
        if not self.config.distributed or self.config.rank == 0:
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                  f"Loss: {losses['total_loss'].item():.4f}, LR: {lr:.2e}")
            
            if self.config.use_wandb:
                log_dict = {
                    "batch_loss": losses["total_loss"].item(),
                    "learning_rate": lr,
                    "epoch": self.current_epoch,
                    "global_step": self.global_step
                }
                
                for key, value in losses.items():
                    if key != "total_loss":
                        log_dict[f"batch_{key}"] = value.item()
                
                wandb.log(log_dict, step=self.global_step)
    
    def _log_metrics(
        self, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        epoch: int
    ):
        """记录epoch指标"""
        if not self.config.distributed or self.config.rank == 0:
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics.get('train_total_loss', 0):.4f}")
            print(f"  Val Loss: {val_metrics.get('val_total_loss', 0):.4f}")
            
            if self.config.use_wandb:
                log_dict = {**train_metrics, **val_metrics, "epoch": epoch}
                wandb.log(log_dict, step=epoch)
    
    def _call_callbacks(self, event: str, data: Dict[str, Any]):
        """调用回调函数"""
        for callback in self.callbacks.values():
            if hasattr(callback, event):
                getattr(callback, event)(data)
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        if not self.config.distributed or self.config.rank == 0:
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "config": asdict(self.config),
                "best_val_loss": self.best_val_loss
            }
            
            save_path = Path(self.config.save_dir) / f"{name}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def _save_best_model(self):
        """保存最佳模型"""
        self.save_checkpoint("best_model")
    
    def evaluate_on_downstream_tasks(self, task_configs: List[Dict[str, Any]]) -> Dict[str, float]:
        """在下游任务上评估模型"""
        # TODO: 实现下游任务评估
        # 例如：分类、分割、变化检测等
        pass
    
    def export_model(self, export_path: str, format: str = "pytorch"):
        """导出模型"""
        if format == "pytorch":
            # 保存完整模型
            torch.save(self.model, export_path)
        elif format == "onnx":
            # 导出ONNX格式
            dummy_input = self._create_dummy_input()
            torch.onnx.export(
                self.model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True
            )
        elif format == "torchscript":
            # 导出TorchScript
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(export_path)
        
        print(f"Model exported to {export_path} in {format} format")
    
    def _create_dummy_input(self) -> Dict[str, torch.Tensor]:
        """创建用于导出的虚拟输入"""
        dummy_batch = {
            "multimodal_data": {
                "optical": torch.randn(1, 16, 224, 224, 13),
                "sar": torch.randn(1, 16, 224, 224, 4),
                "environmental": torch.randn(1, 16, 224, 224, 8)
            },
            "timestamps": torch.randn(1, 16),
            "coordinates": torch.randn(1, 2),
            "valid_periods": torch.randn(1, 2)
        }
        return dummy_batch


def create_trainer_from_config(config_path: str) -> EnhancedEarthTrainer:
    """从配置文件创建训练器"""
    config_dict = OmegaConf.load(config_path)
    config = TrainingConfig(**config_dict)
    
    return EnhancedEarthTrainer(config)