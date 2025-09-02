"""
Main training script for AlphaEarth Enhanced
Supports pretraining, fine-tuning, and evaluation
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path

from models import AlphaEarthEnhanced
from models.fusion import ContrastiveLearning


class Trainer:
    """Main trainer class for AlphaEarth Enhanced"""
    
    def __init__(self, config_path: str, resume: str = None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device and distributed training
        self.setup_distributed()
        
        # Initialize model
        self.model = self.build_model()
        
        # Initialize optimizers and schedulers
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Initialize loss functions
        self.criterion = self.build_criterion()
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config['training']['use_amp'] else None
        
        # Initialize logging
        self.setup_logging()
        
        # Resume from checkpoint if specified
        if resume:
            self.load_checkpoint(resume)
    
    def setup_distributed(self):
        """Setup distributed training environment"""
        if self.config['training']['distributed']:
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            self.device = torch.device(f'cuda:{self.rank}')
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_model(self):
        """Build and initialize model"""
        model = AlphaEarthEnhanced(**self.config['model'])
        
        # Move to device
        model = model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.config['training']['distributed']:
            model = DDP(model, device_ids=[self.rank])
        
        return model
    
    def build_optimizer(self):
        """Build optimizer"""
        opt_config = self.config['training']
        
        # Separate parameters for different learning rates
        params = [
            {'params': self.model.parameters(), 'lr': opt_config['learning_rate']}
        ]
        
        if opt_config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                params,
                lr=opt_config['learning_rate'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(
                params,
                lr=opt_config['learning_rate'],
                betas=opt_config['betas']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
        
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_config = self.config['training']
        
        if scheduler_config['scheduler'] == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['epochs'],
                eta_min=scheduler_config['min_lr']
            )
        elif scheduler_config['scheduler'] == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def build_criterion(self):
        """Build loss functions"""
        criterion = {}
        
        # MAE loss
        if self.config['model']['use_mae']:
            criterion['mae'] = nn.MSELoss()
        
        # Contrastive loss
        if self.config['model']['use_contrastive']:
            criterion['contrastive'] = ContrastiveLearning(
                embed_dim=self.config['model']['embed_dim'],
                temperature=self.config['model']['temperature']
            )
        
        # Task-specific losses
        criterion['classification'] = nn.CrossEntropyLoss()
        criterion['segmentation'] = nn.CrossEntropyLoss()
        
        return criterion
    
    def setup_logging(self):
        """Setup logging with wandb and tensorboard"""
        if self.rank == 0:  # Only log from main process
            if self.config['logging']['use_wandb']:
                wandb.init(
                    project=self.config['logging']['wandb_project'],
                    entity=self.config['logging']['wandb_entity'],
                    config=self.config
                )
            
            if self.config['logging']['use_tensorboard']:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.config['logging']['log_dir'])
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(self.rank != 0))
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self.move_to_device(batch)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    loss = self.compute_loss(batch)
            else:
                loss = self.compute_loss(batch)
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # Log to wandb
            if self.rank == 0 and batch_idx % 100 == 0:
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/epoch': epoch,
                        'train/step': epoch * num_batches + batch_idx
                    })
        
        return total_loss / num_batches
    
    def compute_loss(self, batch):
        """Compute total loss for a batch"""
        total_loss = 0
        loss_weights = self.config['training']['loss_weights']
        
        # MAE loss
        if self.config['model']['use_mae'] and 'optical' in batch:
            mae_loss, pred, mask = self.model.module.forward_mae(
                batch['optical'],
                mask_ratio=self.config['model']['mask_ratio']
            ) if hasattr(self.model, 'module') else self.model.forward_mae(
                batch['optical'],
                mask_ratio=self.config['model']['mask_ratio']
            )
            total_loss += loss_weights['mae'] * mae_loss
        
        # Contrastive loss
        if self.config['model']['use_contrastive'] and 'text' in batch:
            # Get features
            image_features = self.model(optical=batch.get('optical'))
            text_features = self.model(text=batch.get('text'))
            
            # Compute contrastive loss
            contrastive_loss, metrics = self.criterion['contrastive'](
                image_features.mean(dim=1),  # Pool sequence dimension
                text_features.mean(dim=1),
                locations=batch.get('coordinates'),
                timestamps=batch.get('timestamps')
            )
            total_loss += loss_weights['contrastive'] * contrastive_loss
        
        # Task-specific losses
        if 'task' in batch:
            task = batch['task']
            if task == 'classification':
                outputs = self.model(
                    optical=batch.get('optical'),
                    sar=batch.get('sar'),
                    task='classification'
                )
                task_loss = self.criterion['classification'](outputs, batch['labels'])
                total_loss += task_loss
        
        return total_loss
    
    def validate(self, dataloader, epoch):
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", disable=(self.rank != 0)):
                batch = self.move_to_device(batch)
                
                if self.scaler:
                    with autocast():
                        loss = self.compute_loss(batch)
                else:
                    loss = self.compute_loss(batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        # Log validation metrics
        if self.rank == 0:
            print(f"Validation Loss: {avg_loss:.4f}")
            
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'val/loss': avg_loss,
                    'val/epoch': epoch
                })
        
        return avg_loss
    
    def move_to_device(self, batch):
        """Move batch to device"""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_batch[key] = self.move_to_device(value)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        if self.rank == 0:
            checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_loss': val_loss,
                'config': self.config
            }
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            
            # Save best model
            if not hasattr(self, 'best_val_loss') or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, best_path)
                print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch']
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        start_epoch = 0
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(start_epoch, num_epochs):
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            if val_dataloader and epoch % self.config['evaluation']['eval_frequency'] == 0:
                val_loss = self.validate(val_dataloader, epoch)
            else:
                val_loss = train_loss
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            if epoch % self.config['logging']['save_frequency'] == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # Log epoch summary
            if self.rank == 0:
                print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Final checkpoint
        self.save_checkpoint(num_epochs, val_loss)
        
        if self.rank == 0:
            print("Training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train AlphaEarth Enhanced')
    parser.add_argument('--config', type=str, default='configs/pretrain.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config, args.resume)
    
    # Create data loaders (placeholder - implement actual data loading)
    train_dataloader = None  # TODO: Implement data loader
    val_dataloader = None    # TODO: Implement data loader
    
    # Start training
    trainer.train(train_dataloader, val_dataloader)


if __name__ == '__main__':
    main()