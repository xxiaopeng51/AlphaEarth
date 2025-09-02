"""
Training framework for AlphaEarth Foundations model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import wandb
from tqdm import tqdm
import time

from ..models import AlphaEarthFoundations
from .losses import MultiModalLoss, ContrastiveLoss
from .optimizers import get_optimizer, get_scheduler


class Trainer:
    """
    Trainer class for AlphaEarth Foundations model.
    
    This class handles the training loop, validation, checkpointing,
    and logging for the multi-modal foundation model.
    """
    
    def __init__(
        self,
        model: AlphaEarthFoundations,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        config: Dict = None,
        device: str = "cuda",
        **kwargs
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        self.device = device
        
        # Training parameters
        self.epochs = self.config.get('epochs', 100)
        self.lr = self.config.get('lr', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.gradient_clip_norm = self.config.get('gradient_clip_norm', 1.0)
        
        # Loss functions
        self.criterion = MultiModalLoss(
            modalities=self.config.get('modalities', ['optical', 'radar', 'meteorological', 'text']),
            loss_weights=self.config.get('loss_weights', [1.0, 1.0, 1.0, 1.0])
        )
        
        # Optimizer and scheduler
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_type=self.config.get('optimizer', 'adamw'),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.config.get('optimizer_kwargs', {})
        )
        
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.config.get('scheduler', 'cosine_annealing'),
            epochs=self.epochs,
            **self.config.get('scheduler_kwargs', {})
        )
        
        # Mixed precision training
        self.use_amp = self.config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Logging and monitoring
        self.log_every = self.config.get('log_every', 100)
        self.save_every = self.config.get('save_every', 10)
        self.early_stopping_patience = self.config.get('early_stopping_patience', 20)
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup Weights & Biases if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'alphaearth-foundations'),
                name=self.config.get('experiment_name', f'experiment_{int(time.time())}'),
                config=self.config
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs, batch)
            else:
                outputs = self.model(**batch)
                loss = self.criterion(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
            
            # Log intermediate results
            if batch_idx % self.log_every == 0:
                self._log_training_step(batch_idx, loss.item(), outputs)
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = self.criterion(outputs, batch)
                else:
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Returns:
            Dictionary containing training history
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics.get('val_loss', 0.0))
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log epoch results
            self._log_epoch_results(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_metrics.get('val_loss', 0.0))
            
            # Early stopping
            if self._check_early_stopping(val_metrics.get('val_loss', 0.0)):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        self._save_checkpoint(self.current_epoch, self.best_val_loss, is_final=True)
        
        return self.training_history
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to the specified device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = self._move_batch_to_device(value)
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _log_training_step(self, batch_idx: int, loss: float, outputs: Dict):
        """Log training step information."""
        if self.config.get('use_wandb', False):
            wandb.log({
                'train_step_loss': loss,
                'epoch': self.current_epoch,
                'batch': batch_idx,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log epoch results."""
        self.logger.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Val Loss: {val_metrics.get('val_loss', 0.0):.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )
        
        if self.config.get('use_wandb', False):
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            log_dict.update(val_metrics)
            wandb.log(log_dict)
    
    def _save_checkpoint(self, epoch: int, val_loss: float, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        if is_final:
            checkpoint_path = checkpoint_dir / 'final_model.pth'
        else:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping criteria is met."""
        return self.early_stopping_counter >= self.early_stopping_patience
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.test_loader is None:
            self.logger.warning("No test loader provided for evaluation")
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.test_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Evaluation'):
                batch = self._move_batch_to_device(batch)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = self.criterion(outputs, batch)
                else:
                    outputs = self.model(**batch)
                    loss = self.criterion(outputs, batch)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        
        return {'test_loss': avg_loss}