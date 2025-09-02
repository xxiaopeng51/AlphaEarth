#!/usr/bin/env python3
"""
Training script for AlphaEarth Foundations model.

This script provides a comprehensive training pipeline for the multi-modal
foundation model with support for distributed training, mixed precision,
and various optimization strategies.
"""

import argparse
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import AlphaEarthFoundations
from data import MultiModalDataset, DataLoader, collate_fn
from training import Trainer, MultiModalLoss
from utils import setup_logging, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AlphaEarth Foundations model')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--metadata_file', type=str, required=True,
                       help='Path to metadata file')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine_annealing',
                       choices=['cosine_annealing', 'step', 'plateau'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                       help='Number of warmup epochs')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of processes for distributed training')
    
    # Mixed precision arguments
    parser.add_argument('--use_amp', action='store_true',
                       help='Enable automatic mixed precision')
    
    # Logging and checkpointing arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default='alphaearth_experiment',
                       help='Name of the experiment')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='alphaearth-foundations',
                       help='Weights & Biases project name')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation')
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def create_model(config: dict) -> AlphaEarthFoundations:
    """Create AlphaEarth Foundations model."""
    model_config = config.get('model', {})
    
    model = AlphaEarthFoundations(
        optical_config=model_config.get('optical', {}),
        radar_config=model_config.get('radar', {}),
        meteorological_config=model_config.get('meteorological', {}),
        text_config=model_config.get('text', {}),
        fusion_config=model_config.get('fusion', {}),
        stp_config=model_config.get('stp_module', {}),
        task_heads=model_config.get('heads', {}),
        **model_config.get('general', {})
    )
    
    return model


def create_datasets(args, config: dict):
    """Create training and validation datasets."""
    data_config = config.get('data', {})
    
    # Training dataset
    train_dataset = MultiModalDataset(
        data_root=args.data_root,
        metadata_file=args.metadata_file,
        modalities=data_config.get('modalities', ['optical', 'radar', 'meteorological', 'text']),
        transforms=data_config.get('transforms', {}),
        max_samples=data_config.get('max_train_samples')
    )
    
    # Validation dataset (if separate metadata file provided)
    val_dataset = None
    if data_config.get('val_metadata_file'):
        val_dataset = MultiModalDataset(
            data_root=args.data_root,
            metadata_file=data_config['val_metadata_file'],
            modalities=data_config.get('modalities', ['optical', 'radar', 'meteorological', 'text']),
            transforms=data_config.get('val_transforms', {}),
            max_samples=data_config.get('max_val_samples')
        )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, args, config: dict):
    """Create data loaders."""
    data_config = config.get('data', {})
    
    # Training data loader
    train_sampler = None
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    # Validation data loader
    val_loader = None
    if val_dataset is not None:
        val_sampler = None
        if args.distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False
        )
    
    return train_loader, val_loader


def train_worker(rank: int, world_size: int, args, config: dict):
    """Training worker for distributed training."""
    # Setup distributed training
    if args.distributed:
        setup_distributed(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    set_seed(args.seed + rank)
    
    # Setup logging
    if rank == 0:
        setup_logging(args.output_dir, args.experiment_name)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Wrap model with DDP if distributed
    if args.distributed:
        model = DDP(model, device_ids=[rank])
    
    # Create datasets and data loaders
    train_dataset, val_dataset = create_datasets(args, config)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, args, config)
    
    # Create trainer
    trainer_config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'gradient_clip_norm': args.gradient_clip_norm,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'warmup_epochs': args.warmup_epochs,
        'use_amp': args.use_amp,
        'use_wandb': args.use_wandb and rank == 0,
        'wandb_project': args.wandb_project,
        'experiment_name': args.experiment_name,
        'checkpoint_dir': os.path.join(args.output_dir, 'checkpoints'),
        'log_every': 100,
        'save_every': 10,
        'early_stopping_patience': 20
    }
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )
    
    # Load pretrained model if specified
    if args.pretrained_path:
        trainer.load_checkpoint(args.pretrained_path)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    if not args.eval_only:
        trainer.train()
    
    # Evaluate the model
    if rank == 0:
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    
    # Cleanup distributed training
    if args.distributed:
        cleanup_distributed()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Start training
    if args.distributed and args.world_size > 1:
        mp.spawn(
            train_worker,
            args=(args.world_size, args, config),
            nprocs=args.world_size,
            join=True
        )
    else:
        train_worker(0, 1, args, config)


if __name__ == '__main__':
    main()