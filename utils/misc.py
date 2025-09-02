"""
Miscellaneous utility functions.
"""

import torch
import random
import numpy as np
import os
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification (e.g., 'cuda', 'cpu', 'cuda:0')
        
    Returns:
        torch.device object
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: torch.nn.Module) -> str:
    """
    Get the size of a model in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size as a formatted string
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return f"{size_all_mb:.2f} MB"


def save_model_summary(model: torch.nn.Module, save_path: str):
    """
    Save a summary of the model architecture.
    
    Args:
        model: PyTorch model
        save_path: Path to save the summary
    """
    with open(save_path, 'w') as f:
        f.write(f"Model Summary\n")
        f.write(f"=============\n\n")
        f.write(f"Total parameters: {count_parameters(model):,}\n")
        f.write(f"Model size: {get_model_size(model)}\n\n")
        f.write(f"Architecture:\n")
        f.write(str(model))


def create_directory(path: str):
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(number: int) -> str:
    """
    Format a number with appropriate units (K, M, B).
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    if number < 1000:
        return str(number)
    elif number < 1000000:
        return f"{number/1000:.1f}K"
    elif number < 1000000000:
        return f"{number/1000000:.1f}M"
    else:
        return f"{number/1000000000:.1f}B"


def get_gpu_memory_usage() -> dict:
    """
    Get GPU memory usage information.
    
    Returns:
        Dictionary containing GPU memory information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        memory_info[f"gpu_{i}"] = {
            "allocated": torch.cuda.memory_allocated(i) / 1024**3,  # GB
            "cached": torch.cuda.memory_reserved(i) / 1024**3,      # GB
            "total": torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
        }
    
    return memory_info


def clear_gpu_memory():
    """
    Clear GPU memory cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from an optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def update_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """
    Update the learning rate of an optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_iters: int, warmup_factor: float = 1.0 / 3):
    """
    Create a learning rate scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_iters: Number of warmup iterations
        warmup_factor: Warmup factor
        
    Returns:
        Learning rate scheduler
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def cosine_annealing_lr_scheduler(optimizer: torch.optim.Optimizer, T_max: int, eta_min: float = 0):
    """
    Create a cosine annealing learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate
        
    Returns:
        Learning rate scheduler
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)


def exponential_lr_scheduler(optimizer: torch.optim.Optimizer, gamma: float):
    """
    Create an exponential learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        gamma: Decay factor
        
    Returns:
        Learning rate scheduler
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


def step_lr_scheduler(optimizer: torch.optim.Optimizer, step_size: int, gamma: float = 0.1):
    """
    Create a step learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        step_size: Step size
        gamma: Decay factor
        
    Returns:
        Learning rate scheduler
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)