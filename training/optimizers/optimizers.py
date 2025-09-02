"""
Optimizer utilities for AlphaEarth Foundations model.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer for the model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Get model parameters
    params = model.parameters()
    
    # Create optimizer based on type
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    elif optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            alpha=kwargs.get('alpha', 0.99)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def get_optimizer_with_different_lr(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    base_lr: float = 1e-4,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer with different learning rates for different parts of the model.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        base_lr: Base learning rate
        backbone_lr: Learning rate for backbone
        head_lr: Learning rate for task heads
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Separate parameters by module type
    backbone_params = []
    head_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name or 'encoder' in name:
            backbone_params.append(param)
        elif 'head' in name or 'classifier' in name:
            head_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay},
        {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay}
    ]
    
    # Create optimizer
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            param_groups,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(
            param_groups,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            param_groups,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    return optimizer


def get_optimizer_with_warmup(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    warmup_lr: float = 1e-6,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer with warmup learning rate.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        warmup_steps: Number of warmup steps
        warmup_lr: Warmup learning rate
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=warmup_lr,  # Start with warmup learning rate
        weight_decay=weight_decay,
        **kwargs
    )
    
    # Add warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, step / warmup_steps)
    )
    
    return optimizer, warmup_scheduler


def get_optimizer_with_gradient_clipping(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Get optimizer with gradient clipping.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        max_grad_norm: Maximum gradient norm
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
    
    # Add gradient clipping
    def clip_gradients():
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    optimizer.clip_gradients = clip_gradients
    
    return optimizer


def get_optimizer_with_mixed_precision(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs
) -> tuple:
    """
    Get optimizer with mixed precision support.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Tuple of (optimizer, scaler)
    """
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
    
    # Create scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    return optimizer, scaler


def get_optimizer_with_ema(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    ema_decay: float = 0.999,
    **kwargs
) -> tuple:
    """
    Get optimizer with exponential moving average.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        ema_decay: EMA decay rate
        **kwargs: Additional optimizer arguments
        
    Returns:
        Tuple of (optimizer, ema_model)
    """
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
    
    # Create EMA model
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay))
    
    return optimizer, ema_model


def get_optimizer_with_swa(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    swa_lr: float = 1e-5,
    **kwargs
) -> tuple:
    """
    Get optimizer with stochastic weight averaging.
    
    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer
        lr: Learning rate
        weight_decay: Weight decay
        swa_lr: SWA learning rate
        **kwargs: Additional optimizer arguments
        
    Returns:
        Tuple of (optimizer, swa_model, swa_scheduler)
    """
    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        **kwargs
    )
    
    # Create SWA model
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    
    # Create SWA scheduler
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)
    
    return optimizer, swa_model, swa_scheduler