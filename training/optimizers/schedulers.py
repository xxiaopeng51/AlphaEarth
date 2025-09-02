"""
Learning rate scheduler utilities for AlphaEarth Foundations model.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any, Optional


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_annealing",
    epochs: int = 100,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler for the optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        epochs: Number of epochs
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler instance
    """
    if scheduler_type.lower() == "cosine_annealing":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == "cosine_annealing_warmup":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', epochs // 4),
            T_mult=kwargs.get('T_mult', 1),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', epochs // 3),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == "multistep":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [epochs // 3, 2 * epochs // 3]),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type.lower() == "exponential":
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type.lower() == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 1e-6)
        )
    elif scheduler_type.lower() == "linear":
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get('start_factor', 1.0),
            end_factor=kwargs.get('end_factor', 0.0),
            total_iters=epochs
        )
    elif scheduler_type.lower() == "polynomial":
        scheduler = lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=epochs,
            power=kwargs.get('power', 1.0)
        )
    elif scheduler_type.lower() == "one_cycle":
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=epochs,
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy=kwargs.get('anneal_strategy', 'cos')
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def get_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_annealing",
    epochs: int = 100,
    warmup_epochs: int = 10,
    warmup_lr: float = 1e-6,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        epochs: Number of epochs
        warmup_epochs: Number of warmup epochs
        warmup_lr: Warmup learning rate
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler with warmup
    """
    # Create base scheduler
    base_scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        epochs=epochs - warmup_epochs,
        **kwargs
    )
    
    # Create warmup scheduler
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_lr / optimizer.param_groups[0]['lr'],
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    # Combine schedulers
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_epochs]
    )
    
    return scheduler


def get_scheduler_with_restart(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine_annealing",
    epochs: int = 100,
    restart_epochs: int = 25,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler with restarts.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler
        epochs: Number of epochs
        restart_epochs: Number of epochs between restarts
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler with restarts
    """
    if scheduler_type.lower() == "cosine_annealing":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_epochs,
            T_mult=kwargs.get('T_mult', 1),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type.lower() == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=restart_epochs,
            gamma=kwargs.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unsupported scheduler type for restart: {scheduler_type}")
    
    return scheduler


def get_scheduler_with_plateau(
    optimizer: torch.optim.Optimizer,
    mode: str = "min",
    factor: float = 0.5,
    patience: int = 10,
    threshold: float = 1e-4,
    min_lr: float = 1e-6,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler with plateau detection.
    
    Args:
        optimizer: PyTorch optimizer
        mode: Mode for plateau detection ('min' or 'max')
        factor: Factor to reduce learning rate
        patience: Number of epochs to wait before reducing
        threshold: Threshold for improvement
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler with plateau detection
    """
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        min_lr=min_lr,
        **kwargs
    )
    
    return scheduler


def get_scheduler_with_custom(
    optimizer: torch.optim.Optimizer,
    lr_lambda: callable,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler with custom lambda function.
    
    Args:
        optimizer: PyTorch optimizer
        lr_lambda: Lambda function for learning rate
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler with custom lambda
    """
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
        **kwargs
    )
    
    return scheduler


def get_scheduler_with_cycle(
    optimizer: torch.optim.Optimizer,
    max_lr: float = 1e-3,
    total_steps: int = 1000,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler with cycle.
    
    Args:
        optimizer: PyTorch optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of steps
        pct_start: Percentage of steps for warmup
        anneal_strategy: Annealing strategy ('cos' or 'linear')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler with cycle
    """
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        **kwargs
    )
    
    return scheduler


def get_scheduler_with_polynomial(
    optimizer: torch.optim.Optimizer,
    total_iters: int = 1000,
    power: float = 1.0,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler with polynomial decay.
    
    Args:
        optimizer: PyTorch optimizer
        total_iters: Total number of iterations
        power: Power of polynomial
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler with polynomial decay
    """
    scheduler = lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=total_iters,
        power=power,
        **kwargs
    )
    
    return scheduler


def get_scheduler_with_linear(
    optimizer: torch.optim.Optimizer,
    total_iters: int = 1000,
    start_factor: float = 1.0,
    end_factor: float = 0.0,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler with linear decay.
    
    Args:
        optimizer: PyTorch optimizer
        total_iters: Total number of iterations
        start_factor: Start factor for learning rate
        end_factor: End factor for learning rate
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler with linear decay
    """
    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=end_factor,
        total_iters=total_iters,
        **kwargs
    )
    
    return scheduler