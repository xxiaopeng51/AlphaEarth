"""
Logging utilities for AlphaEarth Foundations model.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
import time


def setup_logging(
    output_dir: str,
    experiment_name: str,
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True
):
    """
    Setup logging configuration.
    
    Args:
        output_dir: Output directory for logs
        experiment_name: Name of the experiment
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = output_path / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Setup logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_model_info(logger: logging.Logger, model: torch.nn.Module):
    """
    Log model information.
    
    Args:
        logger: Logger instance
        model: PyTorch model
    """
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


def log_training_info(logger: logging.Logger, config: dict):
    """
    Log training configuration information.
    
    Args:
        logger: Logger instance
        config: Training configuration
    """
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_epoch_results(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    lr: Optional[float] = None,
    additional_metrics: Optional[dict] = None
):
    """
    Log epoch results.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss (optional)
        lr: Learning rate (optional)
        additional_metrics: Additional metrics to log (optional)
    """
    log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f}"
    
    if val_loss is not None:
        log_msg += f", Val Loss: {val_loss:.4f}"
    
    if lr is not None:
        log_msg += f", LR: {lr:.6f}"
    
    logger.info(log_msg)
    
    if additional_metrics:
        for metric_name, metric_value in additional_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")


def log_evaluation_results(logger: logging.Logger, results: dict):
    """
    Log evaluation results.
    
    Args:
        logger: Logger instance
        results: Evaluation results dictionary
    """
    logger.info("Evaluation Results:")
    for metric_name, metric_value in results.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")


def log_gpu_memory(logger: logging.Logger):
    """
    Log GPU memory usage.
    
    Args:
        logger: Logger instance
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.2f}GB total")
    else:
        logger.info("CUDA not available")


def log_data_info(logger: logging.Logger, dataset_info: dict):
    """
    Log dataset information.
    
    Args:
        logger: Logger instance
        dataset_info: Dataset information dictionary
    """
    logger.info("Dataset Information:")
    for key, value in dataset_info.items():
        logger.info(f"  {key}: {value}")


def log_checkpoint_info(logger: logging.Logger, checkpoint_path: str, epoch: int, loss: float):
    """
    Log checkpoint information.
    
    Args:
        logger: Logger instance
        checkpoint_path: Path to checkpoint
        epoch: Epoch number
        loss: Loss value
    """
    logger.info(f"Saved checkpoint: {checkpoint_path} (Epoch {epoch}, Loss: {loss:.4f})")


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log error information.
    
    Args:
        logger: Logger instance
        error: Exception object
        context: Additional context information
    """
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")
    
    logger.error(f"Error type: {type(error).__name__}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")


def log_warning(logger: logging.Logger, message: str, context: str = ""):
    """
    Log warning message.
    
    Args:
        logger: Logger instance
        message: Warning message
        context: Additional context information
    """
    if context:
        logger.warning(f"Warning in {context}: {message}")
    else:
        logger.warning(f"Warning: {message}")


def log_info(logger: logging.Logger, message: str, context: str = ""):
    """
    Log info message.
    
    Args:
        logger: Logger instance
        message: Info message
        context: Additional context information
    """
    if context:
        logger.info(f"Info in {context}: {message}")
    else:
        logger.info(f"Info: {message}")


def log_debug(logger: logging.Logger, message: str, context: str = ""):
    """
    Log debug message.
    
    Args:
        logger: Logger instance
        message: Debug message
        context: Additional context information
    """
    if context:
        logger.debug(f"Debug in {context}: {message}")
    else:
        logger.debug(f"Debug: {message}")


# Import torch here to avoid circular imports
import torch