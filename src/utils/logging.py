"""Logging utilities for the ASR system."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import wandb
from omegaconf import DictConfig


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level.
        log_file: Optional log file path.
        format_string: Custom format string.
        
    Returns:
        Configured logger.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def setup_wandb(config: DictConfig) -> Optional[wandb.Run]:
    """
    Setup Weights & Biases logging.
    
    Args:
        config: Configuration object.
        
    Returns:
        W&B run object or None if disabled.
    """
    if not config.logging.wandb.enabled:
        return None
    
    # Initialize W&B
    run = wandb.init(
        project=config.logging.wandb.project,
        entity=config.logging.wandb.entity,
        config=config,
        name=f"{config.project.name}_{wandb.util.generate_id()}",
        tags=["asr", "accented-speech"],
        notes="Accented Speech Recognition Research"
    )
    
    return run


def log_model_info(logger: logging.Logger, model: torch.nn.Module) -> None:
    """
    Log model information.
    
    Args:
        logger: Logger instance.
        model: PyTorch model.
    """
    from src.utils.device import count_parameters, get_model_size
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    logger.info(f"Model size: {get_model_size(model)}")


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    step: int,
    loss: float,
    metrics: Optional[dict] = None
) -> None:
    """
    Log training progress.
    
    Args:
        logger: Logger instance.
        epoch: Current epoch.
        step: Current step.
        loss: Current loss.
        metrics: Optional metrics dictionary.
    """
    log_msg = f"Epoch {epoch}, Step {step}, Loss: {loss:.4f}"
    
    if metrics:
        metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        log_msg += f", {', '.join(metric_strs)}"
    
    logger.info(log_msg)


def log_evaluation_results(
    logger: logging.Logger,
    results: dict,
    dataset_split: str = "test"
) -> None:
    """
    Log evaluation results.
    
    Args:
        logger: Logger instance.
        results: Evaluation results dictionary.
        dataset_split: Dataset split name.
    """
    logger.info(f"=== {dataset_split.upper()} RESULTS ===")
    
    for metric, value in results.items():
        if isinstance(value, dict):
            logger.info(f"{metric}:")
            for sub_metric, sub_value in value.items():
                logger.info(f"  {sub_metric}: {sub_value:.4f}")
        else:
            logger.info(f"{metric}: {value:.4f}")


class PrivacyAwareLogger:
    """Logger that respects privacy settings."""
    
    def __init__(self, logger: logging.Logger, anonymize: bool = True):
        self.logger = logger
        self.anonymize = anonymize
    
    def info(self, message: str) -> None:
        """Log info message with privacy protection."""
        if self.anonymize:
            message = self._anonymize_message(message)
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message with privacy protection."""
        if self.anonymize:
            message = self._anonymize_message(message)
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message with privacy protection."""
        if self.anonymize:
            message = self._anonymize_message(message)
        self.logger.error(message)
    
    def _anonymize_message(self, message: str) -> str:
        """Anonymize potentially sensitive information in log messages."""
        # Remove file paths that might contain personal information
        import re
        
        # Replace file paths with generic placeholders
        message = re.sub(r'/[^/]+/[^/]+/[^/]+/', '[PATH]/', message)
        
        # Replace potential speaker IDs
        message = re.sub(r'spk_\d+', '[SPEAKER]', message)
        
        return message
