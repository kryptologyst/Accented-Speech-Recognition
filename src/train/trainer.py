"""Training system for accented speech recognition."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.device import get_device, setup_environment
from src.utils.logging import setup_logging, setup_wandb, log_training_progress, log_evaluation_results
from src.metrics.asr_metrics import ASRMetrics

logger = logging.getLogger(__name__)


class ASRTrainer:
    """Trainer class for ASR models."""
    
    def __init__(self, config: DictConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        
        # Setup environment
        self.device = setup_environment(
            seed=config.get("seed", 42),
            device=config.get("device", "auto")
        )
        
        # Setup logging
        self.logger = setup_logging(
            level=config.logging.get("level", "INFO")
        )
        
        # Setup W&B if enabled
        self.wandb_run = setup_wandb(config)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.metrics = ASRMetrics()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        self.logger.info("ASR Trainer initialized")
    
    def setup_model(self, model: nn.Module) -> None:
        """
        Setup the model for training.
        
        Args:
            model: ASR model to train.
        """
        self.model = model.to(self.device)
        
        # Log model info
        if hasattr(model, 'get_model_info'):
            model_info = model.get_model_info()
            self.logger.info(f"Model info: {model_info}")
        
        self.logger.info(f"Model moved to device: {self.device}")
    
    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.optimizer.lr,
            weight_decay=self.config.training.optimizer.weight_decay,
            betas=self.config.training.optimizer.betas,
            eps=self.config.training.optimizer.eps
        )
        
        # Create scheduler
        total_steps = self.config.training.scheduler.num_training_steps
        if total_steps is None:
            # Estimate total steps (will be updated during training)
            total_steps = 10000  # Placeholder
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.scheduler.num_warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info("Optimizer and scheduler setup complete")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
            
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                input_values=batch["input_values"],
                labels=batch.get("labels")
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if hasattr(self.config.training, "max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log training progress
            if self.global_step % self.config.training.logging.log_every_n_steps == 0:
                log_training_progress(
                    self.logger,
                    epoch,
                    self.global_step,
                    loss.item(),
                    {"lr": self.scheduler.get_last_lr()[0]}
                )
                
                # Log to W&B
                if self.wandb_run:
                    self.wandb_run.log({
                        "train/loss": loss.item(),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": self.global_step
                    })
        
        avg_loss = total_loss / num_batches
        
        return {
            "loss": avg_loss,
            "lr": self.scheduler.get_last_lr()[0]
        }
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.
            
        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(val_loader)
        
        # Reset metrics
        self.metrics.reset()
        
        with torch.no_grad():
            progress_bar = tqdm(
                val_loader,
                desc=f"Validation {epoch}",
                leave=False
            )
            
            for batch in progress_bar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    input_values=batch["input_values"],
                    labels=batch.get("labels")
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                predictions = self.model.decode(outputs.logits)
                references = batch.get("labels", [])
                
                # Add to metrics
                self.metrics.add_batch(
                    predictions=predictions,
                    references=references,
                    accents=batch.get("accents"),
                    speaker_ids=batch.get("speaker_ids")
                )
                
                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute metrics
        val_metrics = self.metrics.compute_all_metrics()
        val_metrics["loss"] = total_loss / num_batches
        
        # Log validation results
        log_evaluation_results(self.logger, val_metrics, "validation")
        
        # Log to W&B
        if self.wandb_run:
            wandb_log = {"val/loss": val_metrics["loss"]}
            wandb_log.update({f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))})
            self.wandb_run.log(wandb_log)
        
        return val_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            num_epochs: Number of epochs to train.
            
        Returns:
            Dictionary with training history.
        """
        if num_epochs is None:
            num_epochs = self.config.training.epochs
        
        # Update scheduler with correct number of steps
        total_steps = num_epochs * len(train_loader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.scheduler.num_warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_wer": [],
            "val_cer": []
        }
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader, epoch)
                history["val_loss"].append(val_metrics["loss"])
                history["val_wer"].append(val_metrics["wer"])
                history["val_cer"].append(val_metrics["cer"])
                
                # Check for best model
                monitor_metric = self.config.training.checkpointing.monitor_metric
                if monitor_metric in val_metrics:
                    current_metric = val_metrics[monitor_metric]
                    
                    if self._is_better_metric(current_metric):
                        self.best_metric = current_metric
                        self.patience_counter = 0
                        
                        # Save best model
                        if self.config.training.checkpointing.save_best:
                            self.save_checkpoint("best_model.pt")
                    else:
                        self.patience_counter += 1
                
                # Early stopping
                if self.config.training.early_stopping.enabled:
                    if self.patience_counter >= self.config.training.early_stopping.patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Save checkpoint
            if self.config.training.checkpointing.save_every_n_epochs > 0:
                if (epoch + 1) % self.config.training.checkpointing.save_every_n_epochs == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
            # Save last checkpoint
            if self.config.training.checkpointing.save_last:
                self.save_checkpoint("last_model.pt")
        
        self.logger.info("Training completed")
        return history
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _is_better_metric(self, metric: float) -> bool:
        """Check if metric is better than current best."""
        mode = self.config.training.checkpointing.mode
        if mode == "min":
            return metric < self.best_metric
        else:  # max
            return metric > self.best_metric
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint_dir = Path(self.config.paths.checkpoints_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.wandb_run:
            self.wandb_run.finish()
