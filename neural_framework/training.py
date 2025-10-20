"""
Training Module for Neural Networks

This module provides comprehensive training utilities including:
- Generic training loops
- Multiple optimizers and schedulers
- Loss functions for different tasks
- Callbacks and monitoring
- Model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 100
    weight_decay: float = 1e-5
    
    # Optimizer
    optimizer: str = "adam"  # "adam", "sgd", "rmsprop", "adamw"
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9  # For Adam
    beta2: float = 0.999  # For Adam
    eps: float = 1e-8  # For Adam
    
    # Scheduler
    scheduler: str = "none"  # "none", "step", "cosine", "plateau", "exponential"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Loss function
    loss_function: str = "mse"  # "mse", "mae", "cross_entropy", "bce", "nll"
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training options
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    gradient_clip_norm: Optional[float] = None
    mixed_precision: bool = False
    
    # Logging and saving
    log_interval: int = 10
    save_interval: int = 10
    save_best_model: bool = True
    model_save_path: str = "models"
    log_save_path: str = "logs"
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "momentum": self.momentum,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "scheduler": self.scheduler,
            "scheduler_params": self.scheduler_params,
            "loss_function": self.loss_function,
            "loss_params": self.loss_params,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "gradient_clip_norm": self.gradient_clip_norm,
            "mixed_precision": self.mixed_precision,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "save_best_model": self.save_best_model,
            "model_save_path": self.model_save_path,
            "log_save_path": self.log_save_path,
            "device": self.device
        }


class Callback(ABC):
    """Abstract base class for training callbacks."""
    
    @abstractmethod
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        """Called at the beginning of each batch."""
        pass
    
    @abstractmethod
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        """Called at the end of each batch."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, monitor: str = "val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        current_score = logs.get(self.monitor, float('inf'))
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score - self.min_delta:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        pass


class ModelCheckpointCallback(Callback):
    """Model checkpointing callback."""
    
    def __init__(self, filepath: str, monitor: str = "val_loss", save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best_score = None
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        current_score = logs.get(self.monitor, float('inf'))
        
        if not self.save_best_only or self.best_score is None or current_score < self.best_score:
            self.best_score = current_score
            torch.save(logs.get('model_state_dict'), self.filepath)
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]):
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]):
        pass


class Trainer:
    """
    Generic trainer for neural networks.
    
    This trainer supports:
    - Multiple optimizers and schedulers
    - Various loss functions
    - Callbacks and monitoring
    - Mixed precision training
    - Gradient clipping
    - Early stopping
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 callbacks: Optional[List[Callback]] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            config: Training configuration
            callbacks: List of callbacks
        """
        self.model = model
        self.config = config
        self.callbacks = callbacks or []
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup mixed precision
        if self.config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.config.device)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        if self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum
            )
        elif self.config.optimizer == "rmsprop":
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if self.config.scheduler == "none":
            return None
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_params.get("step_size", 30),
                gamma=self.config.scheduler_params.get("gamma", 0.1)
            )
        elif self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.scheduler_params.get("eta_min", 0)
            )
        elif self.config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.scheduler_params.get("mode", "min"),
                factor=self.config.scheduler_params.get("factor", 0.5),
                patience=self.config.scheduler_params.get("patience", 10),
                verbose=True
            )
        elif self.config.scheduler == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.scheduler_params.get("gamma", 0.95)
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.scheduler}")
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function."""
        if self.config.loss_function == "mse":
            return nn.MSELoss(**self.config.loss_params)
        elif self.config.loss_function == "mae":
            return nn.L1Loss(**self.config.loss_params)
        elif self.config.loss_function == "cross_entropy":
            return nn.CrossEntropyLoss(**self.config.loss_params)
        elif self.config.loss_function == "bce":
            return nn.BCELoss(**self.config.loss_params)
        elif self.config.loss_function == "nll":
            return nn.NLLLoss(**self.config.loss_params)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
    
    def _setup_logging(self):
        """Setup logging."""
        Path(self.config.log_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_save_path).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Prepare batch
            if len(batch) == 2:
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)
            else:
                features = batch[0].to(self.device)
                targets = None
            
            # Callbacks
            logs = {'batch': batch_idx, 'features': features, 'targets': targets}
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, logs)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.scaler and self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    predictions = self.model(features)
                    if targets is not None:
                        loss = self.criterion(predictions, targets)
                    else:
                        loss = self.criterion(predictions, features)  # For autoencoders
            else:
                predictions = self.model(features)
                if targets is not None:
                    loss = self.criterion(predictions, targets)
                else:
                    loss = self.criterion(predictions, features)
            
            # Backward pass
            if self.scaler and self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
            
            # Callbacks
            logs.update({'loss': loss.item(), 'predictions': predictions})
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, logs)
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Prepare batch
                if len(batch) == 2:
                    features, targets = batch
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                else:
                    features = batch[0].to(self.device)
                    targets = None
                
                # Forward pass
                if self.scaler and self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(features)
                        if targets is not None:
                            loss = self.criterion(predictions, targets)
                        else:
                            loss = self.criterion(predictions, features)
                else:
                    predictions = self.model(features)
                    if targets is not None:
                        loss = self.criterion(predictions, targets)
                    else:
                        loss = self.criterion(predictions, features)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Callbacks
            logs = {'epoch': epoch}
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, logs)
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            epoch_time = time.time() - epoch_start
            val_str = f", Val Loss: {val_loss:.6f}" if val_loss is not None else ""
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                           f"Train Loss: {train_loss:.6f}{val_str}, "
                           f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Callbacks
            logs.update({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': current_lr,
                'model_state_dict': self.model.state_dict()
            })
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final model
        self.save_model("final_model.pth")
        
        return self.history
    
    def save_model(self, filename: str):
        """Save model state dict."""
        save_path = Path(self.config.model_save_path) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config.to_dict(),
            'history': self.history
        }, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, filename: str):
        """Load model state dict."""
        load_path = Path(self.config.model_save_path) / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        self.logger.info(f"Model loaded from {load_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Training Loss')
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate plot
        axes[1].plot(self.history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plot saved to {save_path}")
        
        plt.show()


def create_trainer(model: nn.Module, 
                  config: TrainingConfig,
                  callbacks: Optional[List[Callback]] = None) -> Trainer:
    """
    Factory function to create a trainer.
    
    Args:
        model: Neural network model
        config: Training configuration
        callbacks: List of callbacks
        
    Returns:
        Initialized trainer
    """
    return Trainer(model, config, callbacks)
