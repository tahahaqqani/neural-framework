"""
Data Handling and Processing Module

This module provides generic data handling utilities that can work with
various types of data for different machine learning tasks.
"""

import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Preprocessing
    normalize_features: bool = True
    normalize_targets: bool = False
    scaler_type: str = "standard"  # "standard", "minmax", "none"
    
    # Data augmentation
    augment_data: bool = False
    augmentation_factor: float = 1.0
    
    # Time series specific
    sequence_length: int = 1
    prediction_horizon: int = 1
    
    # Text specific
    max_vocab_size: int = 10000
    max_sequence_length: int = 100
    
    # Image specific
    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "normalize_features": self.normalize_features,
            "normalize_targets": self.normalize_targets,
            "scaler_type": self.scaler_type,
            "augment_data": self.augment_data,
            "augmentation_factor": self.augmentation_factor,
            "sequence_length": self.sequence_length,
            "prediction_horizon": self.prediction_horizon,
            "max_vocab_size": self.max_vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "image_size": self.image_size,
            "channels": self.channels
        }


class GenericDataset(data.Dataset):
    """
    Generic PyTorch dataset that can handle various data types.
    """
    
    def __init__(self, 
                 features: Union[np.ndarray, List, torch.Tensor],
                 targets: Optional[Union[np.ndarray, List, torch.Tensor]] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the dataset.
        
        Args:
            features: Input features
            targets: Target values (optional for unsupervised learning)
            transform: Transform to apply to features
            target_transform: Transform to apply to targets
        """
        self.features = self._to_tensor(features)
        self.targets = self._to_tensor(targets) if targets is not None else None
        self.transform = transform
        self.target_transform = target_transform
    
    def _to_tensor(self, data: Union[np.ndarray, List, torch.Tensor]) -> torch.Tensor:
        """Convert data to tensor."""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, list):
            return torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, targets) or (features,) if no targets
        """
        features = self.features[idx]
        targets = self.targets[idx] if self.targets is not None else None
        
        if self.transform:
            features = self.transform(features)
        
        if self.target_transform and targets is not None:
            targets = self.target_transform(targets)
        
        if targets is not None:
            return features, targets
        else:
            return features,


class DataHandler:
    """
    Generic data handler for various machine learning tasks.
    
    This class provides utilities for:
    - Loading data from various sources
    - Preprocessing and normalization
    - Train/validation/test splitting
    - Data augmentation
    - Creating PyTorch data loaders
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize the data handler.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        self.feature_scaler = None
        self.target_scaler = None
        self.label_encoders = {}
        self.vocab = {}
        self.logger = logging.getLogger(__name__)
    
    def load_from_numpy(self, 
                       features: np.ndarray, 
                       targets: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load data from NumPy arrays.
        
        Args:
            features: Feature array
            targets: Target array (optional)
            
        Returns:
            Tuple of (features, targets) tensors
        """
        features_tensor = torch.from_numpy(features).float()
        targets_tensor = torch.from_numpy(targets).float() if targets is not None else None
        
        return features_tensor, targets_tensor
    
    def load_from_pandas(self, 
                        df: pd.DataFrame, 
                        feature_columns: List[str],
                        target_columns: Optional[List[str]] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load data from pandas DataFrame.
        
        Args:
            df: DataFrame
            feature_columns: List of feature column names
            target_columns: List of target column names (optional)
            
        Returns:
            Tuple of (features, targets) tensors
        """
        features = df[feature_columns].values
        targets = df[target_columns].values if target_columns else None
        
        return self.load_from_numpy(features, targets)
    
    def load_from_csv(self, 
                     filepath: Union[str, Path],
                     feature_columns: List[str],
                     target_columns: Optional[List[str]] = None,
                     **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            feature_columns: List of feature column names
            target_columns: List of target column names (optional)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Tuple of (features, targets) tensors
        """
        df = pd.read_csv(filepath, **kwargs)
        return self.load_from_pandas(df, feature_columns, target_columns)
    
    def preprocess_data(self, 
                       features: torch.Tensor, 
                       targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Preprocess the data based on configuration.
        
        Args:
            features: Feature tensor
            targets: Target tensor (optional)
            
        Returns:
            Tuple of preprocessed (features, targets)
        """
        # Convert to numpy for preprocessing
        features_np = features.numpy()
        targets_np = targets.numpy() if targets is not None else None
        
        # Normalize features
        if self.config.normalize_features and self.config.scaler_type != "none":
            self.feature_scaler = self._get_scaler(self.config.scaler_type)
            features_np = self.feature_scaler.fit_transform(features_np)
            self.logger.info(f"Normalized features using {self.config.scaler_type} scaler")
        
        # Normalize targets
        if self.config.normalize_targets and targets_np is not None and self.config.scaler_type != "none":
            self.target_scaler = self._get_scaler(self.config.scaler_type)
            targets_np = self.target_scaler.fit_transform(targets_np)
            self.logger.info(f"Normalized targets using {self.config.scaler_type} scaler")
        
        # Convert back to tensors
        features_processed = torch.from_numpy(features_np).float()
        targets_processed = torch.from_numpy(targets_np).float() if targets_np is not None else None
        
        return features_processed, targets_processed
    
    def _get_scaler(self, scaler_type: str):
        """Get the appropriate scaler."""
        if scaler_type == "standard":
            return StandardScaler()
        elif scaler_type == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def split_data(self, 
                  features: torch.Tensor, 
                  targets: Optional[torch.Tensor] = None,
                  random_state: int = 42) -> Tuple[Tuple[torch.Tensor, torch.Tensor], 
                                                  Tuple[torch.Tensor, torch.Tensor], 
                                                  Tuple[torch.Tensor, torch.Tensor]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            features: Feature tensor
            targets: Target tensor (optional)
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of ((train_features, train_targets), (val_features, val_targets), (test_features, test_targets))
        """
        # Convert to numpy for splitting
        features_np = features.numpy()
        targets_np = targets.numpy() if targets is not None else None
        
        # Calculate split sizes
        total_size = len(features_np)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.val_ratio)
        
        # Split data
        if targets_np is not None:
            # Supervised learning
            X_train, X_temp, y_train, y_temp = train_test_split(
                features_np, targets_np, 
                test_size=(1 - self.config.train_ratio), 
                random_state=random_state
            )
            
            if self.config.test_ratio > 0:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp,
                    test_size=(self.config.test_ratio / (self.config.val_ratio + self.config.test_ratio)),
                    random_state=random_state
                )
            else:
                # No test split, use all remaining data for validation
                X_val, y_val = X_temp, y_temp
                X_test, y_test = np.array([]), np.array([])
            
            return (
                (torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
                (torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()),
                (torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
            )
        else:
            # Unsupervised learning
            X_train, X_temp = train_test_split(
                features_np,
                test_size=(1 - self.config.train_ratio),
                random_state=random_state
            )
            
            if self.config.test_ratio > 0:
                X_val, X_test = train_test_split(
                    X_temp,
                    test_size=(self.config.test_ratio / (self.config.val_ratio + self.config.test_ratio)),
                    random_state=random_state
                )
            else:
                # No test split, use all remaining data for validation
                X_val = X_temp
                X_test = np.array([])
            
            return (
                (torch.from_numpy(X_train).float(), None),
                (torch.from_numpy(X_val).float(), None),
                (torch.from_numpy(X_test).float(), None)
            )
    
    def create_data_loaders(self, 
                           train_data: Tuple[torch.Tensor, Optional[torch.Tensor]],
                           val_data: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
                           test_data: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
                           batch_size: int = 32,
                           num_workers: int = 0) -> Tuple[data.DataLoader, Optional[data.DataLoader], Optional[data.DataLoader]]:
        """
        Create PyTorch data loaders.
        
        Args:
            train_data: Training data tuple
            val_data: Validation data tuple (optional)
            test_data: Test data tuple (optional)
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = GenericDataset(*train_data)
        val_dataset = GenericDataset(*val_data) if val_data and len(val_data[0]) > 0 else None
        test_dataset = GenericDataset(*test_data) if test_data and len(test_data[0]) > 0 else None
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )
        
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        ) if val_dataset else None
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False
        ) if test_dataset else None
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform targets if they were normalized.
        
        Args:
            targets: Normalized target tensor
            
        Returns:
            Original scale target tensor
        """
        if self.target_scaler is not None:
            targets_np = targets.numpy()
            targets_original = self.target_scaler.inverse_transform(targets_np)
            return torch.from_numpy(targets_original).float()
        return targets
    
    def save_preprocessors(self, filepath: Union[str, Path]):
        """Save preprocessors to file."""
        preprocessors = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'label_encoders': self.label_encoders,
            'vocab': self.vocab,
            'config': self.config.to_dict()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        
        self.logger.info(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath: Union[str, Path]):
        """Load preprocessors from file."""
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.feature_scaler = preprocessors['feature_scaler']
        self.target_scaler = preprocessors['target_scaler']
        self.label_encoders = preprocessors['label_encoders']
        self.vocab = preprocessors['vocab']
        
        self.logger.info(f"Preprocessors loaded from {filepath}")


def create_data_handler(config: DatasetConfig) -> DataHandler:
    """
    Factory function to create a data handler.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Initialized data handler
    """
    return DataHandler(config)
