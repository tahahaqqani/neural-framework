"""
Evaluation Module for Neural Networks

This module provides comprehensive evaluation utilities including:
- Multiple evaluation metrics
- Visualization tools
- Performance analysis
- Model comparison utilities
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pandas as pd
import logging


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    # Metrics to compute
    regression_metrics: List[str] = None
    classification_metrics: List[str] = None
    
    # Visualization
    plot_predictions: bool = True
    plot_confusion_matrix: bool = True
    plot_feature_importance: bool = False
    
    # Thresholds
    classification_threshold: float = 0.5
    anomaly_threshold: float = 0.1
    
    # Output
    save_plots: bool = True
    save_metrics: bool = True
    output_dir: str = "evaluation_results"
    
    def __post_init__(self):
        if self.regression_metrics is None:
            self.regression_metrics = ["mse", "mae", "rmse", "r2"]
        if self.classification_metrics is None:
            self.classification_metrics = ["accuracy", "precision", "recall", "f1", "auc"]


class Evaluator:
    """
    Comprehensive evaluator for neural networks.
    
    This evaluator supports:
    - Regression metrics (MSE, MAE, RMSE, R²)
    - Classification metrics (Accuracy, Precision, Recall, F1, AUC)
    - Anomaly detection metrics
    - Time series evaluation
    - Visualization tools
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: EvaluationConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained neural network model
            config: Evaluation configuration
            device: Device to run evaluation on
        """
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def evaluate_regression(self, 
                           test_loader: torch.utils.data.DataLoader,
                           target_scaler: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate model for regression tasks.
        
        Args:
            test_loader: Test data loader
            target_scaler: Scaler used for target normalization (optional)
            
        Returns:
            Dictionary of regression metrics
        """
        predictions, targets = self._get_predictions(test_loader)
        
        # Inverse transform targets if scaler provided
        if target_scaler is not None:
            targets = target_scaler.inverse_transform(targets)
            predictions = target_scaler.inverse_transform(predictions)
        
        # Calculate metrics
        metrics = {}
        
        if "mse" in self.config.regression_metrics:
            metrics["mse"] = mean_squared_error(targets, predictions)
        
        if "mae" in self.config.regression_metrics:
            metrics["mae"] = mean_absolute_error(targets, predictions)
        
        if "rmse" in self.config.regression_metrics:
            metrics["rmse"] = np.sqrt(mean_squared_error(targets, predictions))
        
        if "r2" in self.config.regression_metrics:
            metrics["r2"] = r2_score(targets, predictions)
        
        # Additional regression metrics
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        
        metrics["mape"] = np.mean(np.abs((targets - predictions) / targets)) * 100  # Mean Absolute Percentage Error
        metrics["smape"] = np.mean(2 * np.abs(targets - predictions) / (np.abs(targets) + np.abs(predictions))) * 100  # Symmetric MAPE
        
        return metrics
    
    def evaluate_classification(self, 
                               test_loader: torch.utils.data.DataLoader,
                               num_classes: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model for classification tasks.
        
        Args:
            test_loader: Test data loader
            num_classes: Number of classes (for multi-class)
            
        Returns:
            Dictionary of classification metrics
        """
        predictions, targets = self._get_predictions(test_loader)
        
        # Convert predictions to class labels
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            # Multi-class classification
            pred_labels = np.argmax(predictions, axis=1)
        else:
            # Binary classification
            pred_labels = (predictions > self.config.classification_threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        if "accuracy" in self.config.classification_metrics:
            metrics["accuracy"] = accuracy_score(targets, pred_labels)
        
        if "precision" in self.config.classification_metrics:
            if num_classes and num_classes > 2:
                metrics["precision"] = precision_score(targets, pred_labels, average='weighted')
            else:
                metrics["precision"] = precision_score(targets, pred_labels)
        
        if "recall" in self.config.classification_metrics:
            if num_classes and num_classes > 2:
                metrics["recall"] = recall_score(targets, pred_labels, average='weighted')
            else:
                metrics["recall"] = recall_score(targets, pred_labels)
        
        if "f1" in self.config.classification_metrics:
            if num_classes and num_classes > 2:
                metrics["f1"] = f1_score(targets, pred_labels, average='weighted')
            else:
                metrics["f1"] = f1_score(targets, pred_labels)
        
        if "auc" in self.config.classification_metrics:
            if num_classes and num_classes > 2:
                # Multi-class AUC
                metrics["auc"] = roc_auc_score(targets, predictions, multi_class='ovr')
            else:
                # Binary AUC
                metrics["auc"] = roc_auc_score(targets, predictions)
        
        return metrics
    
    def evaluate_anomaly_detection(self, 
                                  test_loader: torch.utils.data.DataLoader,
                                  normal_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model for anomaly detection tasks.
        
        Args:
            test_loader: Test data loader (contains both normal and anomalous data)
            normal_loader: Normal data loader (optional, for threshold calculation)
            
        Returns:
            Dictionary of anomaly detection metrics
        """
        predictions, targets = self._get_predictions(test_loader)
        
        # Calculate reconstruction error
        if targets is not None:
            reconstruction_error = np.mean((predictions - targets) ** 2, axis=1)
        else:
            reconstruction_error = np.mean(predictions ** 2, axis=1)
        
        # Calculate threshold
        if normal_loader is not None:
            normal_predictions, normal_targets = self._get_predictions(normal_loader)
            if normal_targets is not None:
                normal_error = np.mean((normal_predictions - normal_targets) ** 2, axis=1)
            else:
                normal_error = np.mean(normal_predictions ** 2, axis=1)
            threshold = np.percentile(normal_error, 95)  # 95th percentile
        else:
            threshold = np.percentile(reconstruction_error, 90)  # 90th percentile
        
        # Predict anomalies
        anomaly_predictions = (reconstruction_error > threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            "threshold": threshold,
            "anomaly_rate": np.mean(anomaly_predictions),
            "mean_reconstruction_error": np.mean(reconstruction_error),
            "std_reconstruction_error": np.std(reconstruction_error)
        }
        
        return metrics
    
    def evaluate_time_series(self, 
                            test_loader: torch.utils.data.DataLoader,
                            target_scaler: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate model for time series forecasting tasks.
        
        Args:
            test_loader: Test data loader
            target_scaler: Scaler used for target normalization (optional)
            
        Returns:
            Dictionary of time series metrics
        """
        predictions, targets = self._get_predictions(test_loader)
        
        # Inverse transform targets if scaler provided
        if target_scaler is not None:
            targets = target_scaler.inverse_transform(targets)
            predictions = target_scaler.inverse_transform(predictions)
        
        # Calculate standard regression metrics
        metrics = self.evaluate_regression(test_loader, target_scaler)
        
        # Time series specific metrics
        if predictions.ndim > 1:
            # Multi-step forecasting
            mae_per_step = np.mean(np.abs(targets - predictions), axis=0)
            metrics["mae_per_step"] = mae_per_step.tolist()
            metrics["worst_step_mae"] = np.max(mae_per_step)
            metrics["best_step_mae"] = np.min(mae_per_step)
        
        return metrics
    
    def _get_predictions(self, data_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get predictions from the model.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of (predictions, targets)
        """
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    features, batch_targets = batch
                    features = features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                else:
                    features = batch[0].to(self.device)
                    batch_targets = None
                
                # Get predictions
                batch_predictions = self.model(features)
                
                predictions.append(batch_predictions.cpu().numpy())
                if batch_targets is not None:
                    targets.append(batch_targets.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0) if targets else None
        
        return predictions, targets
    
    def plot_predictions(self, 
                        test_loader: torch.utils.data.DataLoader,
                        target_scaler: Optional[Any] = None,
                        max_samples: int = 1000,
                        save_path: Optional[str] = None):
        """
        Plot predictions vs actual values.
        
        Args:
            test_loader: Test data loader
            target_scaler: Scaler used for target normalization (optional)
            max_samples: Maximum number of samples to plot
            save_path: Path to save the plot (optional)
        """
        predictions, targets = self._get_predictions(test_loader)
        
        # Limit samples for plotting
        if len(predictions) > max_samples:
            indices = np.random.choice(len(predictions), max_samples, replace=False)
            predictions = predictions[indices]
            targets = targets[indices] if targets is not None else None
        
        # Inverse transform if scaler provided
        if target_scaler is not None and targets is not None:
            targets = target_scaler.inverse_transform(targets)
            predictions = target_scaler.inverse_transform(predictions)
        
        # Create plots
        if targets is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Scatter plot
            axes[0, 0].scatter(targets, predictions, alpha=0.6)
            axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', label='Perfect Prediction')
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Predictions vs Actual')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Residuals plot
            residuals = predictions - targets
            axes[0, 1].scatter(predictions, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals Plot')
            axes[0, 1].grid(True)
            
            # Time series plot (if 1D)
            if predictions.ndim == 1:
                axes[1, 0].plot(targets[:100], label='Actual', alpha=0.7)
                axes[1, 0].plot(predictions[:100], label='Predicted', alpha=0.7)
                axes[1, 0].set_xlabel('Time')
                axes[1, 0].set_ylabel('Value')
                axes[1, 0].set_title('Time Series Comparison (First 100 points)')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Distribution plot
            axes[1, 1].hist(targets, bins=30, alpha=0.7, label='Actual', density=True)
            axes[1, 1].hist(predictions, bins=30, alpha=0.7, label='Predicted', density=True)
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Distribution Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
        else:
            # Unsupervised learning - plot predictions only
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram of predictions
            axes[0].hist(predictions.flatten(), bins=30, alpha=0.7)
            axes[0].set_xlabel('Predicted Value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Prediction Distribution')
            axes[0].grid(True)
            
            # Time series of predictions
            if predictions.ndim == 1:
                axes[1].plot(predictions[:100])
                axes[1].set_xlabel('Time')
                axes[1].set_ylabel('Predicted Value')
                axes[1].set_title('Prediction Time Series (First 100 points)')
                axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Prediction plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, 
                             test_loader: torch.utils.data.DataLoader,
                             class_names: Optional[List[str]] = None,
                             save_path: Optional[str] = None):
        """
        Plot confusion matrix for classification tasks.
        
        Args:
            test_loader: Test data loader
            class_names: List of class names
            save_path: Path to save the plot (optional)
        """
        predictions, targets = self._get_predictions(test_loader)
        
        # Convert predictions to class labels
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            pred_labels = np.argmax(predictions, axis=1)
        else:
            pred_labels = (predictions > self.config.classification_threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(targets, pred_labels)
        
        # Generate class names if not provided
        if class_names is None:
            num_classes = cm.shape[0]
            class_names = [f'Class {i}' for i in range(num_classes)]
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, 
                       test_loader: torch.utils.data.DataLoader,
                       task_type: str = "regression",
                       **kwargs) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_loader: Test data loader
            task_type: Type of task ("regression", "classification", "anomaly_detection", "time_series")
            **kwargs: Additional arguments for specific evaluation methods
            
        Returns:
            Dictionary containing evaluation results
        """
        report = {
            "task_type": task_type,
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device)
        }
        
        if task_type == "regression":
            metrics = self.evaluate_regression(test_loader, kwargs.get("target_scaler"))
            report["metrics"] = metrics
            
            if self.config.plot_predictions:
                self.plot_predictions(test_loader, kwargs.get("target_scaler"))
        
        elif task_type == "classification":
            metrics = self.evaluate_classification(test_loader, kwargs.get("num_classes"))
            report["metrics"] = metrics
            
            if self.config.plot_predictions:
                # For classification, use confusion matrix instead of scatter plot
                self.plot_confusion_matrix(test_loader, kwargs.get("class_names"))
            
            if self.config.plot_confusion_matrix:
                self.plot_confusion_matrix(test_loader, kwargs.get("class_names"))
        
        elif task_type == "anomaly_detection":
            metrics = self.evaluate_anomaly_detection(test_loader, kwargs.get("normal_loader"))
            report["metrics"] = metrics
            
            if self.config.plot_predictions:
                self.plot_predictions(test_loader)
        
        elif task_type == "time_series":
            metrics = self.evaluate_time_series(test_loader, kwargs.get("target_scaler"))
            report["metrics"] = metrics
            
            if self.config.plot_predictions:
                self.plot_predictions(test_loader, kwargs.get("target_scaler"))
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Save report
        if self.config.save_metrics:
            report_path = Path(self.config.output_dir) / "evaluation_report.json"
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Convert all values in the report
            json_report = {}
            for key, value in report.items():
                if isinstance(value, dict):
                    json_report[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    json_report[key] = convert_numpy(value)
            
            with open(report_path, 'w') as f:
                json.dump(json_report, f, indent=2)
            self.logger.info(f"Evaluation report saved to {report_path}")
        
        return report


def create_evaluator(model: nn.Module, 
                    config: EvaluationConfig,
                    device: Optional[torch.device] = None) -> Evaluator:
    """
    Factory function to create an evaluator.
    
    Args:
        model: Trained neural network model
        config: Evaluation configuration
        device: Device to run evaluation on
        
    Returns:
        Initialized evaluator
    """
    return Evaluator(model, config, device)
