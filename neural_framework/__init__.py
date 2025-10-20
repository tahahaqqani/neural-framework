"""
Neural Framework - A Generic Deep Learning Framework

A flexible, domain-agnostic neural network framework built on PyTorch that can be
easily adapted for various machine learning tasks including:
- Recommendation Systems
- Natural Language Processing
- Anomaly Detection
- Time Series Forecasting
- Computer Vision
- And more...

Key Features:
- Generic model architectures
- Flexible data handling
- Comprehensive training utilities
- Built-in evaluation metrics
- Easy configuration management
- Multiple domain examples
"""

__version__ = "1.0.0"
__author__ = "Neural Framework Team"

from .core import NeuralModel, ModelConfig, ActivationType
from .data import DataHandler, DatasetConfig
from .training import Trainer, TrainingConfig
from .evaluation import Evaluator, EvaluationConfig
from .utils import setup_logging, set_seed

__all__ = [
    "NeuralModel",
    "ModelConfig",
    "ActivationType",
    "DataHandler",
    "DatasetConfig",
    "Trainer",
    "TrainingConfig",
    "Evaluator",
    "EvaluationConfig",
    "setup_logging",
    "set_seed"
]
