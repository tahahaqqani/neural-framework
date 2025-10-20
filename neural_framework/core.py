"""
Core Neural Network Models and Configurations

This module contains the core neural network architectures and configuration
classes that form the foundation of the framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path


class ActivationType(Enum):
    """Supported activation functions."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    GELU = "gelu"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"
    LINEAR = "linear"


class LayerType(Enum):
    """Supported layer types."""
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"


@dataclass
class LayerConfig:
    """Configuration for a single layer."""
    type: LayerType
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    activation: Optional[ActivationType] = None
    dropout_rate: float = 0.0
    bias: bool = True
    # Convolution specific
    kernel_size: Optional[int] = None
    stride: int = 1
    padding: int = 0
    # RNN specific
    hidden_size: Optional[int] = None
    num_layers: int = 1
    bidirectional: bool = False
    # Normalization specific
    normalized_shape: Optional[Tuple[int, ...]] = None


@dataclass
class ModelConfig:
    """Configuration for the entire model."""
    input_size: int
    output_size: int
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])
    activations: List[ActivationType] = field(default_factory=lambda: [ActivationType.RELU, ActivationType.RELU])
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.1])
    layer_configs: List[LayerConfig] = field(default_factory=list)
    output_activation: Optional[ActivationType] = None
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    residual_connections: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_sizes": self.hidden_sizes,
            "activations": [a.value for a in self.activations],
            "dropout_rates": self.dropout_rates,
            "layer_configs": [
                {
                    "type": lc.type.value,
                    "input_size": lc.input_size,
                    "output_size": lc.output_size,
                    "activation": lc.activation.value if lc.activation else None,
                    "dropout_rate": lc.dropout_rate,
                    "bias": lc.bias,
                    "kernel_size": lc.kernel_size,
                    "stride": lc.stride,
                    "padding": lc.padding,
                    "hidden_size": lc.hidden_size,
                    "num_layers": lc.num_layers,
                    "bidirectional": lc.bidirectional,
                    "normalized_shape": lc.normalized_shape
                }
                for lc in self.layer_configs
            ],
            "output_activation": self.output_activation.value if self.output_activation else None,
            "use_batch_norm": self.use_batch_norm,
            "use_layer_norm": self.use_layer_norm,
            "residual_connections": self.residual_connections
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        # Convert activation strings back to enums
        activations = [ActivationType(a) for a in config_dict.get("activations", [])]
        output_activation = ActivationType(config_dict["output_activation"]) if config_dict.get("output_activation") else None
        
        # Convert layer configs
        layer_configs = []
        for lc_dict in config_dict.get("layer_configs", []):
            layer_config = LayerConfig(
                type=LayerType(lc_dict["type"]),
                input_size=lc_dict.get("input_size"),
                output_size=lc_dict.get("output_size"),
                activation=ActivationType(lc_dict["activation"]) if lc_dict.get("activation") else None,
                dropout_rate=lc_dict.get("dropout_rate", 0.0),
                bias=lc_dict.get("bias", True),
                kernel_size=lc_dict.get("kernel_size"),
                stride=lc_dict.get("stride", 1),
                padding=lc_dict.get("padding", 0),
                hidden_size=lc_dict.get("hidden_size"),
                num_layers=lc_dict.get("num_layers", 1),
                bidirectional=lc_dict.get("bidirectional", False),
                normalized_shape=tuple(lc_dict["normalized_shape"]) if lc_dict.get("normalized_shape") else None
            )
            layer_configs.append(layer_config)
        
        return cls(
            input_size=config_dict["input_size"],
            output_size=config_dict["output_size"],
            hidden_sizes=config_dict.get("hidden_sizes", [64, 32]),
            activations=activations,
            dropout_rates=config_dict.get("dropout_rates", [0.1, 0.1]),
            layer_configs=layer_configs,
            output_activation=output_activation,
            use_batch_norm=config_dict.get("use_batch_norm", False),
            use_layer_norm=config_dict.get("use_layer_norm", False),
            residual_connections=config_dict.get("residual_connections", False)
        )
    
    def save(self, filepath: Union[str, Path]):
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ModelConfig':
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class NeuralModel(nn.Module):
    """
    Generic neural network model that can be configured for various tasks.
    
    This model supports:
    - Feedforward networks
    - Convolutional networks
    - Recurrent networks (LSTM/GRU)
    - Residual connections
    - Batch/Layer normalization
    - Multiple activation functions
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the neural model.
        
        Args:
            config: Model configuration
        """
        super(NeuralModel, self).__init__()
        
        self.config = config
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        
        self._build_model()
        self._initialize_weights()
    
    def _build_model(self):
        """Build the model based on configuration."""
        # If custom layer configs are provided, use them
        if self.config.layer_configs:
            self._build_from_layer_configs()
        else:
            self._build_feedforward()
    
    def _build_from_layer_configs(self):
        """Build model from custom layer configurations."""
        for i, layer_config in enumerate(self.config.layer_configs):
            layer = self._create_layer(layer_config, i)
            if layer is not None:
                self.layers.append(layer)
            
            # Add activation if specified
            if layer_config.activation:
                self.activations.append(self._get_activation(layer_config.activation))
            else:
                self.activations.append(nn.Identity())
            
            # Add dropout if specified
            if layer_config.dropout_rate > 0:
                self.dropouts.append(nn.Dropout(layer_config.dropout_rate))
            else:
                self.dropouts.append(nn.Identity())
            
            # Add normalization if specified
            if self.config.use_batch_norm and layer_config.type in [LayerType.LINEAR, LayerType.CONV1D, LayerType.CONV2D]:
                if layer_config.type == LayerType.LINEAR:
                    self.normalizations.append(nn.BatchNorm1d(layer_config.output_size))
                elif layer_config.type == LayerType.CONV1D:
                    self.normalizations.append(nn.BatchNorm1d(layer_config.output_size))
                elif layer_config.type == LayerType.CONV2D:
                    self.normalizations.append(nn.BatchNorm2d(layer_config.output_size))
            elif self.config.use_layer_norm and layer_config.type in [LayerType.LINEAR, LayerType.LSTM, LayerType.GRU]:
                if layer_config.normalized_shape:
                    self.normalizations.append(nn.LayerNorm(layer_config.normalized_shape))
                else:
                    self.normalizations.append(nn.LayerNorm(layer_config.output_size))
            else:
                self.normalizations.append(nn.Identity())
    
    def _build_feedforward(self):
        """Build a standard feedforward network."""
        # Input layer
        prev_size = self.config.input_size
        
        for i, (hidden_size, activation, dropout_rate) in enumerate(
            zip(self.config.hidden_sizes, self.config.activations, self.config.dropout_rates)
        ):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Activation
            self.activations.append(self._get_activation(activation))
            
            # Dropout
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            else:
                self.dropouts.append(nn.Identity())
            
            # Normalization
            if self.config.use_batch_norm:
                self.normalizations.append(nn.BatchNorm1d(hidden_size))
            elif self.config.use_layer_norm:
                self.normalizations.append(nn.LayerNorm(hidden_size))
            else:
                self.normalizations.append(nn.Identity())
            
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(nn.Linear(prev_size, self.config.output_size))
        
        # Output activation
        if self.config.output_activation:
            self.activations.append(self._get_activation(self.config.output_activation))
        else:
            self.activations.append(nn.Identity())
        
        # No dropout or normalization for output layer
        self.dropouts.append(nn.Identity())
        self.normalizations.append(nn.Identity())
    
    def _create_layer(self, layer_config: LayerConfig, index: int) -> Optional[nn.Module]:
        """Create a specific layer based on configuration."""
        if layer_config.type == LayerType.LINEAR:
            return nn.Linear(layer_config.input_size, layer_config.output_size, bias=layer_config.bias)
        
        elif layer_config.type == LayerType.CONV1D:
            return nn.Conv1d(
                layer_config.input_size, 
                layer_config.output_size,
                kernel_size=layer_config.kernel_size,
                stride=layer_config.stride,
                padding=layer_config.padding,
                bias=layer_config.bias
            )
        
        elif layer_config.type == LayerType.CONV2D:
            return nn.Conv2d(
                layer_config.input_size,
                layer_config.output_size,
                kernel_size=layer_config.kernel_size,
                stride=layer_config.stride,
                padding=layer_config.padding,
                bias=layer_config.bias
            )
        
        elif layer_config.type == LayerType.LSTM:
            return nn.LSTM(
                layer_config.input_size,
                layer_config.hidden_size,
                num_layers=layer_config.num_layers,
                batch_first=True,
                bidirectional=layer_config.bidirectional,
                bias=layer_config.bias
            )
        
        elif layer_config.type == LayerType.GRU:
            return nn.GRU(
                layer_config.input_size,
                layer_config.hidden_size,
                num_layers=layer_config.num_layers,
                batch_first=True,
                bidirectional=layer_config.bidirectional,
                bias=layer_config.bias
            )
        
        elif layer_config.type == LayerType.DROPOUT:
            return nn.Dropout(layer_config.dropout_rate)
        
        elif layer_config.type == LayerType.BATCH_NORM:
            if layer_config.normalized_shape:
                return nn.BatchNorm1d(layer_config.normalized_shape[0])
            else:
                return nn.BatchNorm1d(layer_config.output_size)
        
        elif layer_config.type == LayerType.LAYER_NORM:
            if layer_config.normalized_shape:
                return nn.LayerNorm(layer_config.normalized_shape)
            else:
                return nn.LayerNorm(layer_config.output_size)
        
        return None
    
    def _get_activation(self, activation_type: ActivationType) -> nn.Module:
        """Get activation function module."""
        if activation_type == ActivationType.RELU:
            return nn.ReLU()
        elif activation_type == ActivationType.TANH:
            return nn.Tanh()
        elif activation_type == ActivationType.SIGMOID:
            return nn.Sigmoid()
        elif activation_type == ActivationType.GELU:
            return nn.GELU()
        elif activation_type == ActivationType.SWISH:
            return nn.SiLU()  # SiLU is the same as Swish
        elif activation_type == ActivationType.LEAKY_RELU:
            return nn.LeakyReLU()
        else:
            return nn.Identity()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for i, (layer, activation, dropout, norm) in enumerate(
            zip(self.layers, self.activations, self.dropouts, self.normalizations)
        ):
            # Skip if layer is None (e.g., for dropout-only layers)
            if layer is None:
                continue
            
            # Apply layer
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                x, _ = layer(x)
            else:
                x = layer(x)
            
            # Apply normalization
            x = norm(x)
            
            # Apply activation
            x = activation(x)
            
            # Apply dropout
            x = dropout(x)
        
        return x
    
    def get_parameter_count(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model(self, filepath: Union[str, Path]):
        """Save model state dict."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict()
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'NeuralModel':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        config = ModelConfig.from_dict(checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_model(config: ModelConfig) -> NeuralModel:
    """
    Factory function to create a neural model.
    
    Args:
        config: Model configuration
        
    Returns:
        Initialized neural model
    """
    return NeuralModel(config)


def count_parameters(model: NeuralModel) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Neural model
        
    Returns:
        Number of trainable parameters
    """
    return model.get_parameter_count()
