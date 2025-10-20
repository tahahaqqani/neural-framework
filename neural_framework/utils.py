"""
Utility Functions for Neural Framework

This module provides utility functions for:
- Logging setup
- Random seed setting
- Device management
- Configuration helpers
- Common data transformations
"""

import torch
import numpy as np
import random
import logging
import os
from typing import Optional, Union, Dict, Any
from pathlib import Path
import json
import yaml


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None,
                  log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        log_format: Custom log format (optional)
        
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # Get root logger
    logger = logging.getLogger()
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: Dict[str, Any], filepath: Union[str, Path]):
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        filepath: Path to save the configuration
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


def load_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        filepath: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    filepath = Path(filepath)
    
    if filepath.suffix.lower() == '.yaml' or filepath.suffix.lower() == '.yml':
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def create_directory_structure(base_path: Union[str, Path], 
                             subdirs: Optional[list] = None) -> Path:
    """
    Create directory structure for a project.
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectories to create
        
    Returns:
        Path to base directory
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    if subdirs is None:
        subdirs = ["data", "models", "logs", "results", "configs"]
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(exist_ok=True)
    
    return base_path


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_number(number: Union[int, float]) -> str:
    """
    Format large numbers in human-readable format.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    if number < 1000:
        return str(number)
    elif number < 1000000:
        return f"{number/1000:.1f}K"
    elif number < 1000000000:
        return f"{number/1000000:.1f}M"
    else:
        return f"{number/1000000000:.1f}B"


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": model_size_mb,
        "formatted_total_params": format_number(total_params),
        "formatted_trainable_params": format_number(trainable_params)
    }


def check_gpu_memory() -> Dict[str, Any]:
    """
    Check GPU memory usage.
    
    Returns:
        Dictionary with GPU memory information
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
    
    return {
        "available": True,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "memory_total_mb": memory_total,
        "memory_free_mb": memory_total - memory_reserved,
        "memory_usage_percent": (memory_allocated / memory_total) * 100
    }


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_environment_variables(env_vars: Dict[str, str]):
    """
    Set environment variables.
    
    Args:
        env_vars: Dictionary of environment variable names and values
    """
    for key, value in env_vars.items():
        os.environ[key] = value


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
        "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
        "gpu_info": check_gpu_memory()
    }


def validate_config(config: Dict[str, Any], required_keys: list) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        True if valid, False otherwise
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def print_model_summary(model: torch.nn.Module, input_size: tuple = None):
    """
    Print a summary of the model.
    
    Args:
        model: PyTorch model
        input_size: Input size for the model (optional)
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Model info
    model_info = get_model_size(model)
    print(f"Total Parameters: {model_info['formatted_total_params']}")
    print(f"Trainable Parameters: {model_info['formatted_trainable_params']}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    
    # Layer info
    print("\nLayer Information:")
    print("-" * 80)
    print(f"{'Layer Name':<30} {'Output Shape':<20} {'Parameters':<15}")
    print("-" * 80)
    
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            
            # Get output shape (simplified)
            output_shape = "N/A"
            if hasattr(module, 'out_features'):
                output_shape = f"(*, {module.out_features})"
            elif hasattr(module, 'out_channels'):
                output_shape = f"(*, {module.out_channels}, ...)"
            
            print(f"{name:<30} {output_shape:<20} {format_number(params):<15}")
    
    print("-" * 80)
    print(f"{'Total':<30} {'':<20} {format_number(total_params):<15}")
    print("=" * 80)
