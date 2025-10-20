"""
Simple Regression Example

This example demonstrates how to use the neural framework for a basic regression task.
We'll create synthetic data and train a neural network to predict continuous values.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the parent directory to the path so we can import the framework
sys.path.append(str(Path(__file__).parent.parent))

from neural_framework import (
    NeuralModel, ModelConfig, ActivationType,
    DataHandler, DatasetConfig,
    Trainer, TrainingConfig,
    Evaluator, EvaluationConfig,
    setup_logging, set_seed
)


def generate_synthetic_data(n_samples: int = 1000, noise_level: float = 0.1) -> tuple:
    """
    Generate synthetic regression data.
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (features, targets)
    """
    # Generate features (2D input)
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    
    # Generate targets with a non-linear relationship
    # y = x1^2 + x2^2 + 0.5*x1*x2 + noise
    y = X[:, 0]**2 + X[:, 1]**2 + 0.5 * X[:, 0] * X[:, 1] + noise_level * np.random.randn(n_samples)
    
    return X, y.reshape(-1, 1)


def main():
    """Main function to run the regression example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Simple Regression Example")
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=1000, noise_level=0.1)
    logger.info(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    
    # Configure data handling
    data_config = DatasetConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize_features=True,
        normalize_targets=True,
        scaler_type="standard"
    )
    
    # Create data handler
    data_handler = DataHandler(data_config)
    
    # Load and preprocess data
    features, targets = data_handler.load_from_numpy(X, y)
    features, targets = data_handler.preprocess_data(features, targets)
    
    # Split data
    train_data, val_data, test_data = data_handler.split_data(features, targets)
    logger.info(f"Data split - Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_handler.create_data_loaders(
        train_data, val_data, test_data, batch_size=32
    )
    
    # Configure model
    model_config = ModelConfig(
        input_size=2,
        output_size=1,
        hidden_sizes=[64, 32, 16],
        activations=[ActivationType.RELU, ActivationType.RELU, ActivationType.RELU],
        dropout_rates=[0.1, 0.1, 0.1],
        output_activation=None,  # No activation for regression
        use_batch_norm=True
    )
    
    # Create model
    model = NeuralModel(model_config)
    logger.info(f"Created model with {model.get_parameter_count()} parameters")
    
    # Configure training
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        optimizer="adam",
        loss_function="mse",
        early_stopping_patience=10,
        save_best_model=True,
        model_save_path="models/regression",
        log_save_path="logs/regression"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/regression_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        regression_metrics=["mse", "mae", "rmse", "r2"],
        plot_predictions=True,
        save_plots=True,
        output_dir="results/regression"
    )
    
    # Create evaluator
    evaluator = Evaluator(model, eval_config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    report = evaluator.generate_report(
        test_loader, 
        task_type="regression",
        target_scaler=data_handler.target_scaler
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in report["metrics"].items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Save preprocessors for future use
    data_handler.save_preprocessors("models/regression/preprocessors.pkl")
    
    logger.info("Simple regression example completed!")


if __name__ == "__main__":
    main()
