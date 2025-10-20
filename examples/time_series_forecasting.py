"""
Time Series Forecasting Example

This example demonstrates how to use the neural framework for time series forecasting.
We'll create synthetic time series data and train a neural network to predict future values.
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


def generate_time_series_data(n_samples: int = 1000, 
                            sequence_length: int = 10, 
                            prediction_horizon: int = 1,
                            noise_level: float = 0.1) -> tuple:
    """
    Generate synthetic time series data.
    
    Args:
        n_samples: Number of samples to generate
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict ahead
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (sequences, targets)
    """
    np.random.seed(42)
    
    # Generate a long time series
    total_length = n_samples + sequence_length + prediction_horizon
    t = np.linspace(0, 4 * np.pi, total_length)
    
    # Create a complex time series with trend, seasonality, and noise
    trend = 0.01 * t
    seasonal = np.sin(t) + 0.5 * np.sin(2 * t) + 0.25 * np.sin(4 * t)
    noise = noise_level * np.random.randn(total_length)
    
    time_series = trend + seasonal + noise
    
    # Create sequences and targets
    sequences = []
    targets = []
    
    for i in range(n_samples):
        # Input sequence
        seq = time_series[i:i + sequence_length]
        sequences.append(seq)
        
        # Target (next value or values)
        if prediction_horizon == 1:
            target = time_series[i + sequence_length]
        else:
            target = time_series[i + sequence_length:i + sequence_length + prediction_horizon]
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


def main():
    """Main function to run the time series forecasting example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Time Series Forecasting Example")
    
    # Generate synthetic time series data
    logger.info("Generating synthetic time series data...")
    sequences, targets = generate_time_series_data(
        n_samples=1000, 
        sequence_length=10, 
        prediction_horizon=1,
        noise_level=0.1
    )
    logger.info(f"Generated {sequences.shape[0]} sequences with length {sequences.shape[1]}")
    
    # Configure data handling
    data_config = DatasetConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize_features=True,
        normalize_targets=True,
        scaler_type="standard",
        sequence_length=10,
        prediction_horizon=1
    )
    
    # Create data handler
    data_handler = DataHandler(data_config)
    
    # Load and preprocess data
    features, targets = data_handler.load_from_numpy(sequences, targets.reshape(-1, 1))
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
        input_size=10,  # sequence length
        output_size=1,  # prediction horizon
        hidden_sizes=[64, 32],
        activations=[ActivationType.RELU, ActivationType.RELU],
        dropout_rates=[0.1, 0.1],
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
        model_save_path="models/time_series",
        log_save_path="logs/time_series"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/time_series_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        regression_metrics=["mse", "mae", "rmse", "r2"],
        plot_predictions=True,
        save_plots=True,
        output_dir="results/time_series"
    )
    
    # Create evaluator
    evaluator = Evaluator(model, eval_config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    report = evaluator.generate_report(
        test_loader, 
        task_type="time_series",
        target_scaler=data_handler.target_scaler
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in report["metrics"].items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric.upper()}: {value:.4f}")
        else:
            logger.info(f"  {metric.upper()}: {value}")
    
    # Visualize time series predictions
    plot_time_series_predictions(model, sequences, targets, data_handler, 
                                "results/time_series_predictions.png")
    
    logger.info("Time series forecasting example completed!")


def plot_time_series_predictions(model, sequences, targets, data_handler, save_path=None):
    """Plot time series predictions."""
    # Get predictions on test data
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), 32):  # Process in batches
            batch_sequences = sequences[i:i+32]
            batch_tensor = torch.FloatTensor(batch_sequences)
            
            # Normalize if scaler exists
            if data_handler.feature_scaler:
                batch_tensor = torch.FloatTensor(
                    data_handler.feature_scaler.transform(batch_tensor.numpy())
                )
            
            batch_pred = model(batch_tensor)
            
            # Denormalize if scaler exists
            if data_handler.target_scaler:
                batch_pred = data_handler.target_scaler.inverse_transform(
                    batch_pred.numpy()
                )
                batch_pred = torch.FloatTensor(batch_pred)
            
            predictions.extend(batch_pred.numpy().flatten())
    
    predictions = np.array(predictions)
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    # Convert targets to numpy if it's a tensor
    if hasattr(targets, 'numpy'):
        targets_np = targets.numpy()
    else:
        targets_np = targets
    
    # Plot original time series
    plt.subplot(2, 1, 1)
    time_indices = np.arange(len(sequences))
    plt.plot(time_indices, targets_np[:len(sequences)], label='Actual', alpha=0.7)
    plt.plot(time_indices, predictions[:len(sequences)], label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    
    # Plot prediction errors
    plt.subplot(2, 1, 2)
    errors = predictions[:len(sequences)] - targets_np[:len(sequences)]
    plt.plot(time_indices, errors, label='Prediction Error', color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title('Prediction Errors Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
