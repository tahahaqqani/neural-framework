"""
House Price Prediction Example

This example demonstrates how to use the neural framework for real estate price prediction
using the California Housing dataset. This is a classic machine learning problem that
showcases regression capabilities on real-world data.

Dataset: California Housing Prices (from sklearn)
Features: 8 numerical features including median income, house age, rooms per household, etc.
Target: Median house value in California districts (in $100,000s)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Add the parent directory to the path so we can import the framework
sys.path.append(str(Path(__file__).parent.parent))

from neural_framework import (
    NeuralModel, ModelConfig, ActivationType,
    DataHandler, DatasetConfig,
    Trainer, TrainingConfig,
    Evaluator, EvaluationConfig,
    setup_logging, set_seed
)


def load_california_housing_data():
    """
    Load the California Housing dataset.
    
    Returns:
        Tuple of (features, target, feature_names)
    """
    print("Loading California Housing dataset...")
    
    # Load the dataset
    housing_data = fetch_california_housing()
    X, y = housing_data.data, housing_data.target
    feature_names = housing_data.feature_names
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Target range: ${y.min():.2f}K - ${y.max():.2f}K")
    
    return X, y, feature_names


def create_advanced_model_config(input_size: int) -> ModelConfig:
    """
    Create a more sophisticated model configuration for house price prediction.
    
    Args:
        input_size: Number of input features
        
    Returns:
        ModelConfig for the house price prediction model
    """
    return ModelConfig(
        input_size=input_size,
        output_size=1,  # Single output: house price
        hidden_sizes=[128, 64, 32, 16],  # Deeper network
        activations=[
            ActivationType.RELU, 
            ActivationType.RELU, 
            ActivationType.RELU,
            ActivationType.RELU
        ],
        dropout_rates=[0.2, 0.3, 0.2, 0.1],  # Dropout for regularization
        output_activation=None,  # Linear output for regression
        use_batch_norm=True
    )


def plot_feature_importance(model, feature_names, X_test, y_test, save_path=None):
    """
    Plot feature importance using correlation analysis.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X_test: Test features
        y_test: Test targets
        save_path: Path to save the plot
    """
    # Convert to numpy for analysis
    X_test_np = X_test.numpy() if hasattr(X_test, 'numpy') else X_test
    y_test_np = y_test.numpy() if hasattr(y_test, 'numpy') else y_test
    
    # Get model predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_np)
        predictions = model(X_tensor).numpy().flatten()
    
    # Calculate correlation between features and predictions
    correlations = []
    for i in range(X_test_np.shape[1]):
        corr = np.corrcoef(X_test_np[:, i], predictions)[0, 1]
        correlations.append(abs(corr))  # Use absolute correlation
    
    # Sort features by correlation
    sorted_idx = np.argsort(correlations)[::-1]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(feature_names)), 
                   [correlations[i] for i in sorted_idx])
    plt.yticks(range(len(feature_names)), 
              [feature_names[i] for i in sorted_idx])
    plt.xlabel('Correlation with Predictions')
    plt.title('Feature Importance for House Price Prediction')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def plot_price_distribution(y_train, y_val, y_test, save_path=None):
    """
    Plot the distribution of house prices across train/val/test sets.
    
    Args:
        y_train: Training targets
        y_val: Validation targets  
        y_test: Test targets
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Convert to numpy if needed
    y_train_np = y_train.numpy() if hasattr(y_train, 'numpy') else y_train
    y_val_np = y_val.numpy() if hasattr(y_val, 'numpy') else y_val
    y_test_np = y_test.numpy() if hasattr(y_test, 'numpy') else y_test
    
    plt.subplot(1, 3, 1)
    plt.hist(y_train_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('House Price ($100K)')
    plt.ylabel('Frequency')
    plt.title(f'Training Set\nMean: ${y_train_np.mean():.2f}K, Std: ${y_train_np.std():.2f}K')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(y_val_np, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('House Price ($100K)')
    plt.ylabel('Frequency')
    plt.title(f'Validation Set\nMean: ${y_val_np.mean():.2f}K, Std: ${y_val_np.std():.2f}K')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(y_test_np, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('House Price ($100K)')
    plt.ylabel('Frequency')
    plt.title(f'Test Set\nMean: ${y_test_np.mean():.2f}K, Std: ${y_test_np.std():.2f}K')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Price distribution plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to run the house price prediction example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting House Price Prediction Example")
    
    # Load real-world data
    X, y, feature_names = load_california_housing_data()
    
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
    features, targets = data_handler.load_from_numpy(X, y.reshape(-1, 1))
    features, targets = data_handler.preprocess_data(features, targets)
    
    # Split data
    train_data, val_data, test_data = data_handler.split_data(features, targets)
    logger.info(f"Data split - Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[1])}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_handler.create_data_loaders(
        train_data, val_data, test_data, batch_size=64
    )
    
    # Configure model
    model_config = create_advanced_model_config(input_size=8)
    
    # Create model
    model = NeuralModel(model_config)
    logger.info(f"Created model with {model.get_parameter_count()} parameters")
    
    # Configure training
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=64,
        num_epochs=200,
        optimizer="adam",
        loss_function="mse",
        early_stopping_patience=20,
        save_best_model=True,
        model_save_path="models/house_price",
        log_save_path="logs/house_price"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/house_price_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        regression_metrics=["mse", "mae", "rmse", "r2", "mape", "smape"],
        plot_predictions=True,
        save_plots=True,
        output_dir="results/house_price"
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
        if isinstance(value, (int, float)):
            logger.info(f"  {metric.upper()}: {value:.4f}")
        else:
            logger.info(f"  {metric.upper()}: {value}")
    
    # Additional visualizations
    logger.info("Creating additional visualizations...")
    
    # Plot feature importance
    plot_feature_importance(model, feature_names, test_data[0], test_data[1], 
                          "results/house_price_feature_importance.png")
    
    # Plot price distribution
    plot_price_distribution(train_data[1], val_data[1], test_data[1],
                          "results/house_price_distribution.png")
    
    # Print some predictions
    logger.info("Sample predictions:")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        sample_features, sample_targets = sample_batch
        predictions = model(sample_features)
        
        # Denormalize if needed
        if data_handler.target_scaler:
            predictions = data_handler.target_scaler.inverse_transform(predictions.numpy())
            sample_targets = data_handler.target_scaler.inverse_transform(sample_targets.numpy())
        else:
            predictions = predictions.numpy()
            sample_targets = sample_targets.numpy()
        
        for i in range(min(5, len(predictions))):
            pred_price = predictions[i][0] * 100000  # Convert back to actual price
            actual_price = sample_targets[i][0] * 100000
            error = abs(pred_price - actual_price)
            logger.info(f"  Sample {i+1}: Predicted=${pred_price:,.0f}, Actual=${actual_price:,.0f}, Error=${error:,.0f}")
    
    logger.info("House price prediction example completed!")


if __name__ == "__main__":
    main()
