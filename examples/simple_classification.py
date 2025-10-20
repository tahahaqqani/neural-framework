"""
Simple Classification Example

This example demonstrates how to use the neural framework for a basic classification task.
We'll create synthetic data and train a neural network to classify data points.
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


def generate_synthetic_classification_data(n_samples: int = 1000, n_classes: int = 3) -> tuple:
    """
    Generate synthetic classification data.
    
    Args:
        n_samples: Number of samples to generate
        n_classes: Number of classes
        
    Returns:
        Tuple of (features, targets)
    """
    np.random.seed(42)
    
    # Generate features (2D input)
    X = np.random.randn(n_samples, 2)
    
    # Generate targets based on distance from origin and some noise
    distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    angles = np.arctan2(X[:, 1], X[:, 0])
    
    # Create class boundaries
    if n_classes == 2:
        # Binary classification: inner circle vs outer ring
        y = (distances > 1.0).astype(int)
    else:
        # Multi-class: divide by angle
        y = (angles + np.pi) / (2 * np.pi) * n_classes
        y = np.floor(y).astype(int)
        y = np.clip(y, 0, n_classes - 1)
    
    return X, y


def main():
    """Main function to run the classification example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Simple Classification Example")
    
    # Generate synthetic data
    logger.info("Generating synthetic classification data...")
    X, y = generate_synthetic_classification_data(n_samples=1000, n_classes=3)
    logger.info(f"Generated {X.shape[0]} samples with {X.shape[1]} features and {len(np.unique(y))} classes")
    
    # Configure data handling
    data_config = DatasetConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize_features=True,
        normalize_targets=False,  # Don't normalize class labels
        scaler_type="standard"
    )
    
    # Create data handler
    data_handler = DataHandler(data_config)
    
    # Load and preprocess data
    features, targets = data_handler.load_from_numpy(X, y.reshape(-1, 1))
    features, targets = data_handler.preprocess_data(features, targets)
    
    # Convert targets to long tensor for classification
    targets = targets.long().squeeze()
    
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
        output_size=3,  # 3 classes
        hidden_sizes=[64, 32],
        activations=[ActivationType.RELU, ActivationType.RELU],
        dropout_rates=[0.2, 0.1],
        output_activation=None,  # No activation for raw logits
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
        loss_function="cross_entropy",
        early_stopping_patience=10,
        save_best_model=True,
        model_save_path="models/classification",
        log_save_path="logs/classification"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/classification_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        classification_metrics=["accuracy", "precision", "recall", "f1"],
        plot_predictions=True,
        plot_confusion_matrix=True,
        save_plots=True,
        output_dir="results/classification"
    )
    
    # Create evaluator
    evaluator = Evaluator(model, eval_config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    report = evaluator.generate_report(
        test_loader, 
        task_type="classification",
        num_classes=3,
        class_names=["Class 0", "Class 1", "Class 2"]
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in report["metrics"].items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Visualize decision boundary
    plot_decision_boundary(model, X, y, "results/classification_decision_boundary.png")
    
    logger.info("Simple classification example completed!")


def plot_decision_boundary(model, X, y, save_path=None):
    """Plot decision boundary for 2D classification."""
    # Create a mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.FloatTensor(mesh_points)
    
    model.eval()
    with torch.no_grad():
        Z = model(mesh_tensor)
        Z = torch.argmax(Z, dim=1).numpy()
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision boundary plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
