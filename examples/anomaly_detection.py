"""
Anomaly Detection Example

This example demonstrates how to use the neural framework for anomaly detection.
We'll create synthetic data with normal and anomalous patterns and train an autoencoder
to detect anomalies based on reconstruction error.
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


def generate_anomaly_data(n_normal: int = 800, n_anomalies: int = 200, n_features: int = 10) -> tuple:
    """
    Generate synthetic data with normal and anomalous patterns.
    
    Args:
        n_normal: Number of normal samples
        n_anomalies: Number of anomalous samples
        n_features: Number of features
        
    Returns:
        Tuple of (features, labels) where labels are 0 for normal, 1 for anomaly
    """
    np.random.seed(42)
    
    # Generate normal data (multivariate Gaussian)
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Generate anomalous data (different distribution)
    # Mix of different anomaly types
    n_anomalies_1 = n_anomalies // 2
    n_anomalies_2 = n_anomalies - n_anomalies_1
    
    # Type 1: Shifted mean
    anomaly_data_1 = np.random.multivariate_normal(
        mean=np.ones(n_features) * 2,  # Shifted mean
        cov=np.eye(n_features) * 0.5,  # Smaller variance
        size=n_anomalies_1
    )
    
    # Type 2: Different covariance structure
    cov_matrix = np.eye(n_features)
    cov_matrix[0, 1] = cov_matrix[1, 0] = 0.8  # High correlation
    anomaly_data_2 = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=cov_matrix,
        size=n_anomalies_2
    )
    
    # Combine all data
    all_data = np.vstack([normal_data, anomaly_data_1, anomaly_data_2])
    labels = np.hstack([
        np.zeros(n_normal),  # Normal
        np.ones(n_anomalies_1),  # Anomaly type 1
        np.ones(n_anomalies_2)   # Anomaly type 2
    ])
    
    # Shuffle the data
    indices = np.random.permutation(len(all_data))
    all_data = all_data[indices]
    labels = labels[indices]
    
    return all_data, labels


def create_autoencoder_config(input_size: int) -> ModelConfig:
    """
    Create configuration for an autoencoder model.
    
    Args:
        input_size: Size of input features
        
    Returns:
        ModelConfig for autoencoder
    """
    # Encoder: input -> hidden -> bottleneck
    # Decoder: bottleneck -> hidden -> input
    
    # Calculate hidden layer size (between input and bottleneck)
    hidden_size = max(input_size // 2, 8)
    bottleneck_size = max(input_size // 4, 4)
    
    # Create layer configurations for encoder and decoder
    layer_configs = [
        # Encoder layers
        {
            "type": "linear",
            "input_size": input_size,
            "output_size": hidden_size,
            "activation": "relu",
            "dropout_rate": 0.1
        },
        {
            "type": "linear", 
            "input_size": hidden_size,
            "output_size": bottleneck_size,
            "activation": "relu",
            "dropout_rate": 0.1
        },
        # Decoder layers
        {
            "type": "linear",
            "input_size": bottleneck_size,
            "output_size": hidden_size,
            "activation": "relu",
            "dropout_rate": 0.1
        },
        {
            "type": "linear",
            "input_size": hidden_size,
            "output_size": input_size,
            "activation": "none",  # No activation for reconstruction
            "dropout_rate": 0.0
        }
    ]
    
    return ModelConfig(
        input_size=input_size,
        output_size=input_size,  # Reconstruct input
        hidden_sizes=[hidden_size, bottleneck_size, hidden_size],
        activations=[ActivationType.RELU, ActivationType.RELU, ActivationType.RELU],
        dropout_rates=[0.1, 0.1, 0.1],
        output_activation=None,  # No activation for reconstruction
        use_batch_norm=True
    )


def main():
    """Main function to run the anomaly detection example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Anomaly Detection Example")
    
    # Generate synthetic data
    logger.info("Generating synthetic anomaly data...")
    X, y = generate_anomaly_data(n_normal=800, n_anomalies=200, n_features=10)
    logger.info(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Normal samples: {np.sum(y == 0)}, Anomalous samples: {np.sum(y == 1)}")
    
    # Split data into normal (for training) and all (for testing)
    normal_mask = y == 0
    normal_data = X[normal_mask]
    all_data = X
    all_labels = y
    
    # Configure data handling
    data_config = DatasetConfig(
        train_ratio=0.8,
        val_ratio=0.2,
        test_ratio=0.0,  # We'll use all data for testing
        normalize_features=True,
        normalize_targets=False,  # Don't normalize for autoencoder
        scaler_type="standard"
    )
    
    # Create data handler
    data_handler = DataHandler(data_config)
    
    # Load and preprocess normal data for training
    normal_features, _ = data_handler.load_from_numpy(normal_data, normal_data)  # Autoencoder: input = target
    normal_features, normal_targets = data_handler.preprocess_data(normal_features, normal_features)
    
    # Split normal data for training
    train_data, val_data, _ = data_handler.split_data(normal_features, normal_targets)
    logger.info(f"Normal data split - Train: {len(train_data[0])}, Val: {len(val_data[0])}")
    
    # Create data loaders for training
    train_loader, val_loader, _ = data_handler.create_data_loaders(
        train_data, val_data, None, batch_size=32
    )
    
    # Create test data loader with all data
    all_features, _ = data_handler.load_from_numpy(all_data, all_data)
    all_features, all_targets = data_handler.preprocess_data(all_features, all_features)
    test_dataset = data_handler.create_data_loaders(
        (all_features, all_targets), None, None, batch_size=32
    )[0]
    
    # Configure autoencoder model
    model_config = create_autoencoder_config(input_size=10)
    logger.info(f"Created autoencoder with {model_config.input_size} -> {model_config.output_size}")
    
    # Create model
    model = NeuralModel(model_config)
    logger.info(f"Created model with {model.get_parameter_count()} parameters")
    
    # Configure training
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=100,
        optimizer="adam",
        loss_function="mse",  # Reconstruction loss
        early_stopping_patience=10,
        save_best_model=True,
        model_save_path="models/anomaly_detection",
        log_save_path="logs/anomaly_detection"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/anomaly_detection_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        regression_metrics=["mse", "mae", "rmse"],
        plot_predictions=True,
        save_plots=True,
        output_dir="results/anomaly_detection"
    )
    
    # Create evaluator
    evaluator = Evaluator(model, eval_config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    report = evaluator.generate_report(
        test_dataset, 
        task_type="anomaly_detection"
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in report["metrics"].items():
        logger.info(f"  {metric.upper()}: {value:.4f}")
    
    # Visualize anomaly detection
    plot_anomaly_detection(model, all_data, all_labels, data_handler, 
                          "results/anomaly_detection_results.png")
    
    logger.info("Anomaly detection example completed!")


def plot_anomaly_detection(model, data, labels, data_handler, save_path=None):
    """Plot anomaly detection results."""
    # Get reconstruction errors
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for i in range(0, len(data), 32):  # Process in batches
            batch_data = data[i:i+32]
            batch_tensor = torch.FloatTensor(batch_data)
            
            # Normalize if scaler exists
            if data_handler.feature_scaler:
                batch_tensor = torch.FloatTensor(
                    data_handler.feature_scaler.transform(batch_tensor.numpy())
                )
            
            # Get reconstruction
            reconstruction = model(batch_tensor)
            
            # Calculate reconstruction error (MSE per sample)
            error = torch.mean((batch_tensor - reconstruction) ** 2, dim=1)
            reconstruction_errors.extend(error.numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    
    # Calculate threshold (95th percentile of normal data)
    normal_mask = labels == 0
    normal_errors = reconstruction_errors[normal_mask]
    threshold = np.percentile(normal_errors, 95)
    
    # Predict anomalies
    predicted_anomalies = reconstruction_errors > threshold
    
    # Calculate metrics
    true_anomalies = labels == 1
    accuracy = np.mean(predicted_anomalies == true_anomalies)
    precision = np.sum(predicted_anomalies & true_anomalies) / np.sum(predicted_anomalies)
    recall = np.sum(predicted_anomalies & true_anomalies) / np.sum(true_anomalies)
    f1 = 2 * precision * recall / (precision + recall)
    
    print(f"Anomaly Detection Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Threshold: {threshold:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reconstruction errors by class
    axes[0, 0].hist(reconstruction_errors[normal_mask], bins=30, alpha=0.7, 
                   label='Normal', color='blue', density=True)
    axes[0, 0].hist(reconstruction_errors[~normal_mask], bins=30, alpha=0.7, 
                   label='Anomaly', color='red', density=True)
    axes[0, 0].axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Reconstruction Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # ROC-like plot
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(true_anomalies, reconstruction_errors)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # Reconstruction errors over time
    axes[1, 0].plot(reconstruction_errors, alpha=0.7, label='Reconstruction Error')
    axes[1, 0].axhline(threshold, color='red', linestyle='--', label=f'Threshold')
    axes[1, 0].fill_between(range(len(reconstruction_errors)), 0, reconstruction_errors, 
                           where=labels==1, alpha=0.3, color='red', label='True Anomalies')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Reconstruction Error')
    axes[1, 0].set_title('Reconstruction Errors Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_anomalies, predicted_anomalies)
    
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Anomaly detection plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
