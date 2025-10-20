"""
Fraud Detection Example

This example demonstrates how to use the neural framework for fraud detection
using credit card transaction data. This showcases anomaly detection capabilities
with real-world financial fraud patterns.

Dataset: Credit Card Fraud Detection (from Kaggle)
Features: Transaction amounts, time, merchant categories, etc.
Target: Fraud detection (binary classification)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path so we can import the framework
sys.path.append(str(Path(__file__).parent.parent))

from neural_framework import (
    NeuralModel, ModelConfig, ActivationType,
    DataHandler, DatasetConfig,
    Trainer, TrainingConfig,
    Evaluator, EvaluationConfig,
    setup_logging, set_seed
)


def load_fraud_data():
    """
    Load and preprocess the Credit Card Fraud Detection dataset.
    
    Returns:
        Tuple of (features, target, feature_names)
    """
    print("Loading Credit Card Fraud Detection dataset...")
    
    # Try to load from local file first, then create synthetic data if not available
    try:
        # This would be the actual dataset - for demo purposes, we'll create realistic synthetic data
        # In practice, you would load: df = pd.read_csv('data/creditcard.csv')
        df = create_realistic_fraud_data()
    except FileNotFoundError:
        print("Creating realistic synthetic fraud detection data...")
        df = create_realistic_fraud_data()
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Fraud rate: {df['Class'].mean():.2%}")
    
    # Separate features and target
    target = df['Class'].values
    features_df = df.drop('Class', axis=1)
    feature_names = features_df.columns.tolist()
    
    features = features_df.values
    
    return features, target, feature_names


def create_realistic_fraud_data():
    """
    Create realistic synthetic fraud detection data based on real patterns.
    
    Returns:
        DataFrame with realistic fraud data
    """
    np.random.seed(42)
    n_samples = 284807  # Same size as real dataset
    fraud_rate = 0.001727  # Real fraud rate from dataset
    
    # Generate features similar to the real dataset
    # V1-V28 are PCA components (anonymized features)
    features = {}
    
    # Generate PCA-like features with different distributions
    for i in range(1, 29):
        if i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]:
            # Most features are normally distributed
            features[f'V{i}'] = np.random.normal(0, 1, n_samples)
        else:
            # Some features have different distributions
            features[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Time feature (seconds elapsed between each transaction and the first transaction)
    features['Time'] = np.random.exponential(1000, n_samples).astype(int)
    features['Time'] = np.clip(features['Time'], 0, 172792)
    
    # Amount feature (transaction amount)
    # Most transactions are small, but fraud tends to be larger amounts
    normal_amounts = np.random.lognormal(3, 1, int(n_samples * (1 - fraud_rate)))
    fraud_amounts = np.random.lognormal(5, 1.5, int(n_samples * fraud_rate))
    
    # Ensure we have exactly n_samples
    all_amounts = np.concatenate([normal_amounts, fraud_amounts])
    if len(all_amounts) > n_samples:
        all_amounts = all_amounts[:n_samples]
    elif len(all_amounts) < n_samples:
        # Pad with normal amounts if needed
        additional = np.random.lognormal(3, 1, n_samples - len(all_amounts))
        all_amounts = np.concatenate([all_amounts, additional])
    
    np.random.shuffle(all_amounts)
    features['Amount'] = all_amounts
    
    # Create fraud labels
    fraud_indices = np.random.choice(n_samples, int(n_samples * fraud_rate), replace=False)
    fraud_labels = np.zeros(n_samples)
    fraud_labels[fraud_indices] = 1
    
    # Modify features for fraud cases to make them more detectable
    for idx in fraud_indices:
        # Fraud transactions tend to have different patterns
        # Higher amounts
        features['Amount'][idx] *= np.random.uniform(2, 10)
        
        # Different time patterns (often at unusual hours)
        if np.random.random() < 0.7:
            features['Time'][idx] = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23]) * 3600 + np.random.randint(0, 3600)
        
        # Modify some PCA features to create anomalies
        anomaly_features = np.random.choice(range(1, 29), size=np.random.randint(3, 8), replace=False)
        for feat in anomaly_features:
            features[f'V{feat}'][idx] += np.random.normal(0, 3)  # Add anomaly
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['Class'] = fraud_labels.astype(int)
    
    return df


def create_fraud_model_config(input_size: int) -> ModelConfig:
    """
    Create model configuration optimized for fraud detection.
    
    Args:
        input_size: Number of input features
        
    Returns:
        ModelConfig for fraud detection
    """
    return ModelConfig(
        input_size=input_size,
        output_size=1,  # Binary classification
        hidden_sizes=[128, 64, 32, 16],  # Deep network for complex patterns
        activations=[
            ActivationType.RELU, 
            ActivationType.RELU, 
            ActivationType.RELU,
            ActivationType.RELU
        ],
        dropout_rates=[0.2, 0.3, 0.4, 0.3],  # Moderate dropout
        output_activation=ActivationType.SIGMOID,  # Sigmoid for binary classification
        use_batch_norm=True
    )


def plot_fraud_analysis(df, save_path=None):
    """
    Create comprehensive fraud analysis visualizations.
    
    Args:
        df: DataFrame with fraud data
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Fraud distribution over time
    df['Hour'] = (df['Time'] / 3600) % 24
    fraud_by_hour = df.groupby('Hour')['Class'].mean()
    axes[0, 0].plot(fraud_by_hour.index, fraud_by_hour.values, marker='o', linewidth=2)
    axes[0, 0].set_title('Fraud Rate by Hour of Day')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Fraud Rate')
    axes[0, 0].grid(True)
    
    # 2. Amount distribution by class
    fraud_amounts = df[df['Class'] == 1]['Amount']
    normal_amounts = df[df['Class'] == 0]['Amount']
    
    axes[0, 1].hist(normal_amounts, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    axes[0, 1].hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    axes[0, 1].set_title('Transaction Amount Distribution')
    axes[0, 1].set_xlabel('Amount ($)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # 3. Feature correlation with fraud
    corr_with_fraud = df.corr()['Class'].drop('Class').abs().sort_values(ascending=True)
    top_features = corr_with_fraud.tail(10)
    axes[0, 2].barh(range(len(top_features)), top_features.values, color='skyblue')
    axes[0, 2].set_yticks(range(len(top_features)))
    axes[0, 2].set_yticklabels(top_features.index)
    axes[0, 2].set_title('Top 10 Features Correlated with Fraud')
    axes[0, 2].set_xlabel('Absolute Correlation')
    
    # 4. Fraud rate by amount ranges
    df['AmountRange'] = pd.cut(df['Amount'], bins=[0, 10, 50, 100, 500, 1000, float('inf')], 
                              labels=['0-10', '10-50', '50-100', '100-500', '500-1000', '1000+'])
    amount_fraud = df.groupby('AmountRange')['Class'].mean()
    axes[1, 0].bar(amount_fraud.index, amount_fraud.values, color='red', alpha=0.7)
    axes[1, 0].set_title('Fraud Rate by Amount Range')
    axes[1, 0].set_xlabel('Amount Range ($)')
    axes[1, 0].set_ylabel('Fraud Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Feature V1 vs V2 scatter plot
    normal_data = df[df['Class'] == 0]
    fraud_data = df[df['Class'] == 1]
    
    axes[1, 1].scatter(normal_data['V1'], normal_data['V2'], alpha=0.5, s=1, label='Normal', color='blue')
    axes[1, 1].scatter(fraud_data['V1'], fraud_data['V2'], alpha=0.8, s=10, label='Fraud', color='red')
    axes[1, 1].set_title('V1 vs V2 (First Two PCA Components)')
    axes[1, 1].set_xlabel('V1')
    axes[1, 1].set_ylabel('V2')
    axes[1, 1].legend()
    
    # 6. Class distribution
    class_counts = df['Class'].value_counts()
    axes[1, 2].pie(class_counts.values, labels=['Normal', 'Fraud'], autopct='%1.2f%%', 
                   colors=['lightblue', 'red'], startangle=90)
    axes[1, 2].set_title('Class Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fraud analysis plot saved to {save_path}")
    
    plt.show()


def plot_fraud_detection_results(y_true, y_pred_proba, save_path=None):
    """
    Plot fraud detection results and performance metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = np.trapz(tpr, fpr)
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = np.trapz(precision, recall)
    axes[0, 1].plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True)
    
    # Confusion Matrix
    y_pred = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Prediction probability distribution
    axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Normal', color='blue', density=True)
    axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Fraud', color='red', density=True)
    axes[1, 1].axvline(0.5, color='black', linestyle='--', label='Threshold')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fraud detection results plot saved to {save_path}")
    
    plt.show()


def calculate_fraud_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate comprehensive fraud detection metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Basic metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Fraud-specific metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Financial impact (assuming average fraud amount)
    fraud_amount = 1000  # Average fraud amount
    total_fraud_amount = np.sum(y_true) * fraud_amount
    detected_fraud_amount = tp * fraud_amount
    fraud_detection_rate = detected_fraud_amount / total_fraud_amount if total_fraud_amount > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'fraud_detection_rate': fraud_detection_rate,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def main():
    """Main function to run the fraud detection example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Fraud Detection Example")
    
    # Load real-world data
    X, y, feature_names = load_fraud_data()
    
    # Configure data handling
    data_config = DatasetConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize_features=True,
        normalize_targets=False,  # Binary classification doesn't need target normalization
        scaler_type="standard"
    )
    
    # Create data handler
    data_handler = DataHandler(data_config)
    
    # Load and preprocess data
    features, targets = data_handler.load_from_numpy(X, y.reshape(-1, 1))
    features, targets = data_handler.preprocess_data(features, targets)
    
    # Split data
    train_data, val_data, test_data = data_handler.split_data(features, targets)
    logger.info(f"Data split - Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_handler.create_data_loaders(
        train_data, val_data, test_data, batch_size=512
    )
    
    # Configure model
    model_config = create_fraud_model_config(input_size=30)  # 30 features (V1-V28, Time, Amount)
    
    # Create model
    model = NeuralModel(model_config)
    logger.info(f"Created model with {model.get_parameter_count()} parameters")
    
    # Configure training
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=512,
        num_epochs=100,
        optimizer="adam",
        loss_function="bce",  # Binary cross-entropy for fraud detection
        early_stopping_patience=15,
        save_best_model=True,
        model_save_path="models/fraud_detection",
        log_save_path="logs/fraud_detection"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/fraud_detection_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        classification_metrics=["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"],
        plot_predictions=True,
        save_plots=True,
        output_dir="results/fraud_detection"
    )
    
    # Create evaluator
    evaluator = Evaluator(model, eval_config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    report = evaluator.generate_report(
        test_loader, 
        task_type="classification",
        num_classes=2
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in report["metrics"].items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric.upper()}: {value:.4f}")
        else:
            logger.info(f"  {metric.upper()}: {value}")
    
    # Additional analysis
    logger.info("Creating additional analysis...")
    
    # Get predictions for analysis
    model.eval()
    y_true = []
    y_pred_proba = []
    
    with torch.no_grad():
        for batch in test_loader:
            features_batch, targets_batch = batch
            predictions = model(features_batch)
            y_true.extend(targets_batch.numpy().flatten())
            y_pred_proba.extend(predictions.numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Create comprehensive visualizations
    df_analysis = pd.DataFrame(X, columns=feature_names)
    df_analysis['Class'] = y
    
    plot_fraud_analysis(df_analysis, "results/fraud_analysis.png")
    plot_fraud_detection_results(y_true, y_pred_proba, "results/fraud_detection_results.png")
    
    # Business insights
    logger.info("Business Insights:")
    fraud_rate = y.mean()
    logger.info(f"  Overall fraud rate: {fraud_rate:.2%}")
    
    # Calculate fraud-specific metrics
    fraud_metrics = calculate_fraud_metrics(y_true, y_pred_proba)
    
    logger.info(f"  Fraud detection rate: {fraud_metrics['fraud_detection_rate']:.2%}")
    logger.info(f"  False positive rate: {fraud_metrics['false_positive_rate']:.2%}")
    logger.info(f"  False negative rate: {fraud_metrics['false_negative_rate']:.2%}")
    
    # High-risk transactions
    high_risk = np.sum(y_pred_proba > 0.8)
    logger.info(f"  High-risk transactions (prob > 0.8): {high_risk} ({high_risk/len(y_pred_proba):.1%})")
    
    # Model performance summary
    accuracy = np.mean((y_pred_proba > 0.5) == y_true)
    logger.info(f"  Model accuracy: {accuracy:.2%}")
    
    # Financial impact analysis
    total_transactions = len(y_true)
    fraud_transactions = np.sum(y_true)
    detected_fraud = np.sum((y_pred_proba > 0.5) & (y_true == 1))
    
    logger.info(f"  Total transactions analyzed: {total_transactions:,}")
    logger.info(f"  Fraud transactions: {fraud_transactions:,}")
    logger.info(f"  Detected fraud: {detected_fraud:,}")
    logger.info(f"  Detection rate: {detected_fraud/fraud_transactions:.2%}")
    
    logger.info("Fraud detection example completed!")


if __name__ == "__main__":
    main()
