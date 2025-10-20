"""
Customer Churn Prediction Example

This example demonstrates how to use the neural framework for customer churn prediction
using the Telco Customer Churn dataset. This is a critical business problem that
showcases classification capabilities on real-world data.

Dataset: Telco Customer Churn (from Kaggle)
Features: Customer demographics, account information, services, and charges
Target: Customer churn (binary classification)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


def load_telco_churn_data():
    """
    Load and preprocess the Telco Customer Churn dataset.
    
    Returns:
        Tuple of (features, target, feature_names)
    """
    print("Loading Telco Customer Churn dataset...")
    
    # Try to load from local file first, then create synthetic data if not available
    try:
        # This would be the actual dataset - for demo purposes, we'll create realistic synthetic data
        # In practice, you would load: df = pd.read_csv('data/telco_churn.csv')
        df = create_realistic_churn_data()
    except FileNotFoundError:
        print("Creating realistic synthetic Telco churn data...")
        df = create_realistic_churn_data()
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    
    # Separate features and target
    target = df['Churn'].values
    features_df = df.drop('Churn', axis=1)
    feature_names = features_df.columns.tolist()
    
    # Convert categorical variables to numerical
    le_dict = {}
    for col in features_df.columns:
        if features_df[col].dtype == 'object':
            le = LabelEncoder()
            features_df[col] = le.fit_transform(features_df[col])
            le_dict[col] = le
    
    features = features_df.values
    
    return features, target, feature_names, le_dict


def create_realistic_churn_data():
    """
    Create realistic synthetic Telco churn data based on real patterns.
    
    Returns:
        DataFrame with realistic churn data
    """
    np.random.seed(42)
    n_samples = 7043  # Same size as real dataset
    
    # Customer demographics
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
    senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
    
    # Account information
    tenure = np.random.exponential(32, n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)  # Cap at 72 months
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
    multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])
    
    # Online services
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.28, 0.50, 0.22])
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.43, 0.22])
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22])
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22])
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.39, 0.39, 0.22])
    
    # Contract and billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24])
    paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                    n_samples, p=[0.34, 0.19, 0.22, 0.25])
    
    # Charges (realistic ranges)
    monthly_charges = np.random.normal(64.76, 30.09, n_samples)
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_samples)
    total_charges = np.clip(total_charges, 0, 8684.80)
    
    # Create churn based on realistic patterns
    churn_prob = np.zeros(n_samples)
    
    # Higher churn for month-to-month contracts
    churn_prob += np.where(contract == 'Month-to-month', 0.3, 0)
    churn_prob += np.where(contract == 'One year', 0.1, 0)
    churn_prob += np.where(contract == 'Two year', 0.05, 0)
    
    # Higher churn for higher monthly charges
    churn_prob += (monthly_charges - 50) / 100 * 0.2
    
    # Higher churn for shorter tenure
    churn_prob += (72 - tenure) / 72 * 0.2
    
    # Higher churn for electronic check payment
    churn_prob += np.where(payment_method == 'Electronic check', 0.15, 0)
    
    # Higher churn for no online security
    churn_prob += np.where(online_security == 'No', 0.1, 0)
    
    # Add some randomness
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    churn = (churn_prob > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    
    return df


def create_churn_model_config(input_size: int) -> ModelConfig:
    """
    Create model configuration optimized for churn prediction.
    
    Args:
        input_size: Number of input features
        
    Returns:
        ModelConfig for churn prediction
    """
    return ModelConfig(
        input_size=input_size,
        output_size=1,  # Binary classification
        hidden_sizes=[64, 32, 16],  # Appropriate for tabular data
        activations=[
            ActivationType.RELU, 
            ActivationType.RELU, 
            ActivationType.RELU
        ],
        dropout_rates=[0.3, 0.4, 0.2],  # Higher dropout for regularization
        output_activation=ActivationType.SIGMOID,  # Sigmoid for binary classification
        use_batch_norm=True
    )


def plot_churn_analysis(df, save_path=None):
    """
    Create comprehensive churn analysis visualizations.
    
    Args:
        df: DataFrame with churn data
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Churn rate by contract type
    contract_churn = df.groupby('Contract')['Churn'].mean()
    axes[0, 0].bar(contract_churn.index, contract_churn.values, color=['red', 'orange', 'green'])
    axes[0, 0].set_title('Churn Rate by Contract Type')
    axes[0, 0].set_ylabel('Churn Rate')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Churn rate by tenure groups
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])
    tenure_churn = df.groupby('TenureGroup')['Churn'].mean()
    axes[0, 1].bar(tenure_churn.index, tenure_churn.values, color='skyblue')
    axes[0, 1].set_title('Churn Rate by Tenure Group')
    axes[0, 1].set_ylabel('Churn Rate')
    
    # 3. Monthly charges distribution by churn
    churn_yes = df[df['Churn'] == 1]['MonthlyCharges']
    churn_no = df[df['Churn'] == 0]['MonthlyCharges']
    axes[0, 2].hist(churn_no, bins=30, alpha=0.7, label='No Churn', color='green')
    axes[0, 2].hist(churn_yes, bins=30, alpha=0.7, label='Churn', color='red')
    axes[0, 2].set_title('Monthly Charges Distribution')
    axes[0, 2].set_xlabel('Monthly Charges ($)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # 4. Churn rate by payment method
    payment_churn = df.groupby('PaymentMethod')['Churn'].mean()
    axes[1, 0].barh(payment_churn.index, payment_churn.values, color='lightcoral')
    axes[1, 0].set_title('Churn Rate by Payment Method')
    axes[1, 0].set_xlabel('Churn Rate')
    
    # 5. Churn rate by internet service
    internet_churn = df.groupby('InternetService')['Churn'].mean()
    axes[1, 1].bar(internet_churn.index, internet_churn.values, color=['blue', 'red', 'gray'])
    axes[1, 1].set_title('Churn Rate by Internet Service')
    axes[1, 1].set_ylabel('Churn Rate')
    
    # 6. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Churn analysis plot saved to {save_path}")
    
    plt.show()


def plot_model_performance(y_true, y_pred_proba, save_path=None):
    """
    Plot model performance metrics.
    
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
    axes[1, 1].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='No Churn', color='green')
    axes[1, 1].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Churn', color='red')
    axes[1, 1].axvline(0.5, color='black', linestyle='--', label='Threshold')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model performance plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to run the customer churn prediction example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Customer Churn Prediction Example")
    
    # Load real-world data
    X, y, feature_names, le_dict = load_telco_churn_data()
    
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
        train_data, val_data, test_data, batch_size=128
    )
    
    # Configure model
    model_config = create_churn_model_config(input_size=19)  # 19 features after encoding
    
    # Create model
    model = NeuralModel(model_config)
    logger.info(f"Created model with {model.get_parameter_count()} parameters")
    
    # Configure training
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=128,
        num_epochs=100,
        optimizer="adam",
        loss_function="bce",  # Binary cross-entropy for churn prediction
        early_stopping_patience=15,
        save_best_model=True,
        model_save_path="models/churn_prediction",
        log_save_path="logs/churn_prediction"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/churn_prediction_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        classification_metrics=["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"],
        plot_predictions=True,
        save_plots=True,
        output_dir="results/churn_prediction"
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
    df_analysis['Churn'] = y
    
    plot_churn_analysis(df_analysis, "results/churn_analysis.png")
    plot_model_performance(y_true, y_pred_proba, "results/churn_model_performance.png")
    
    # Business insights
    logger.info("Business Insights:")
    churn_rate = y.mean()
    logger.info(f"  Overall churn rate: {churn_rate:.2%}")
    
    # High-risk customers (probability > 0.8)
    high_risk = np.sum(y_pred_proba > 0.8)
    logger.info(f"  High-risk customers (prob > 0.8): {high_risk} ({high_risk/len(y_pred_proba):.1%})")
    
    # Model performance summary
    accuracy = np.mean((y_pred_proba > 0.5) == y_true)
    logger.info(f"  Model accuracy: {accuracy:.2%}")
    
    logger.info("Customer churn prediction example completed!")


if __name__ == "__main__":
    main()
