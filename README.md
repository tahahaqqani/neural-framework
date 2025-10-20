# Neural Framework

A flexible, domain-agnostic neural network framework built on PyTorch that can be easily adapted for various machine learning tasks.

## Features

- **Generic Architecture**: Configurable neural networks for any input/output dimensions
- **Multiple Domains**: Built-in support for regression, classification, time series, NLP, and more
- **Flexible Data Handling**: Support for various data types and preprocessing
- **Comprehensive Training**: Multiple optimizers, schedulers, and loss functions
- **Rich Evaluation**: Extensive metrics and visualization tools
- **Easy Configuration**: YAML/JSON configuration management
- **Production Ready**: Model saving, loading, and deployment utilities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd neural-framework

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pyyaml
```

## Quick Start

### Simple Regression Example

```python
from neural_framework import (
    NeuralModel, ModelConfig, ActivationType,
    DataHandler, DatasetConfig,
    Trainer, TrainingConfig,
    Evaluator, EvaluationConfig
)

# 1. Configure model
model_config = ModelConfig(
    input_size=10,
    output_size=1,
    hidden_sizes=[64, 32],
    activations=[ActivationType.RELU, ActivationType.RELU],
    dropout_rates=[0.1, 0.1]
)

# 2. Create model
model = NeuralModel(model_config)

# 3. Configure data handling
data_config = DatasetConfig(
    normalize_features=True,
    normalize_targets=True
)

# 4. Load and preprocess data
data_handler = DataHandler(data_config)
features, targets = data_handler.load_from_numpy(X, y)
features, targets = data_handler.preprocess_data(features, targets)

# 5. Split data and create loaders
train_data, val_data, test_data = data_handler.split_data(features, targets)
train_loader, val_loader, test_loader = data_handler.create_data_loaders(
    train_data, val_data, test_data
)

# 6. Configure training
training_config = TrainingConfig(
    learning_rate=0.001,
    num_epochs=100,
    optimizer="adam",
    loss_function="mse"
)

# 7. Train model
trainer = Trainer(model, training_config)
history = trainer.train(train_loader, val_loader)

# 8. Evaluate model
eval_config = EvaluationConfig(regression_metrics=["mse", "mae", "r2"])
evaluator = Evaluator(model, eval_config)
report = evaluator.generate_report(test_loader, task_type="regression")
```

## Architecture

### Core Components

1. **NeuralModel**: Generic neural network with configurable architecture
2. **DataHandler**: Flexible data loading and preprocessing
3. **Trainer**: Comprehensive training utilities
4. **Evaluator**: Rich evaluation and visualization tools

### Model Configuration

```python
model_config = ModelConfig(
    input_size=784,           # Input dimension
    output_size=10,           # Output dimension
    hidden_sizes=[256, 128],  # Hidden layer sizes
    activations=[             # Activation functions
        ActivationType.RELU,
        ActivationType.RELU
    ],
    dropout_rates=[0.2, 0.1], # Dropout rates
    use_batch_norm=True,      # Batch normalization
    residual_connections=True # Residual connections
)
```

### Advanced Layer Configuration

```python
from neural_framework.core import LayerConfig, LayerType

# Custom layer configuration
layer_configs = [
    LayerConfig(
        type=LayerType.LINEAR,
        input_size=784,
        output_size=256,
        activation=ActivationType.RELU,
        dropout_rate=0.2
    ),
    LayerConfig(
        type=LayerType.LSTM,
        input_size=256,
        hidden_size=128,
        num_layers=2,
        bidirectional=True
    ),
    LayerConfig(
        type=LayerType.LINEAR,
        input_size=256,  # 128 * 2 for bidirectional
        output_size=10,
        activation=ActivationType.SOFTMAX
    )
]

model_config = ModelConfig(
    input_size=784,
    output_size=10,
    layer_configs=layer_configs
)
```

## Supported Tasks

### 1. Regression

- **Use Cases**: Price prediction, demand forecasting, sensor readings
- **Metrics**: MSE, MAE, RMSE, R², MAPE
- **Example**: `examples/simple_regression.py`

### 2. Classification

- **Use Cases**: Image classification, sentiment analysis, fraud detection
- **Metrics**: Accuracy, Precision, Recall, F1, AUC
- **Example**: `examples/classification.py`

### 3. Time Series Forecasting

- **Use Cases**: Stock prices, weather prediction, sales forecasting
- **Metrics**: Time series specific metrics, multi-step evaluation
- **Example**: `examples/time_series.py`

### 4. Anomaly Detection

- **Use Cases**: Fraud detection, equipment monitoring, network security
- **Metrics**: Anomaly rate, reconstruction error, threshold analysis
- **Example**: `examples/anomaly_detection.py`

### 5. Natural Language Processing

- **Use Cases**: Text classification, sentiment analysis, language modeling
- **Metrics**: NLP specific metrics, perplexity, BLEU
- **Example**: `examples/nlp_sentiment.py`

### 6. Recommendation Systems

- **Use Cases**: Collaborative filtering, content-based filtering
- **Metrics**: Precision@K, Recall@K, NDCG, Hit Rate
- **Example**: `examples/recommendation_system.py`

## Configuration

### Training Configuration

```python
training_config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    num_epochs=100,
    optimizer="adam",           # adam, sgd, rmsprop, adamw
    scheduler="plateau",        # none, step, cosine, plateau, exponential
    loss_function="mse",        # mse, mae, cross_entropy, bce, nll
    early_stopping_patience=10,
    gradient_clip_norm=1.0,
    mixed_precision=True
)
```

### Data Configuration

```python
data_config = DatasetConfig(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    normalize_features=True,
    normalize_targets=True,
    scaler_type="standard",     # standard, minmax, none
    augment_data=True,
    augmentation_factor=1.5
)
```

## Evaluation and Visualization

### Regression Evaluation

```python
eval_config = EvaluationConfig(
    regression_metrics=["mse", "mae", "rmse", "r2"],
    plot_predictions=True,
    save_plots=True
)

evaluator = Evaluator(model, eval_config)
report = evaluator.generate_report(test_loader, task_type="regression")
```

### Classification Evaluation

```python
eval_config = EvaluationConfig(
    classification_metrics=["accuracy", "precision", "recall", "f1", "auc"],
    plot_confusion_matrix=True,
    classification_threshold=0.5
)

evaluator = Evaluator(model, eval_config)
report = evaluator.generate_report(test_loader, task_type="classification")
```

## Examples

### Basic Examples

#### 1. Simple Regression

```bash
python examples/simple_regression.py
```

#### 2. Simple Classification

```bash
python examples/simple_classification.py
```

#### 3. Time Series Forecasting

```bash
python examples/time_series_forecasting.py
```

#### 4. Anomaly Detection

```bash
python examples/anomaly_detection.py
```

### Real-World Applications

#### 1. House Price Prediction

**California Housing Dataset** - Real estate market analysis

```bash
python examples/house_price_prediction.py
```

- **Dataset**: 20,640 California housing records
- **Features**: Median income, house age, rooms, population, location
- **Business Value**: Real estate investment, urban planning, market analysis
- **Key Insights**: Feature importance analysis, price distribution patterns

#### 2. Customer Churn Prediction

**Telco Customer Churn Dataset** - Customer retention analysis

```bash
python examples/customer_churn_prediction.py
```

- **Dataset**: 7,043 telecom customer records
- **Features**: Demographics, account info, services, charges
- **Business Value**: Customer retention, marketing optimization, revenue protection
- **Key Insights**: Churn patterns, risk factors, retention strategies

#### 3. Stock Price Forecasting

**Real Market Data** - Financial market prediction

```bash
python examples/stock_price_forecasting.py
```

- **Dataset**: Real stock data (AAPL, GOOGL, MSFT, etc.)
- **Features**: OHLCV data, technical indicators (RSI, MACD, Bollinger Bands)
- **Business Value**: Algorithmic trading, portfolio management, risk assessment
- **Key Insights**: Volatility analysis, trading signals, risk metrics

#### 4. Fraud Detection

**Credit Card Fraud Dataset** - Financial fraud prevention

```bash
python examples/fraud_detection.py
```

- **Dataset**: 284,807 credit card transactions
- **Features**: Anonymized PCA components, transaction amounts, time patterns
- **Business Value**: Fraud prevention, financial security, cost reduction
- **Key Insights**: Fraud patterns, detection rates, financial impact analysis

## Production Usage

### Model Saving and Loading

```python
# Save model
model.save_model("models/my_model.pth")

# Load model
loaded_model = NeuralModel.load_model("models/my_model.pth")

# Save preprocessors
data_handler.save_preprocessors("models/preprocessors.pkl")

# Load preprocessors
data_handler.load_preprocessors("models/preprocessors.pkl")
```

### Configuration Management

```python
# Save configuration
config_dict = model_config.to_dict()
save_config(config_dict, "configs/model_config.yaml")

# Load configuration
config_dict = load_config("configs/model_config.yaml")
model_config = ModelConfig.from_dict(config_dict)
```

## Advanced Features

### Custom Loss Functions

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, predictions, targets):
        return 0.7 * self.mse(predictions, targets) + 0.3 * self.mae(predictions, targets)
```

### Callbacks

```python
from neural_framework.training import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        if logs['val_loss'] < 0.1:
            print(f"Great performance at epoch {epoch}!")

# Use in training
callbacks = [CustomCallback()]
trainer = Trainer(model, training_config, callbacks)
```

### Mixed Precision Training

```python
training_config = TrainingConfig(
    mixed_precision=True,  # Enable mixed precision
    # ... other config
)
```

## API Reference

### Core Classes

- `NeuralModel`: Main neural network class
- `ModelConfig`: Model configuration
- `DataHandler`: Data processing utilities
- `Trainer`: Training utilities
- `Evaluator`: Evaluation utilities

### Configuration Classes

- `TrainingConfig`: Training parameters
- `DatasetConfig`: Data processing parameters
- `EvaluationConfig`: Evaluation parameters

### Utility Functions

- `set_seed()`: Set random seed
- `setup_logging()`: Configure logging
- `get_device()`: Get appropriate device
- `count_parameters()`: Count model parameters

---
