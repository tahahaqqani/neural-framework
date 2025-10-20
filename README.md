# Neural Framework

A flexible PyTorch-based neural network framework for machine learning tasks including regression, classification, time series forecasting, and anomaly detection.

## Quick Start

```bash
# Clone and install
git clone https://github.com/tahahaqqani/neural-framework.git
cd neural-framework
pip install -r requirements.txt

# Run examples
python examples/simple_regression.py
python examples/simple_classification.py
```

## Features

- **Multi-task Support**: Regression, classification, time series, anomaly detection
- **Easy Configuration**: Simple config classes for models, training, and data
- **Built-in Evaluation**: Comprehensive metrics and visualizations
- **Production Ready**: Model saving/loading and deployment utilities

## Architecture

```python
from neural_framework import NeuralModel, ModelConfig, DataHandler, Trainer, Evaluator

# 1. Configure model
model_config = ModelConfig(
    input_size=10,
    output_size=1,
    hidden_sizes=[64, 32],
    activations=['relu', 'relu']
)

# 2. Create and train
model = NeuralModel(model_config)
trainer = Trainer(model, training_config)
trainer.train(train_loader, val_loader)

# 3. Evaluate
evaluator = Evaluator(model, eval_config)
report = evaluator.generate_report(test_loader, task_type="regression")
```

## Examples

### Real-World Applications

| Example                        | Dataset            | Task              | Description                                     |
| ------------------------------ | ------------------ | ----------------- | ----------------------------------------------- |
| `house_price_prediction.py`    | California Housing | Regression        | Predict house prices using demographic features |
| `customer_churn_prediction.py` | Telco Churn        | Classification    | Predict customer churn for retention strategies |
| `stock_price_forecasting.py`   | Real Market Data   | Time Series       | Forecast stock prices with technical indicators |
| `fraud_detection.py`           | Credit Card Fraud  | Anomaly Detection | Detect fraudulent transactions                  |

### Basic Examples

- `simple_regression.py` - Basic regression example
- `simple_classification.py` - Basic classification example
- `time_series_forecasting.py` - Time series prediction
- `anomaly_detection.py` - Anomaly detection

## Configuration

### Model Configuration

```python
ModelConfig(
    input_size=784,
    output_size=10,
    hidden_sizes=[256, 128],
    activations=['relu', 'relu'],
    dropout_rates=[0.2, 0.1],
    use_batch_norm=True
)
```

### Training Configuration

```python
TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    num_epochs=100,
    optimizer="adam",
    loss_function="mse",
    early_stopping_patience=10
)
```

## Evaluation

### Supported Metrics

- **Regression**: MSE, MAE, RMSE, R², MAPE
- **Classification**: Accuracy, Precision, Recall, F1, AUC
- **Time Series**: MAE, RMSE, MAPE, directional accuracy
- **Anomaly Detection**: Precision, Recall, F1, AUC

### Visualization

- Training history plots
- Prediction vs actual scatter plots
- Confusion matrices
- Feature importance analysis
- Time series forecasting plots

## Production Usage

```python
# Save model and preprocessors
model.save_model("models/my_model.pth")
data_handler.save_preprocessors("models/preprocessors.pkl")

# Load for inference
loaded_model = NeuralModel.load_model("models/my_model.pth")
data_handler.load_preprocessors("models/preprocessors.pkl")
```

## Project Structure

```
neural-framework/
├── neural_framework/          # Core framework
│   ├── core.py               # NeuralModel, ModelConfig
│   ├── data.py               # DataHandler, DatasetConfig
│   ├── training.py           # Trainer, TrainingConfig
│   ├── evaluation.py         # Evaluator, EvaluationConfig
│   └── utils.py              # Utility functions
├── examples/                 # Example scripts
├── models/                   # Saved models
├── results/                  # Evaluation results and plots
└── requirements.txt          # Dependencies
```

## Installation

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pyyaml
```

## License

This project is open source and available under the [MIT License](LICENSE).

---

**Repository**: https://github.com/tahahaqqani/neural-framework
