"""
Stock Price Forecasting Example

This example demonstrates how to use the neural framework for stock price forecasting
using real market data. This showcases time series forecasting capabilities with
real-world financial data.

Dataset: Real stock price data (AAPL, GOOGL, MSFT, etc.)
Features: Historical prices, technical indicators, volume
Target: Future stock price (regression)
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import yfinance as yf
from datetime import datetime, timedelta
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


def load_stock_data(symbol="AAPL", period="2y", interval="1d"):
    """
    Load real stock data using yfinance.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')
        period: Data period ('1y', '2y', '5y', 'max')
        interval: Data interval ('1d', '1h', '5m')
        
    Returns:
        Tuple of (features, targets, feature_names, dates)
    """
    print(f"Loading {symbol} stock data for {period}...")
    
    try:
        # Download stock data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
            
        print(f"Loaded {len(data)} data points from {data.index[0].date()} to {data.index[-1].date()}")
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Creating realistic synthetic stock data...")
        data = create_realistic_stock_data(symbol, period, interval)
    
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Prepare features and targets
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26',
        'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
        'BB_upper', 'BB_middle', 'BB_lower',
        'ATR', 'Stochastic_K', 'Stochastic_D',
        'Williams_R', 'CCI', 'ROC'
    ]
    
    # Remove any columns that might be missing
    available_columns = [col for col in feature_columns if col in data.columns]
    feature_data = data[available_columns].fillna(method='ffill').fillna(method='bfill')
    
    # Create sequences for time series
    sequence_length = 30  # Use 30 days to predict next day
    target_column = 'Close'
    
    # Create sequences
    sequences = []
    targets = []
    dates = []
    
    for i in range(sequence_length, len(feature_data)):
        # Features: past 30 days of all indicators
        seq = feature_data.iloc[i-sequence_length:i][available_columns].values
        sequences.append(seq.flatten())  # Flatten to 1D for neural network
        
        # Target: next day's closing price
        target = data.iloc[i][target_column]
        targets.append(target)
        dates.append(data.index[i])
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    dates = np.array(dates)
    
    # Feature names
    feature_names = [f"{col}_t-{j}" for col in available_columns for j in range(sequence_length-1, -1, -1)]
    
    print(f"Created {len(sequences)} sequences with {len(feature_names)} features each")
    print(f"Price range: ${targets.min():.2f} - ${targets.max():.2f}")
    
    return sequences, targets, feature_names, dates


def create_realistic_stock_data(symbol, period, interval):
    """
    Create realistic synthetic stock data if real data is not available.
    
    Args:
        symbol: Stock symbol
        period: Data period
        interval: Data interval
        
    Returns:
        DataFrame with realistic stock data
    """
    # Determine number of days based on period
    period_days = {
        '1y': 365, '2y': 730, '5y': 1825, 'max': 3650
    }.get(period, 730)
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate realistic stock price movement
    np.random.seed(42)
    n_days = len(dates)
    
    # Start with a base price
    base_price = 150.0
    
    # Generate price movements using geometric Brownian motion
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = [base_price]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        daily_volatility = abs(np.random.normal(0, 0.01))
        high = close * (1 + daily_volatility)
        low = close * (1 - daily_volatility)
        open_price = close * (1 + np.random.normal(0, 0.005))
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher on volatile days)
        base_volume = 1000000
        volume_multiplier = 1 + daily_volatility * 10
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def calculate_technical_indicators(data):
    """
    Calculate technical indicators for stock data.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicators
    """
    df = data.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    df['BB_middle'] = df['Close'].rolling(window=bb_period).mean()
    bb_std_val = df['Close'].rolling(window=bb_period).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std_val * bb_std)
    df['BB_lower'] = df['BB_middle'] - (bb_std_val * bb_std)
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
    
    # Williams %R
    df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # Commodity Channel Index (CCI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=20).mean()
    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    return df


def create_stock_model_config(input_size: int) -> ModelConfig:
    """
    Create model configuration optimized for stock price forecasting.
    
    Args:
        input_size: Number of input features
        
    Returns:
        ModelConfig for stock forecasting
    """
    return ModelConfig(
        input_size=input_size,
        output_size=1,  # Single price prediction
        hidden_sizes=[256, 128, 64, 32],  # Deeper network for complex patterns
        activations=[
            ActivationType.RELU, 
            ActivationType.RELU, 
            ActivationType.RELU,
            ActivationType.RELU
        ],
        dropout_rates=[0.3, 0.4, 0.3, 0.2],  # Moderate dropout
        output_activation=ActivationType.LINEAR,  # Linear for regression
        use_batch_norm=True
    )


def plot_stock_analysis(data, symbol, save_path=None):
    """
    Create comprehensive stock analysis visualizations.
    
    Args:
        data: DataFrame with stock data and indicators
        symbol: Stock symbol
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    
    # 1. Price and Moving Averages
    axes[0, 0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
    if 'SMA_20' in data.columns:
        axes[0, 0].plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    if 'SMA_50' in data.columns:
        axes[0, 0].plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
    axes[0, 0].set_title(f'{symbol} Stock Price and Moving Averages')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Volume (if available)
    if 'Volume' in data.columns:
        axes[0, 1].bar(data.index, data['Volume'], alpha=0.7, color='orange')
        axes[0, 1].set_title(f'{symbol} Trading Volume')
        axes[0, 1].set_ylabel('Volume')
    else:
        # Use a different metric if Volume is not available
        axes[0, 1].plot(data.index, data['Close'], alpha=0.7, color='orange')
        axes[0, 1].set_title(f'{symbol} Close Price')
        axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].grid(True)
    
    # 3. RSI
    if 'RSI' in data.columns:
        axes[1, 0].plot(data.index, data['RSI'], color='purple', linewidth=2)
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1, 0].set_title('RSI (Relative Strength Index)')
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. MACD
    if 'MACD' in data.columns and 'MACD_signal' in data.columns:
        axes[1, 1].plot(data.index, data['MACD'], label='MACD', linewidth=2)
        axes[1, 1].plot(data.index, data['MACD_signal'], label='Signal', linewidth=2)
        axes[1, 1].bar(data.index, data['MACD_histogram'], label='Histogram', alpha=0.6)
        axes[1, 1].set_title('MACD (Moving Average Convergence Divergence)')
        axes[1, 1].set_ylabel('MACD')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # 5. Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
        axes[2, 0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
        axes[2, 0].plot(data.index, data['BB_upper'], label='Upper Band', alpha=0.7)
        axes[2, 0].plot(data.index, data['BB_middle'], label='Middle Band', alpha=0.7)
        axes[2, 0].plot(data.index, data['BB_lower'], label='Lower Band', alpha=0.7)
        axes[2, 0].fill_between(data.index, data['BB_lower'], data['BB_upper'], alpha=0.1)
        axes[2, 0].set_title('Bollinger Bands')
        axes[2, 0].set_ylabel('Price ($)')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
    
    # 6. Price Distribution
    axes[2, 1].hist(data['Close'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[2, 1].axvline(data['Close'].mean(), color='red', linestyle='--', label=f'Mean: ${data["Close"].mean():.2f}')
    axes[2, 1].set_title('Price Distribution')
    axes[2, 1].set_xlabel('Price ($)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Stock analysis plot saved to {save_path}")
    
    plt.show()


def plot_forecast_results(dates, actual, predicted, symbol, save_path=None):
    """
    Plot forecasting results.
    
    Args:
        dates: Date array
        actual: Actual prices
        predicted: Predicted prices
        symbol: Stock symbol
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # 1. Time series comparison
    axes[0].plot(dates, actual, label='Actual', linewidth=2, alpha=0.8)
    axes[0].plot(dates, predicted, label='Predicted', linewidth=2, alpha=0.8)
    axes[0].set_title(f'{symbol} Stock Price Forecast')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Prediction errors
    errors = predicted - actual
    axes[1].plot(dates, errors, color='red', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_title('Prediction Errors')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Error ($)')
    axes[1].grid(True)
    
    # Add error statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    axes[1].text(0.02, 0.95, f'MAE: ${mae:.2f}\nRMSE: ${rmse:.2f}', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Forecast results plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function to run the stock price forecasting example."""
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting Stock Price Forecasting Example")
    
    # Load real stock data
    symbol = "AAPL"  # Apple Inc.
    sequences, targets, feature_names, dates = load_stock_data(symbol=symbol, period="2y")
    
    # Configure data handling
    data_config = DatasetConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        normalize_features=True,
        normalize_targets=True,  # Normalize prices for better training
        scaler_type="standard"
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
        train_data, val_data, test_data, batch_size=64
    )
    
    # Configure model
    model_config = create_stock_model_config(input_size=len(feature_names))
    
    # Create model
    model = NeuralModel(model_config)
    logger.info(f"Created model with {model.get_parameter_count()} parameters")
    
    # Configure training
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=64,
        num_epochs=150,
        optimizer="adam",
        loss_function="mse",
        early_stopping_patience=20,
        save_best_model=True,
        model_save_path="models/stock_forecasting",
        log_save_path="logs/stock_forecasting"
    )
    
    # Create trainer
    trainer = Trainer(model, training_config)
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("results/stock_forecasting_training_history.png")
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        regression_metrics=["mse", "rmse", "mae", "mape", "r2"],
        plot_predictions=True,
        save_plots=True,
        output_dir="results/stock_forecasting"
    )
    
    # Create evaluator
    evaluator = Evaluator(model, eval_config)
    
    # Evaluate model
    logger.info("Evaluating model...")
    report = evaluator.generate_report(
        test_loader, 
        task_type="regression"
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
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            features_batch, targets_batch = batch
            predictions = model(features_batch)
            y_true.extend(targets_batch.numpy().flatten())
            y_pred.extend(predictions.numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Denormalize predictions if targets were normalized
    if data_config.normalize_targets:
        y_true = data_handler.target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = data_handler.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    # Get test dates
    test_dates = dates[-len(y_true):]
    
    # Create comprehensive visualizations
    # Load full data for analysis
    full_data, _, _, _ = load_stock_data(symbol=symbol, period="2y")
    full_df = pd.DataFrame(full_data, columns=feature_names)
    full_df['Date'] = dates
    full_df['Close'] = targets
    
    plot_stock_analysis(full_df, symbol, "results/stock_analysis.png")
    plot_forecast_results(test_dates, y_true, y_pred, symbol, "results/stock_forecast_results.png")
    
    # Trading insights
    logger.info("Trading Insights:")
    price_change = (y_pred[-1] - y_pred[0]) / y_pred[0] * 100
    logger.info(f"  Predicted price change: {price_change:.2f}%")
    
    # Volatility analysis
    returns = np.diff(y_pred) / y_pred[:-1]
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
    logger.info(f"  Predicted annualized volatility: {volatility:.2%}")
    
    # Risk metrics
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    logger.info(f"  Predicted Sharpe ratio: {sharpe_ratio:.2f}")
    
    logger.info("Stock price forecasting example completed!")


if __name__ == "__main__":
    main()
