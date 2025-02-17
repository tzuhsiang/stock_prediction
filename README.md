# Stock Price Predictor

This project uses LSTM (Long Short-Term Memory) neural networks to predict stock prices based on historical data.

## Features
- Fetches historical stock data using yfinance
- Preprocesses data using MinMaxScaler
- Implements LSTM neural network for time series prediction
- Provides performance metrics (RMSE and R² score)
- Visualizes predictions vs actual prices

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The script can be run directly:

```bash
python stock_predictor.py
```

By default, it will predict TSMC (2330.TW) stock prices. You can modify the parameters in the script to predict other stocks:

```python
predict_stock(symbol='YOUR_STOCK_SYMBOL', start_date='2022-01-01', end_date='2023-12-31')
```

Examples:
- TSMC: '2330.TW'
- Apple: 'AAPL'
- Taiwan Stock Exchange: 'TAIEX'

## Model Details
- Uses 60-day sequences to predict the next day's price
- LSTM architecture with two layers (50 units each)
- Dropout layers (0.2) for regularization
- Training/Testing split: 80%/20%
- 50 epochs for training

## Output
The script will display:
- Training and Testing RMSE (Root Mean Square Error)
- R² scores for both training and testing sets
- A plot comparing predicted vs actual prices
