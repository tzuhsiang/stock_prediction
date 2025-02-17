# Stock Price Predictor

This project uses LSTM (Long Short-Term Memory) neural networks to predict stock prices based on historical data.

## Features
- Fetches historical stock data using yfinance
- Preprocesses data using MinMaxScaler
- Implements LSTM neural network for time series prediction
- Provides performance metrics (RMSE and R² score)
- Visualizes predictions vs actual prices
- Docker support for easy deployment

## Web Demo
!(imgs/demo.png)



## Local Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Docker Usage

### Using Docker Compose (Recommended)

1. Build and start the container:
```bash
docker-compose up -d --build
```

2. View logs:
```bash
docker-compose logs -f
```

3. Stop the container:
```bash
docker-compose down
```

### Using Docker Directly

1. Build the image:
```bash
docker build -t stock-predictor .
```

2. Run the container:
```bash
docker run -d \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name stock-predictor \
  stock-predictor
```

## Model Configuration

The stock prediction model:
- Uses 60-day sequences to predict the next day's price
- LSTM architecture with two layers (50 units each)
- Dropout layers (0.2) for regularization
- Training/Testing split: 80%/20%
- 50 epochs for training

## Data Storage
- Trained models are saved in the `models/` directory
- Historical data is cached in the `data/` directory
- Both directories are mounted as Docker volumes for persistence

## Default Settings
- Default stock: TSMC (2330.TW)
- Time range: 2022-01-01 to 2023-12-31
- Timezone: Asia/Taipei

To predict different stocks, modify the parameters in stock_predictor.py:
```python
predict_stock(symbol='YOUR_STOCK_SYMBOL', start_date='2022-01-01', end_date='2023-12-31')
```

Examples:
- TSMC: '2330.TW'
- Apple: 'AAPL'
- Taiwan Stock Exchange: 'TAIEX'

## Output
The script will display:
- Training and Testing RMSE (Root Mean Square Error)
- R² scores for both training and testing sets
- A plot comparing predicted vs actual prices
