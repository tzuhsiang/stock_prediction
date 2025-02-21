import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from urllib import request
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        
        # Get proxy configuration from environment variables
        http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
        
        self.proxies = {}
        if http_proxy and https_proxy:
            self.proxies = {
                'http': http_proxy,
                'https': https_proxy
            }
            print(f"Using proxy settings from environment: {self.proxies}")
        else:
            print("Warning: No proxy settings found in environment variables")
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock data directly from Yahoo Finance API"""
        # Configure session
        session = requests.Session()
        if self.proxies:
            session.proxies = self.proxies
            session.verify = False
            urllib3.disable_warnings(InsecureRequestWarning)

        # Add headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        session.headers.update(headers)

        print(f"Fetching data for {symbol}...")

        try:
            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp())
            end_ts = int(pd.Timestamp(end_date).timestamp())
            
            # Construct URL with parameters
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d"
            
            # Get data
            response = session.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Fix and parse JSON
            fixed_json = response.text.replace('""', '","')
            parsed_data = json.loads(fixed_json)
            
            # Extract chart data
            result = parsed_data['chart']['result'][0]
            timestamps = result['timestamp']
            quote = result['indicators']['quote'][0]
            
            # Create pandas DataFrame
            data = pd.DataFrame({
                'Open': quote['open'],
                'High': quote['high'],
                'Low': quote['low'],
                'Close': quote['close'],
                'Volume': quote['volume']
            }, index=pd.to_datetime([pd.Timestamp(ts, unit='s', tz='Asia/Taipei') for ts in timestamps]))
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Handle NaN values
            if data['Close'].isna().any():
                print("Found NaN values in Close prices, filling with forward/backward fill")
                data['Close'] = data['Close'].fillna(method='ffill').fillna(method='bfill')
            
            # Fill NaN values in other columns
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Double check for any remaining NaN
            if data.isna().any().any():
                raise ValueError(f"Unable to clean data for {symbol}: still contains NaN values")
            
            if len(data) < self.sequence_length:
                raise ValueError(f"Not enough data points. Need at least {self.sequence_length} days of data.")
            
            print(f"Successfully retrieved {len(data)} rows of data")
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def prepare_data(self, data: pd.DataFrame):
        """Prepare data for LSTM model"""
        prices = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(self.sequence_length, len(prices)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data (80% train, 20% test)
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        return X_train, X_test, y_train, y_test, prices
    
    def create_model(self):
        """Create LSTM model"""
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        return model

    def train_model(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1):
        """Train the model"""
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)
    
    def predict_future(self, last_sequence, days=30):
        """Predict future stock prices
        
        Args:
            last_sequence: The last sequence of actual prices
            days: Number of days to predict into the future
        """
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Reshape the sequence for prediction
            current_sequence_scaled = self.scaler.transform(current_sequence)
            current_sequence_reshaped = current_sequence_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            
            # Predict the next day
            next_pred = self.model.predict(current_sequence_reshaped)
            next_pred_original = self.scaler.inverse_transform(next_pred)
            
            # Add the prediction to our results
            future_predictions.append(next_pred_original[0, 0])
            
            # Update the sequence by removing the first element and adding the new prediction
            current_sequence = np.vstack([current_sequence[1:], next_pred_original])
            
        return np.array(future_predictions)

    def evaluate(self, y_true, y_pred):
        """Calculate error metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return rmse, r2

    def run_prediction(self, symbol: str, start_date: str, end_date: str, epochs=50):
        """Run full prediction pipeline"""
        # Fetch and prepare data
        data = self.fetch_data(symbol, start_date, end_date)
        X_train, X_test, y_train, y_test, prices = self.prepare_data(data)
        
        # Create and train model
        self.create_model()
        history = self.train_model(X_train, y_train, epochs=epochs)
        
        # Make predictions
        train_pred = self.predict(X_train)
        test_pred = self.predict(X_test)
        
        # Prepare actual values for comparison
        y_train_inv = self.scaler.inverse_transform([y_train]).T
        y_test_inv = self.scaler.inverse_transform([y_test]).T
        
        # Generate future predictions
        future_days = 30  # Predict next 30 days
        future_pred = self.predict_future(prices, future_days)
        
        # Create future dates for plotting
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        # Calculate metrics
        train_rmse, train_r2 = self.evaluate(y_train_inv, train_pred)
        test_rmse, test_r2 = self.evaluate(y_test_inv, test_pred)
        
        return {
            'data': data,
            'predictions': {
                'train': train_pred,
                'test': test_pred,
                'future': future_pred
            },
            'future_dates': future_dates,
            'actual': {
                'train': y_train_inv,
                'test': y_test_inv
            },
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'history': history
        }
