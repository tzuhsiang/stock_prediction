import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class StockPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    
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
        
        # Calculate metrics
        train_rmse, train_r2 = self.evaluate(y_train_inv, train_pred)
        test_rmse, test_r2 = self.evaluate(y_test_inv, test_pred)
        
        return {
            'data': data,
            'predictions': {
                'train': train_pred,
                'test': test_pred
            },
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
