import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance"""
    data = yf.download(symbol, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {symbol}")
    return data

def prepare_data(data, sequence_length=60):
    """Prepare data for LSTM model"""
    # Use closing prices
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(prices)
    
    X, y = [], []
    for i in range(sequence_length, len(prices)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def create_model(sequence_length):
    """Create LSTM model"""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_stock(symbol='AAPL', start_date='2020-01-01', end_date='2023-12-31'):
    """Main function to predict stock prices"""
    # Fetch data
    data = fetch_stock_data(symbol, start_date, end_date)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)
    
    # Reshape data for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Create and train model
    model = create_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    y_test_inv = scaler.inverse_transform([y_test])
    
    # Calculate error metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_inv.T, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv.T, test_predict))
    train_r2 = r2_score(y_train_inv.T, train_predict)
    test_r2 = r2_score(y_test_inv.T, test_predict)
    
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Train R2 Score: {train_r2:.2f}")
    print(f"Test R2 Score: {test_r2:.2f}")
    
    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(data.index[-len(test_predict):], test_predict, label='Predictions')
    plt.plot(data.index[-len(test_predict):], y_test_inv.T, label='Actual')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage
    predict_stock(symbol='2330.TW', start_date='2022-01-01', end_date='2023-12-31')
