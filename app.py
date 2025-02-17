import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from stock_predictor import StockPredictor
import numpy as np

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.title("Stock Price Predictor 📈")

# Sidebar inputs
with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Stock Symbol", value="2330.TW")
    
    # Date inputs
    today = date.today()
    past_date = today - timedelta(days=365*2)  # 2 years ago
    start_date = st.date_input("Start Date", value=past_date)
    end_date = st.date_input("End Date", value=today)
    
    # Model parameters
    st.header("Model Parameters")
    sequence_length = st.slider("Sequence Length (days)", min_value=30, max_value=90, value=60)
    epochs = st.slider("Training Epochs", min_value=10, max_value=100, value=50)
    
    predict_button = st.button("Predict", type="primary")

# Main content
if predict_button:
    try:
        with st.spinner("Training model and making predictions..."):
            # Initialize and run prediction
            predictor = StockPredictor(sequence_length=sequence_length)
            results = predictor.run_prediction(
                symbol=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                epochs=epochs
            )
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train RMSE", f"{results['metrics']['train_rmse']:.2f}")
            with col2:
                st.metric("Test RMSE", f"{results['metrics']['test_rmse']:.2f}")
            with col3:
                st.metric("Train R²", f"{results['metrics']['train_r2']:.3f}")
            with col4:
                st.metric("Test R²", f"{results['metrics']['test_r2']:.3f}")
            
            # Plot results
            st.subheader("Price Prediction Results")
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add actual prices
            fig.add_trace(go.Scatter(
                x=results['data'].index[-len(results['actual']['test']):],
                y=results['actual']['test'].flatten(),
                name="Actual",
                line=dict(color='blue')
            ))
            
            # Add predictions
            fig.add_trace(go.Scatter(
                x=results['data'].index[-len(results['predictions']['test']):],
                y=results['predictions']['test'].flatten(),
                name="Predicted",
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Stock Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified',
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Training history
            st.subheader("Training History")
            history = results['history']
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                y=history.history['loss'],
                name="Training Loss"
            ))
            fig_history.add_trace(go.Scatter(
                y=history.history['val_loss'],
                name="Validation Loss"
            ))
            fig_history.update_layout(
                title="Model Loss During Training",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                plot_bgcolor='white'
            )
            st.plotly_chart(fig_history, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Enter a stock symbol and click 'Predict' to start the prediction process.")
    
# Display description
st.markdown("""
### About
This application uses an LSTM (Long Short-Term Memory) neural network to predict stock prices. 
The model is trained on historical data and attempts to predict future price movements.

### Features
- Real-time data fetching from Yahoo Finance
- Interactive plots with actual vs predicted prices
- Model performance metrics (RMSE and R² score)
- Customizable model parameters
""")
