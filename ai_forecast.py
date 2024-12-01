import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import requests

def create_sequences(data, seq_length):
    """Create sequences for LSTM model"""
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def build_lstm_model(seq_length):
    """Build and return LSTM model"""
    inputs = Input(shape=(seq_length, 1))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.1)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = LSTM(32, return_sequences=False)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    return model

def ai_price_forecast(
    api_url, 
    crypto_id, 
    historical_days=60, 
    forecast_days=7, 
    seq_length=10,
    epochs=100,
    batch_size=32,
    patience=20,
    learning_rate=0.0005,
    verbose=1
):
    """
    Generate cryptocurrency price forecasts using LSTM neural network
    
    Parameters:
    - api_url: CoinGecko API endpoint
    - crypto_id: Cryptocurrency identifier (e.g., 'bitcoin')
    - historical_days: Number of past days to train on
    - forecast_days: Number of days to forecast
    - seq_length: Sequence length for LSTM input
    - epochs: Maximum number of training epochs
    - batch_size: Training batch size
    - patience: Number of epochs to wait for improvement before early stopping
    - learning_rate: Learning rate for the Adam optimizer
    - verbose: Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch)
    """
    try:
        # Create model checkpoint directory if it doesn't exist
        checkpoint_dir = "model_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Fetch historical data
        response = requests.get(f"{api_url}/coins/{crypto_id}/market_chart", params={
            "vs_currency": "usd",
            "days": historical_days
        })
        response.raise_for_status()
        data = response.json()
        
        # Extract timestamps and prices with UTC timezone
        timestamps = [datetime.fromtimestamp(price[0] // 1000, timezone.utc) 
                     for price in data["prices"]]
        prices = np.array([price[1] for price in data["prices"]])
        
        # Scale the data
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
        
        # Create sequences for training
        X, y = create_sequences(prices_scaled, seq_length)
        
        # Split data into training and validation sets (80-20 split)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        model = build_lstm_model(seq_length)
        model.optimizer.learning_rate = learning_rate
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{crypto_id}_model.keras"),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Generate forecasts
        forecast_dates = []
        forecasted_prices = []
        confidence_upper = []
        confidence_lower = []
        
        last_sequence = prices_scaled[-seq_length:]
        
        # Generate multiple predictions for uncertainty estimation
        n_simulations = 10
        
        for _ in range(forecast_days):
            current_sequence = last_sequence[-seq_length:].reshape(1, seq_length, 1)
            
            # Run multiple predictions with dropout enabled
            predictions = []
            for _ in range(n_simulations):
                pred = model.predict(current_sequence, verbose=0)
                predictions.append(pred[0][0])
            
            # Calculate mean and standard deviation
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Transform predictions back to original scale
            predicted_price = scaler.inverse_transform([[mean_pred]])[0][0]
            upper_bound = scaler.inverse_transform([[mean_pred + 2*std_pred]])[0][0]
            lower_bound = scaler.inverse_transform([[mean_pred - 2*std_pred]])[0][0]
            
            # Store results
            next_date = timestamps[-1] + timedelta(days=1)
            forecast_dates.append(next_date)
            forecasted_prices.append(predicted_price)
            confidence_upper.append(upper_bound)
            confidence_lower.append(lower_bound)
            
            # Update sequence for next prediction
            last_sequence = np.append(last_sequence[1:], [[mean_pred]], axis=0)
            timestamps.append(next_date)
        
        # Plot forecast results
        plt.figure(figsize=(12, 7))
        plt.plot(timestamps[-30-forecast_days:-forecast_days], 
                prices[-30:], 
                label='Historical Prices', 
                color='blue')
        plt.plot(forecast_dates, 
                forecasted_prices, 
                label='AI Forecast', 
                color='red', 
                linestyle='--')
        plt.fill_between(forecast_dates, 
                        confidence_lower, 
                        confidence_upper,
                        color='red', 
                        alpha=0.1, 
                        label='Confidence Interval')
        
        plt.title(f"{crypto_id.capitalize()} AI Price Forecast (LSTM)")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return forecast_dates, forecasted_prices, confidence_upper, confidence_lower
        
    except KeyboardInterrupt:
        print("\nForecast interrupted by user")
        plt.close('all')
        return None
    except Exception as e:
        print(f"An error occurred during AI forecasting: {e}")
        return None

if __name__ == "__main__":
    api_url = "https://api.coingecko.com/api/v3"
    crypto_id = "bitcoin"
    ai_price_forecast(api_url, crypto_id) 