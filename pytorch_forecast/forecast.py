import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import requests
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import logging
from model import LSTMForecaster, TimeSeriesDataset, create_features, train_model, generate_forecast

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_historical_data(api_url: str, crypto_id: str, historical_days: int) -> Optional[List]:
    """Fetch historical price data from the API"""
    try:
        with requests.Session() as session:
            response = session.get(
                f"{api_url}/coins/{crypto_id}/market_chart",
                params={
                    "vs_currency": "usd",
                    "days": historical_days
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if "prices" not in data:
                raise ValueError("No prices data found in the response")
            
            return data["prices"]
            
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def prepare_data(
    prices: np.ndarray,
    seq_length: int,
    batch_size: int,
    train_split: float = 0.8
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepare data loaders for training"""
    # Create features
    features = create_features(prices, seq_length)
    
    # Split into train and validation sets
    train_size = int(len(features) * train_split)
    train_data = features[:train_size]
    val_data = features[train_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    
    return train_loader, val_loader

def plot_forecast(
    timestamps: List[datetime],
    prices: np.ndarray,
    forecast_dates: List[datetime],
    mean_preds: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    crypto_id: str
):
    """Plot the forecast results"""
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(timestamps[-90:], 
            prices[-90:], 
            label='Historical Prices', 
            color='blue',
            linewidth=2)
    
    # Plot forecast
    plt.plot(forecast_dates, 
            mean_preds, 
            label='AI Forecast', 
            color='red', 
            linestyle='--',
            linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(forecast_dates, 
                    lower_bound, 
                    upper_bound,
                    color='red', 
                    alpha=0.3, 
                    label='95% Confidence Interval')
    
    plt.title(f"{crypto_id.capitalize()} Price Forecast (LSTM with Attention)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Format price labels
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses: List[float], val_losses: List[float]):
    """Plot training history"""
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def crypto_forecast(
    api_url: str,
    crypto_id: str,
    historical_days: int = 365,
    forecast_days: int = 30,
    seq_length: int = 60,
    hidden_dim: int = 256,
    num_layers: int = 3,
    epochs: int = 500,
    batch_size: int = 32,
    learning_rate: float = 0.0001
) -> Optional[Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]]:
    """Generate cryptocurrency price forecasts using PyTorch LSTM model"""
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Fetch data
        price_data = fetch_historical_data(api_url, crypto_id, historical_days)
        if price_data is None:
            return None
        
        # Extract timestamps and prices
        timestamps = [datetime.fromtimestamp(price[0] // 1000, timezone.utc) 
                     for price in price_data]
        prices = np.array([price[1] for price in price_data])
        
        # Scale the data
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Prepare data loaders
        train_loader, val_loader = prepare_data(prices_scaled, seq_length, batch_size)
        
        # Initialize model
        input_dim = 8  # Number of features per timestep
        model = LSTMForecaster(input_dim, hidden_dim, num_layers).to(device)
        
        # Train model
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            epochs, learning_rate, device
        )
        
        # Plot training history
        plot_training_history(train_losses, val_losses)
        
        # Generate forecast
        last_sequence = create_features(prices_scaled[-seq_length:], seq_length)[-1]
        last_sequence = torch.FloatTensor(last_sequence).unsqueeze(0)
        mean_preds, lower_bound, upper_bound = generate_forecast(
            model, last_sequence, forecast_days, device
        )
        
        # Transform predictions back to original scale
        mean_preds = scaler.inverse_transform(mean_preds.reshape(-1, 1)).flatten()
        lower_bound = scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
        upper_bound = scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
        
        # Generate forecast dates
        forecast_dates = [timestamps[-1] + timedelta(days=i+1) for i in range(forecast_days)]
        
        # Plot results
        plot_forecast(
            timestamps, prices, forecast_dates,
            mean_preds, lower_bound, upper_bound,
            crypto_id
        )
        
        return forecast_dates, mean_preds, lower_bound, upper_bound
        
    except Exception as e:
        logging.error(f"An error occurred during forecasting: {e}")
        return None

if __name__ == "__main__":
    api_url = "https://api.coingecko.com/api/v3"
    crypto_id = "bitcoin"
    
    print(f"\nGenerating forecast for {crypto_id}...")
    
    forecast_data = crypto_forecast(
        api_url=api_url,
        crypto_id=crypto_id,
        historical_days=365,    # Use 1 year of historical data
        forecast_days=30,       # Predict 30 days ahead
        seq_length=60,          # Use 60 days of data for each prediction
        hidden_dim=256,         # Size of LSTM hidden layers
        num_layers=3,           # Number of LSTM layers
        epochs=500,             # Maximum training epochs
        batch_size=32,          # Batch size
        learning_rate=0.0001    # Learning rate
    )
    
    if forecast_data:
        forecast_dates, mean_preds, lower_bound, upper_bound = forecast_data
        print("\n=== Forecast Results ===")
        print(f"Cryptocurrency: {crypto_id.upper()}")
        print(f"Training Data: 1 year")
        print(f"Forecast Horizon: 30 days")
        print("------------------------")
        for date, price, lower, upper in zip(
            forecast_dates, mean_preds, lower_bound, upper_bound
        ):
            print(f"\nDate: {date.strftime('%Y-%m-%d')}")
            print(f"Predicted Price: ${price:,.2f}")
            print(f"Range: ${lower:,.2f} - ${upper:,.2f}")
            print("------------------------")
    else:
        print("Failed to generate forecast.") 