import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from api import fetch_and_plot_crypto_prices

def create_features(prices, lookback=30):
    """
    Create features for the forecasting model using previous prices
    """
    df = pd.DataFrame(prices, columns=['price'])
    for i in range(1, lookback + 1):
        df[f'price_t-{i}'] = df['price'].shift(i)
    return df.dropna()

def forecast_crypto_prices(prices, timestamps, forecast_days=7):
    """
    Forecasts cryptocurrency prices using historical data.
    
    Parameters:
    - prices (list): Historical price data
    - timestamps (list): Corresponding timestamps for the price data
    - forecast_days (int): Number of days to forecast
    
    Returns:
    - tuple: (forecast_dates, forecasted_prices, confidence_upper, confidence_lower)
    """
    # Prepare the data
    lookback = 30  # Use 30 days of historical data for prediction
    df = create_features(prices, lookback)
    
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Train the model
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    
    # Prepare data for forecasting
    last_prices = prices[-lookback:]
    forecasted_prices = []
    confidence_upper = []
    confidence_lower = []
    
    # Generate forecast dates
    last_date = timestamps[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    
    # Make predictions
    for _ in range(forecast_days):
        features = np.array(last_prices[-lookback:]).reshape(1, -1)
        features_scaled = scaler_X.transform(features)
        
        # Predict and inverse transform
        pred_scaled = model.predict(features_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        
        # Add confidence intervals (using 5% variance as an example)
        confidence = pred * 0.05
        confidence_upper.append(pred + confidence)
        confidence_lower.append(pred - confidence)
        
        # Store prediction and update last_prices for next prediction
        forecasted_prices.append(pred)
        last_prices = np.append(last_prices[1:], pred)
    
    return forecast_dates, forecasted_prices, confidence_upper, confidence_lower

def plot_with_forecast(timestamps, prices, forecast_dates, forecasted_prices, 
                      confidence_upper, confidence_lower, crypto_id):
    """
    Plots historical prices with forecast and confidence intervals
    """
    plt.figure(figsize=(12, 7))
    
    # Plot historical data
    plt.plot(timestamps, prices, label='Historical Prices', color='blue')
    
    # Plot forecast
    plt.plot(forecast_dates, forecasted_prices, label='Forecast', color='red', linestyle='--')
    
    # Plot confidence intervals
    plt.fill_between(forecast_dates, confidence_lower, confidence_upper, 
                    color='red', alpha=0.1, label='Confidence Interval')
    
    plt.title(f"{crypto_id.capitalize()} Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_price_forecast(api_url, crypto_id, historical_days=30, forecast_days=7):
    """
    Main function to fetch data and create price forecast
    """
    try:
        # Fetch historical data using the existing API function
        import requests
        response = requests.get(f"{api_url}/coins/{crypto_id}/market_chart", params={
            "vs_currency": "usd",
            "days": historical_days
        })
        response.raise_for_status()
        data = response.json()
        
        # Extract timestamps and prices
        timestamps = [datetime.utcfromtimestamp(price[0] // 1000) for price in data["prices"]]
        prices = [price[1] for price in data["prices"]]
        
        # Generate forecast
        forecast_dates, forecasted_prices, confidence_upper, confidence_lower = \
            forecast_crypto_prices(prices, timestamps, forecast_days)
        
        # Plot results
        plot_with_forecast(timestamps, prices, forecast_dates, forecasted_prices,
                         confidence_upper, confidence_lower, crypto_id)
        
        return forecast_dates, forecasted_prices, confidence_upper, confidence_lower
        
    except Exception as e:
        print(f"An error occurred during forecasting: {e}")
        return None

# Example usage:
if __name__ == "__main__":
    api_url = "https://api.coingecko.com/api/v3"
    crypto_id = "bitcoin"
    get_price_forecast(api_url, crypto_id, historical_days=30, forecast_days=7) 