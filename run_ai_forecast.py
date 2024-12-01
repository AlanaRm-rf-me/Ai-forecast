from ai_forecast import ai_price_forecast

def main():
    api_url = "https://api.coingecko.com/api/v3"
    crypto_id = "bitcoin"
    
    print(f"\nGenerating AI forecast for {crypto_id}...")
    
    # Get AI-powered forecast with custom training parameters
    forecast_data = ai_price_forecast(
        api_url=api_url,
        crypto_id=crypto_id,
        historical_days=60,    # Use 60 days for training
        forecast_days=7,       # Predict 7 days ahead
        seq_length=10,         # Use 10 days of data for each prediction
        epochs=200,            # Maximum number of training epochs
        batch_size=32,         # Training batch size
        patience=20,           # Early stopping patience
        learning_rate=0.0005,  # Learning rate
        verbose=1             # Show training progress
    )
    
    if forecast_data:
        forecast_dates, forecasted_prices, upper_bounds, lower_bounds = forecast_data
        print("\n=== AI Forecast Results ===")
        print(f"Cryptocurrency: {crypto_id.upper()}")
        print("------------------------")
        for date, price, upper, lower in zip(
            forecast_dates, 
            forecasted_prices, 
            upper_bounds, 
            lower_bounds
        ):
            print(f"\nDate: {date.strftime('%Y-%m-%d')}")
            print(f"Predicted Price: ${price:,.2f}")
            print(f"Range: ${lower:,.2f} - ${upper:,.2f}")
            print("------------------------")
    else:
        print("Failed to generate forecast.")

if __name__ == "__main__":
    main() 