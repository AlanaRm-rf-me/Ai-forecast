from ai_forecast import ai_price_forecast

def main():
    # CoinGecko API endpoint
    api_url = "https://api.coingecko.com/api/v3"
    
    # Choose cryptocurrency to forecast
    crypto_id = "bitcoin"
    
    # Get and display forecast
    forecast_data = ai_price_forecast(
        api_url=api_url,
        crypto_id=crypto_id,
        historical_days=30,  # Use 30 days of historical data
        forecast_days=7      # Predict 7 days ahead
    )
    
    if forecast_data:
        forecast_dates, forecasted_prices, upper_bounds, lower_bounds = forecast_data
        print("\nForecast Results:")
        for date, price, upper, lower in zip(
            forecast_dates, 
            forecasted_prices, 
            upper_bounds, 
            lower_bounds
        ):
            print(f"\nDate: {date.strftime('%Y-%m-%d')}")
            print(f"Predicted Price: ${price:,.2f}")
            print(f"Range: ${lower:,.2f} - ${upper:,.2f}")

if __name__ == "__main__":
    main() 