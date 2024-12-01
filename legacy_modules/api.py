import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_and_plot_crypto_prices(api_url, crypto_id, days=7):
    """
    Fetches cryptocurrency price data and plots a graph.

    Parameters:
    - api_url (str): API endpoint for fetching cryptocurrency prices (e.g., CoinGecko API).
    - crypto_id (str): The cryptocurrency ID (e.g., 'bitcoin' for Bitcoin).
    - days (int): Number of past days to fetch data for.

    Returns:
    - None: Displays a graph of prices.
    """
    try:
        # Fetch historical price data
        response = requests.get(f"{api_url}/coins/{crypto_id}/market_chart", params={
            "vs_currency": "usd",
            "days": days
        })
        response.raise_for_status()
        data = response.json()

        # Extract timestamps and prices
        timestamps = [datetime.utcfromtimestamp(price[0] // 1000) for price in data["prices"]]
        prices = [price[1] for price in data["prices"]]

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, prices, marker="o", linestyle="-")
        plt.title(f"{crypto_id.capitalize()} Prices Over the Last {days} Days")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
api_url = "https://api.coingecko.com/api/v3"
crypto_id = "bitcoin"
fetch_and_plot_crypto_prices(api_url, crypto_id, days=30)
