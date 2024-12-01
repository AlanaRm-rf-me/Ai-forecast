import os
import json
from datetime import datetime
from ai_forecast import ai_price_forecast
import requests
from typing import Dict, List, Optional
import logging

# Use existing logger if configured
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CryptoForecastClient:
    def __init__(self):
        self.api_url = "https://api.coingecko.com/api/v3"
        self.models_dir = "saved_models"
        self.history_dir = "forecast_history"
        self.plots_dir = "forecast_plots"
        self.available_coins = {}
        
        # Create necessary directories
        for directory in [self.models_dir, self.history_dir, self.plots_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load available cryptocurrencies
        self._update_available_coins()
    
    def _update_available_coins(self):
        """Fetch list of available cryptocurrencies"""
        try:
            response = requests.get(f"{self.api_url}/coins/list")
            coins = response.json()
            self.available_coins = {coin['id']: coin['name'] for coin in coins}
            
            # Save to file for offline access
            with open(os.path.join(self.history_dir, 'available_coins.json'), 'w') as f:
                json.dump(self.available_coins, f)
                
        except Exception as e:
            logging.warning(f"Could not fetch coin list: {e}")
            # Try to load from cached file
            try:
                with open(os.path.join(self.history_dir, 'available_coins.json'), 'r') as f:
                    self.available_coins = json.load(f)
            except:
                logging.error("Could not load cached coin list")
    
    def search_coins(self, query: str) -> Dict[str, str]:
        """Search available cryptocurrencies"""
        query = query.lower()
        return {
            id: name for id, name in self.available_coins.items()
            if query in id.lower() or query in name.lower()
        }
    
    def _load_history(self, crypto_id: str) -> List[Dict]:
        """Load forecast history for a cryptocurrency"""
        history_file = os.path.join(self.history_dir, f"{crypto_id}_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self, crypto_id: str, forecast_data: Dict):
        """Save forecast results to history"""
        history = self._load_history(crypto_id)
        history.append(forecast_data)
        
        history_file = os.path.join(self.history_dir, f"{crypto_id}_history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f)
    
    def generate_forecast(self, crypto_id: str, days: int = 30) -> Optional[Dict]:
        """Generate forecast for specified cryptocurrency"""
        if crypto_id not in self.available_coins:
            logging.error(f"Unknown cryptocurrency: {crypto_id}")
            return None
            
        print(f"\nGenerating forecast for {self.available_coins[crypto_id]} ({crypto_id})...")
        
        # Load previous model if it exists
        model_path = os.path.join(self.models_dir, f"{crypto_id}_model.keras")
        
        try:
            forecast_data = ai_price_forecast(
                api_url=self.api_url,
                crypto_id=crypto_id,
                historical_days=365,
                forecast_days=days,
                model_path=model_path,  # Pass model path to save/load
                save_path=model_path,   # Where to save the updated model
                timeout=900  # Increase timeout to 15 minutes
            )
            
            if forecast_data:
                forecast_dates, forecasted_prices, upper_bounds, lower_bounds = forecast_data
                
                # Prepare results
                results = {
                    "timestamp": datetime.now().isoformat(),
                    "crypto_id": crypto_id,
                    "crypto_name": self.available_coins[crypto_id],
                    "forecast_days": days,
                    "forecasts": [
                        {
                            "date": date.strftime('%Y-%m-%d'),
                            "price": price,
                            "upper_bound": upper,
                            "lower_bound": lower
                        }
                        for date, price, upper, lower in zip(
                            forecast_dates, forecasted_prices, 
                            upper_bounds, lower_bounds
                        )
                    ]
                }
                
                # Save to history
                self._save_history(crypto_id, results)
                
                # Print results
                print(f"\n=== Forecast Results for {self.available_coins[crypto_id]} ===")
                print(f"Generated on: {results['timestamp']}")
                print(f"Forecast Horizon: {days} days")
                print("------------------------")
                
                for forecast in results["forecasts"]:
                    print(f"\nDate: {forecast['date']}")
                    print(f"Predicted Price: ${forecast['price']:,.2f}")
                    print(f"Range: ${forecast['lower_bound']:,.2f} - ${forecast['upper_bound']:,.2f}")
                    print("------------------------")
                
                return results
        except TimeoutError:
            print("\nForecast generation timed out. This can happen during initial model training.")
            print("Try again, the saved partial model may help speed up the next attempt.")
            return None
        except Exception as e:
            print(f"\nAn error occurred during forecasting: {e}")
            return None

def main():
    client = CryptoForecastClient()
    
    while True:
        print("\n=== Crypto Forecast Client ===")
        print("1. Search cryptocurrencies")
        print("2. Generate forecast")
        print("3. View forecast history")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            query = input("Enter search term: ").strip()
            results = client.search_coins(query)
            
            if results:
                print("\nFound cryptocurrencies:")
                for id, name in results.items():
                    print(f"- {name} (ID: {id})")
            else:
                print("No cryptocurrencies found matching your search.")
                
        elif choice == "2":
            crypto_id = input("Enter cryptocurrency ID: ").strip()
            if crypto_id in client.available_coins:
                try:
                    days = int(input("Enter forecast days (7-30): ").strip())
                    days = max(7, min(30, days))  # Clamp between 7 and 30
                    client.generate_forecast(crypto_id, days)
                except ValueError:
                    print("Invalid number of days. Using default (30)")
                    client.generate_forecast(crypto_id)
            else:
                print("Unknown cryptocurrency ID. Use search to find correct ID.")
                
        elif choice == "3":
            crypto_id = input("Enter cryptocurrency ID: ").strip()
            if crypto_id in client.available_coins:
                history = client._load_history(crypto_id)
                if history:
                    print(f"\nForecast History for {client.available_coins[crypto_id]}:")
                    for entry in history:
                        print(f"\nGenerated on: {entry['timestamp']}")
                        print(f"Forecast Horizon: {entry['forecast_days']} days")
                        print("First prediction:", entry['forecasts'][0])
                        print("Last prediction:", entry['forecasts'][-1])
                        print("------------------------")
                else:
                    print("No forecast history found.")
            else:
                print("Unknown cryptocurrency ID. Use search to find correct ID.")
                
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 