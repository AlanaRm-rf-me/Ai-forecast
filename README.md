# AI Cryptocurrency Price Forecaster

An advanced deep learning-based cryptocurrency price forecasting tool that uses LSTM neural networks to predict future price movements with confidence intervals.

## Features

- Historical price data fetching from cryptocurrency APIs
- Advanced LSTM model with technical indicators
- Monte Carlo simulation for uncertainty estimation
- Automatic CPU/GPU detection and optimization
- Beautiful visualization of forecasts with confidence intervals
- Automatic plot saving functionality
- Robust error handling and logging

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Requests
- Scikit-learn

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:

```bash
pip install tensorflow numpy pandas matplotlib requests scikit-learn
```

## Usage

The main functionality is provided through the `ai_price_forecast` function in `ai_forecast.py`:

```python
from ai_forecast import ai_price_forecast

# Example usage
forecast_dates, prices, upper_bound, lower_bound = ai_price_forecast(
    api_url="https://api.coingecko.com/api/v3",
    crypto_id="bitcoin",
    historical_days=180,
    forecast_days=30
)
```

### Parameters

- `api_url`: Base URL for the cryptocurrency API
- `crypto_id`: Identifier for the cryptocurrency (e.g., "bitcoin")
- `historical_days`: Number of historical days to use for training (default: 180)
- `forecast_days`: Number of days to forecast into the future (default: 30)
- `seq_length`: Sequence length for LSTM input (default: 90)
- `epochs`: Number of training epochs (default: 500)
- `batch_size`: Batch size for training (default: 32)
- `patience`: Early stopping patience (default: 30)
- `model_path`: Path to save/load the model (optional)
- `save_path`: Path to save the best model during training (optional)

Currently you are only prompted for forecast length 

## Features in Detail

### Technical Indicators
The model uses several technical indicators for improved forecasting:
- Moving averages (5, 10, and 20-day)
- Volatility measures
- Price momentum
- Rate of change indicators

### Hardware Optimization
- Automatic GPU detection and utilization when available
- Graceful fallback to CPU when GPU is unavailable or encounters errors
- Multi-core CPU optimization

### Visualization
- Historical price plots
- Forecast line with confidence intervals
- Automatic plot saving with timestamps

## Output

The function returns four lists:
1. Forecast dates
2. Forecasted prices
3. Upper confidence bounds
4. Lower confidence bounds

Plots are automatically saved in the `forecast_plots` directory.

## Error Handling

The system includes comprehensive error handling for:
- API connection issues
- Data processing errors
- Hardware configuration problems
- Model training issues

All errors are logged with detailed messages for debugging.

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
