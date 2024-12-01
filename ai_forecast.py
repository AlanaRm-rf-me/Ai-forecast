import os
import numpy as np
import tensorflow as tf
import multiprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib
# Set non-interactive backend at the start of the file
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import requests
from typing import Tuple, List, Optional, Union
import logging
import time

# Configure logging only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get number of CPU cores
num_cores = multiprocessing.cpu_count()

# Configure TensorFlow to use all CPU cores effectively
tf.config.threading.set_inter_op_parallelism_threads(num_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.set_soft_device_placement(True)

# Enable CPU optimization flags
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPU_DETERMINISTIC_OPS'] = '0'
os.environ['TF_NUM_INTEROP_THREADS'] = str(num_cores)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(num_cores)

# Suppress TF info messages but keep warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print(f"Training will utilize {num_cores} CPU cores")

# Configure GPU settings
physical_devices = tf.config.list_physical_devices()
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} AMD GPU(s). Training will use GPU acceleration")
        # Use mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No compatible GPU found. Training will proceed on CPU")
    print("Available devices:", physical_devices)

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences with enhanced features for better prediction"""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        # Get the sequence
        seq = data[i:(i + seq_length)]
        
        # Calculate technical indicators
        ma5 = np.mean(seq[-5:]) if len(seq) >= 5 else seq[-1]
        ma10 = np.mean(seq[-10:]) if len(seq) >= 10 else seq[-1]
        ma20 = np.mean(seq[-20:]) if len(seq) >= 20 else seq[-1]
        
        # Calculate volatility features
        volatility = np.std(seq)
        range_price = np.max(seq) - np.min(seq)
        momentum = seq[-1] - seq[0]
        
        # Rate of change
        roc_1 = (seq[-1] - seq[-2]) / seq[-2] if len(seq) >= 2 else 0
        roc_5 = (seq[-1] - seq[-5]) / seq[-5] if len(seq) >= 5 else 0
        
        # Create feature matrix for this sequence
        feature_matrix = np.zeros((seq_length, 9))  # 9 features per timestep
        
        # Fill in the features for each timestep
        for j in range(seq_length):
            feature_matrix[j] = np.array([
                float(seq[j]),          # Price
                float(ma5),            # 5-day MA
                float(ma10),           # 10-day MA
                float(ma20),           # 20-day MA
                float(volatility),     # Volatility
                float(range_price),    # Price range
                float(momentum),       # Momentum
                float(roc_1),          # 1-day rate of change
                float(roc_5)           # 5-day rate of change
            ])
        
        sequences.append(feature_matrix)
        targets.append(float(data[i + seq_length]) if i + seq_length < len(data) else float(data[-1]))
    
    return np.array(sequences), np.array(targets)

def build_lstm_model(seq_length: int) -> Model:
    """
    Build enhanced LSTM model with better prediction capabilities
    """
    n_features = 9  # Original price + 8 technical indicators
    
    inputs = Input(shape=(seq_length, n_features))
    
    # First LSTM layer with increased complexity
    x = LSTM(256, return_sequences=True,
             kernel_regularizer='l2',
             recurrent_regularizer='l2')(inputs)
    x = Dropout(0.4)(x)
    
    # Second LSTM layer
    x = LSTM(128, return_sequences=True,
             kernel_regularizer='l2',
             recurrent_regularizer='l2')(x)
    x = Dropout(0.4)(x)
    
    # Third LSTM layer
    x = LSTM(64, return_sequences=False,
             kernel_regularizer='l2',
             recurrent_regularizer='l2')(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(32, activation='relu', kernel_regularizer='l2')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='huber',  # Robust loss function
        metrics=['mae']
    )
    return model

def fetch_historical_data(api_url: str, crypto_id: str, historical_days: int) -> Optional[List]:
    """
    Fetch historical price data from the API
    
    Args:
        api_url: Base API URL
        crypto_id: Cryptocurrency identifier
        historical_days: Number of days of historical data to fetch
    
    Returns:
        List of price data or None if request fails
    """
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
            
    except requests.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        logging.error(str(e))
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None

def plot_with_fallback(figure, title: str):
    """Helper function to handle plot display with fallback options"""
    logging.info(f"Attempting to save {title}...")
    try:
        # Always save to file since we're using Agg backend
        plots_dir = "forecast_plots"
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(plots_dir, f"{title.lower().replace(' ', '_')}_{timestamp}.png")
        figure.savefig(filename)
        logging.info(f"Plot saved to file: {filename}")
    except Exception as e:
        logging.error(f"Failed to save plot: {e}")
    finally:
        plt.close(figure)

def ai_price_forecast(
    api_url: str,
    crypto_id: str,
    historical_days: int = 365,
    forecast_days: int = 30,
    seq_length: int = 60,
    epochs: int = 500,
    batch_size: int = 32,
    patience: int = 30,
    learning_rate: float = 0.0001,
    verbose: int = 1,
    model_path: Optional[str] = None,
    save_path: Optional[str] = None,
    timeout: int = 900
) -> Optional[Tuple[List[datetime], List[float], List[float], List[float]]]:
    """Generate cryptocurrency price forecasts with enhanced uncertainty estimation"""
    try:
        start_time = time.time()
        logging.info(f"Starting price forecast for {crypto_id}")
        logging.info(f"Configuration: {historical_days} days history, {forecast_days} days forecast")
        
        def check_timeout():
            if time.time() - start_time > timeout:
                raise TimeoutError("Forecast generation timed out after 15 minutes")
        
        # Use larger batch size for CPU optimization
        batch_size = max(64, batch_size)
        logging.info(f"Using batch size: {batch_size}")
        
        # Fetch historical data
        logging.info(f"Fetching historical data from {api_url}...")
        price_data = fetch_historical_data(api_url, crypto_id, historical_days)
        if price_data is None:
            return None
        
        logging.info(f"Successfully fetched {len(price_data)} data points")
        
        # Extract timestamps and prices
        logging.info("Processing historical data...")
        timestamps = [datetime.fromtimestamp(price[0] // 1000, timezone.utc) 
                     for price in price_data]
        prices = np.array([float(price[1]) for price in price_data])
        
        if len(prices) < seq_length + forecast_days:
            logging.error(f"Insufficient data: need at least {seq_length + forecast_days} points")
            return None
        
        # Scale the data
        logging.info("Scaling price data...")
        scaler = MinMaxScaler()
        prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        
        # Create sequences with technical indicators
        logging.info("Creating feature sequences...")
        X, y = create_sequences(prices_scaled, seq_length)
        logging.info(f"Created {len(X)} sequences with {X.shape[-1]} features each")
        
        # Split data into training and validation sets
        logging.info("Splitting data into training and validation sets...")
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Try to load existing model
        if model_path and os.path.exists(model_path):
            try:
                logging.info(f"Loading existing model from {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                model.compile(
                    optimizer=Adam(learning_rate=learning_rate * 0.1),
                    loss='huber',
                    metrics=['mae']
                )
                logging.info("Model loaded successfully")
                logging.info(f"Adjusted learning rate to {learning_rate * 0.1} for fine-tuning")
            except Exception as e:
                logging.warning(f"Could not load model: {e}. Building new one.")
                model = build_lstm_model(seq_length)
        else:
            logging.info("Building new LSTM model...")
            model = build_lstm_model(seq_length)
        
        # Train model
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        if save_path:
            callbacks.append(ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Generate forecasts with uncertainty
        logging.info("\nStarting forecast generation phase...")
        forecast_dates = []
        forecasted_prices = []
        confidence_upper = []
        confidence_lower = []
        
        # Prepare initial sequence for forecasting
        last_sequence = prices_scaled[-seq_length:]
        last_features, _ = create_sequences(last_sequence, seq_length)
        
        if len(last_features) == 0:
            last_features = np.zeros((1, seq_length, 9))
            for i in range(seq_length):
                last_features[0, i] = np.array([
                    float(last_sequence[i]),
                    float(np.mean(last_sequence[max(0, i-4):i+1])),
                    float(np.mean(last_sequence[max(0, i-9):i+1])),
                    float(np.mean(last_sequence[max(0, i-19):i+1])),
                    float(np.std(last_sequence[:i+1])),
                    float(np.max(last_sequence[:i+1]) - np.min(last_sequence[:i+1])),
                    float(last_sequence[i] - last_sequence[0]),
                    float((last_sequence[i] - last_sequence[i-1])/last_sequence[i-1] if i > 0 else 0),
                    float((last_sequence[i] - last_sequence[max(0, i-5)])/last_sequence[max(0, i-5)] if i > 4 else 0)
                ])
        
        # Monte Carlo simulation
        n_simulations = 100
        logging.info(f"Running {n_simulations} Monte Carlo simulations for {forecast_days} days...")
        
        try:
            for day in range(forecast_days):
                check_timeout()
                logging.info(f"Generating forecast for day {day + 1}/{forecast_days}")
                predictions = []
                
                try:
                    for sim in range(n_simulations):
                        try:
                            # Add small random noise to predictions for Monte Carlo
                            pred = model.predict(last_features, verbose=0)
                            if pred is None or pred.size == 0:
                                raise ValueError("Model prediction returned None or empty array")
                            # Add noise scaled to recent volatility
                            noise = np.random.normal(0, 0.02)  # 2% standard deviation
                            predictions.append(float(pred[0, 0]) * (1 + noise))
                            
                            if sim % 20 == 0:
                                logging.info(f"  - Completed simulation {sim + 1}/{n_simulations}")
                        except Exception as sim_error:
                            logging.error(f"Error in simulation {sim + 1}: {sim_error}")
                            continue
                    
                    if not predictions:
                        raise ValueError("No valid predictions generated for this day")
                    
                    # Calculate statistics
                    mean_pred = np.mean(predictions)
                    std_pred = np.std(predictions)
                    
                    # Transform predictions back to original scale
                    try:
                        predicted_price = float(scaler.inverse_transform([[mean_pred]])[0, 0])
                        upper_bound = float(scaler.inverse_transform([[mean_pred + 2*std_pred]])[0, 0])
                        lower_bound = float(scaler.inverse_transform([[mean_pred - 2*std_pred]])[0, 0])
                    except Exception as scale_error:
                        logging.error(f"Error scaling predictions: {scale_error}")
                        raise
                    
                    # Store results
                    next_date = timestamps[-1] + timedelta(days=1 + day)
                    
                    # Validate predictions are reasonable
                    if day > 0:
                        prev_price = forecasted_prices[-1]
                        max_change = prev_price * 0.5  # Max 50% change per day
                        if abs(predicted_price - prev_price) > max_change:
                            logging.warning(f"Large price change detected: ${prev_price:.2f} -> ${predicted_price:.2f}")
                            # Dampen extreme predictions
                            if predicted_price > prev_price:
                                predicted_price = prev_price + max_change
                            else:
                                predicted_price = prev_price - max_change
                            # Recalculate bounds
                            upper_bound = predicted_price * 1.1  # 10% upper bound
                            lower_bound = predicted_price * 0.9  # 10% lower bound
                    
                    forecast_dates.append(next_date)
                    forecasted_prices.append(predicted_price)
                    confidence_upper.append(upper_bound)
                    confidence_lower.append(lower_bound)
                    
                    # Update sequence for next prediction
                    try:
                        # Scale the predicted price back to normalized space
                        scaled_pred = scaler.transform([[predicted_price]])[0, 0]
                        # Update the sequence with the new prediction
                        last_sequence = np.append(last_sequence[1:], scaled_pred)
                        # Create new features for the updated sequence
                        new_features = np.zeros((1, seq_length, 9))
                        
                        # Calculate technical indicators for the new sequence
                        ma5 = np.mean(last_sequence[-5:]) if len(last_sequence) >= 5 else last_sequence[-1]
                        ma10 = np.mean(last_sequence[-10:]) if len(last_sequence) >= 10 else last_sequence[-1]
                        ma20 = np.mean(last_sequence[-20:]) if len(last_sequence) >= 20 else last_sequence[-1]
                        volatility = np.std(last_sequence)
                        range_price = np.max(last_sequence) - np.min(last_sequence)
                        momentum = last_sequence[-1] - last_sequence[0]
                        
                        for i in range(seq_length):
                            new_features[0, i] = np.array([
                                float(last_sequence[i]),
                                float(ma5),
                                float(ma10),
                                float(ma20),
                                float(volatility),
                                float(range_price),
                                float(momentum),
                                float((last_sequence[i] - last_sequence[i-1])/last_sequence[i-1] if i > 0 else 0),
                                float((last_sequence[i] - last_sequence[max(0, i-5)])/last_sequence[max(0, i-5)] if i > 4 else 0)
                            ])
                        
                        last_features = new_features
                        
                    except Exception as seq_error:
                        logging.error(f"Error updating sequence: {seq_error}")
                        raise
                    
                    # Validate we're not getting stuck
                    if len(forecast_dates) > 1 and forecast_dates[-1] <= forecast_dates[-2]:
                        logging.error("Date progression error detected")
                        raise ValueError("Forecast dates are not progressing correctly")
                    
                    logging.info(f"  Forecast for {next_date.strftime('%Y-%m-%d')}: ${predicted_price:,.2f}")
                    logging.info(f"  Range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
                    
                except Exception as day_error:
                    logging.error(f"Error processing day {day + 1}: {day_error}")
                    if day == 0:  # If we fail on the first day, abort
                        raise
                    break  # For later days, try to return partial results
            
            if not forecast_dates:  # If we have no results at all
                raise ValueError("No forecasts were generated successfully")
                
        except Exception as mc_error:
            logging.error(f"Monte Carlo simulation failed: {mc_error}")
            raise
        
        # Create final visualization
        logging.info("\nGenerating final forecast visualization...")
        try:
            forecast_fig = plt.figure(figsize=(15, 8))
            
            # Plot historical prices (last 90 days)
            plt.plot(timestamps[-90-forecast_days:-forecast_days], 
                    prices[-90:], 
                    label='Historical Prices', 
                    color='blue',
                    linewidth=2)
            
            # Plot forecast
            plt.plot(forecast_dates, 
                    forecasted_prices, 
                    label='AI Forecast', 
                    color='red', 
                    linestyle='--',
                    linewidth=2)
            
            # Plot confidence interval
            plt.fill_between(forecast_dates, 
                           confidence_lower, 
                           confidence_upper,
                           color='red', 
                           alpha=0.3, 
                           label='95% Confidence Interval')
            
            plt.title(f"{crypto_id.capitalize()} AI Price Forecast (LSTM)")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            plt.tight_layout()
            
            plot_with_fallback(forecast_fig, "Forecast Plot")
            
        except Exception as e:
            logging.error(f"Error generating forecast plot: {e}")
            plt.close('all')
        
        logging.info("Forecast generation complete!")
        logging.info(f"Total time: {time.time() - start_time:.1f} seconds")
        
        return forecast_dates, forecasted_prices, confidence_upper, confidence_lower
        
    except Exception as e:
        logging.error(f"An error occurred during AI forecasting: {e}")
        plt.close('all')
        return None 