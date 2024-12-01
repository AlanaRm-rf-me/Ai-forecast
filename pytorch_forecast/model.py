import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional
import logging

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Dense layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input if needed (batch_size, seq_length, features)
        if x.dim() == 4:
            batch_size, seq_length, _, features = x.size()
            x = x.view(batch_size, seq_length, features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # Take the last output and pass through dense layers
        out = self.fc_layers(attn_out[:, -1, :])
        return out

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, seq_length: int):
        if data.ndim == 3:
            self.data = torch.FloatTensor(data)
        else:
            # Reshape data to (samples, sequence_length, features)
            n_samples = len(data) - seq_length + 1
            n_features = data.shape[-1] if data.ndim > 1 else 1
            reshaped_data = np.zeros((n_samples, seq_length, n_features))
            
            for i in range(n_samples):
                reshaped_data[i] = data[i:i + seq_length]
            
            self.data = torch.FloatTensor(reshaped_data)
        
        self.seq_length = seq_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return sequence and next value
        sequence = self.data[idx]
        if idx + 1 < len(self.data):
            target = self.data[idx + 1, -1, 0]  # Next closing price
        else:
            target = self.data[idx, -1, 0]  # Use last price if at the end
        return sequence, target

def create_features(data: np.ndarray, seq_length: int) -> np.ndarray:
    """Create technical indicators and features for the model"""
    features = []
    
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        
        # Technical indicators
        ma5 = np.mean(seq[-5:]) if len(seq) >= 5 else seq[-1]
        ma10 = np.mean(seq[-10:]) if len(seq) >= 10 else seq[-1]
        ma20 = np.mean(seq[-20:]) if len(seq) >= 20 else seq[-1]
        
        # Volatility
        volatility = np.std(seq)
        
        # Momentum
        momentum = seq[-1] - seq[0]
        
        # Rate of change
        roc = (seq[-1] - seq[0]) / seq[0] if seq[0] != 0 else 0
        
        # RSI (simplified)
        diff = np.diff(seq)
        gains = np.sum(diff[diff > 0])
        losses = -np.sum(diff[diff < 0])
        rsi = gains / (gains + losses) if (gains + losses) != 0 else 0.5
        
        # Combine features
        feature_vector = np.column_stack((
            seq,
            np.full(seq_length, ma5),
            np.full(seq_length, ma10),
            np.full(seq_length, ma20),
            np.full(seq_length, volatility),
            np.full(seq_length, momentum),
            np.full(seq_length, roc),
            np.full(seq_length, rsi)
        ))
        
        features.append(feature_vector)
    
    return np.array(features)

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device
) -> Tuple[List[float], List[float]]:
    """Train the model and return training history"""
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_pred = model(batch_x)
                val_loss += criterion(y_pred.squeeze(), batch_y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return train_losses, val_losses

def generate_forecast(
    model: nn.Module,
    last_sequence: torch.Tensor,
    n_steps: int,
    device: torch.device,
    n_simulations: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate forecasts with uncertainty estimation using Monte Carlo dropout"""
    model.eval()
    
    predictions = []
    for _ in range(n_simulations):
        model.train()  # Enable dropout
        with torch.no_grad():
            curr_seq = last_sequence.clone()
            step_preds = []
            
            for _ in range(n_steps):
                pred = model(curr_seq.unsqueeze(0).to(device))
                step_preds.append(pred.cpu().numpy())
                
                # Update sequence
                curr_seq = torch.cat([curr_seq[1:], pred.cpu()], dim=0)
            
            predictions.append(step_preds)
    
    predictions = np.array(predictions)
    
    # Calculate mean and confidence intervals
    mean_preds = np.mean(predictions, axis=0).squeeze()
    lower_bound = np.percentile(predictions, 2.5, axis=0).squeeze()
    upper_bound = np.percentile(predictions, 97.5, axis=0).squeeze()
    
    return mean_preds, lower_bound, upper_bound 