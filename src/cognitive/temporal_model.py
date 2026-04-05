"""
Temporal model for learning behavioral patterns over time.

Uses LSTM/GRU to process sequences of features.
"""

import numpy as np
from typing import Optional, List
from collections import deque

import torch
import torch.nn as nn

from ..config import CognitiveConfig


class LSTMEncoder(nn.Module):
    """LSTM-based temporal encoder."""
    
    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Layer normalization on input
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output projection
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Embedding tensor of shape (batch, 64)
        """
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Project to embedding
        embedding = self.output_proj(last_output)
        
        return embedding


class TemporalModel:
    """Manages temporal sequence processing."""
    
    def __init__(self, config: Optional[CognitiveConfig] = None):
        """
        Initialize temporal model.
        
        Args:
            config: Cognitive modeling configuration.
        """
        if config is None:
            config = CognitiveConfig()
        
        self.config = config
        self.sequence_length = config.sequence_length
        
        # Sequence buffer
        self._sequence_buffer: deque = deque(maxlen=self.sequence_length)
        self._timestamps: deque = deque(maxlen=self.sequence_length)
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMEncoder(
            input_size=32,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers
        ).to(self.device)
        self.model.eval()  # Start in eval mode
    
    def add_features(self, features: np.ndarray, timestamp: float) -> None:
        """
        Add feature vector to sequence buffer.
        
        Args:
            features: Feature vector of shape (32,)
            timestamp: Timestamp of the features.
        """
        self._sequence_buffer.append(features.astype(np.float32))
        self._timestamps.append(timestamp)
    
    def get_embedding(self) -> Optional[np.ndarray]:
        """
        Get current cognitive embedding.
        
        Returns:
            Embedding of shape (64,) or None if insufficient data.
        """
        if len(self._sequence_buffer) < 10:  # Minimum frames needed
            return None
        
        # Prepare sequence tensor
        sequence = self._prepare_sequence()
        
        with torch.no_grad():
            embedding = self.model(sequence)
        
        return embedding.cpu().numpy().squeeze()
    
    def _prepare_sequence(self) -> torch.Tensor:
        """Prepare sequence tensor from buffer."""
        # Create padded sequence
        seq_array = np.zeros((self.sequence_length, 32), dtype=np.float32)
        
        buffer_list = list(self._sequence_buffer)
        n = len(buffer_list)
        
        # Fill from the end (most recent at the end)
        for i, features in enumerate(buffer_list):
            seq_array[self.sequence_length - n + i] = features
        
        # Convert to tensor and add batch dimension
        tensor = torch.tensor(seq_array, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def get_sequence_quality(self) -> float:
        """
        Get quality score for current sequence.
        
        Returns:
            Quality score [0, 1] based on valid frames.
        """
        if len(self._sequence_buffer) == 0:
            return 0.0
        
        # Count frames with non-zero features (face detected)
        valid_count = sum(
            1 for f in self._sequence_buffer if np.any(f != 0)
        )
        
        return valid_count / len(self._sequence_buffer)
    
    def reset(self) -> None:
        """Reset sequence buffer."""
        self._sequence_buffer.clear()
        self._timestamps.clear()
    
    def save_model(self, path: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
