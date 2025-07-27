"""
Model I/O utilities for saving and loading LSTM models
"""

import os
from typing import Any, Dict

import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class ModelIO:
    """Handles saving and loading of LSTM models and scalers"""

    @staticmethod
    def save_model(
        model: nn.Module,
        scaler: MinMaxScaler,
        features_list: list,
        sequence_length: int,
        hidden_sizes: list,
        model_path: str = "models/lstm_model.pth",
        scaler_path: str = "models/scaler.pkl",
    ):
        """
        Save model and scaler

        Args:
            model: PyTorch model to save
            scaler: Fitted scaler to save
            features_list: List of feature names
            sequence_length: Sequence length used by model
            hidden_sizes: Hidden layer sizes
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        # Create directories
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        # Save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "features_list": features_list,
                "sequence_length": sequence_length,
                "input_size": len(features_list),
                "hidden_sizes": hidden_sizes,
            },
            model_path,
        )

        # Save scaler
        joblib.dump(scaler, scaler_path)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    @staticmethod
    def load_model_checkpoint(
        model_path: str = "models/lstm_model.pth", device: torch.device = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint

        Args:
            model_path: Path to model file
            device: Device to load model on

        Returns:
            Dictionary with model checkpoint data
        """
        if device is None:
            device = torch.device("cpu")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=device)
        return checkpoint

    @staticmethod
    def load_scaler(scaler_path: str = "models/scaler.pkl") -> MinMaxScaler:
        """
        Load scaler from file

        Args:
            scaler_path: Path to scaler file

        Returns:
            Loaded MinMaxScaler
        """
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        return joblib.load(scaler_path)
