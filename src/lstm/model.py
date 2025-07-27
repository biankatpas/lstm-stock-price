"""
LSTM model definition
"""

import warnings
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.logger_config import get_logger

# Get logger
logger = get_logger(__name__)

from .data_processor import DataProcessor
from .metrics import MetricsCalculator
from .mlflow_manager import MLflowManager
from .model_io import ModelIO

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LSTMStockPrice(nn.Module):
    """LSTM Stock Price Model"""

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.sequence_length = sequence_length
        self.features_list = ["Open", "High", "Low", "Close", "Volume"]
        self.device = torch.device("cpu")
        self.is_trained = False
        self.hidden_sizes = hidden_sizes or [128, 64]

        # Initialize components
        self.data_processor = DataProcessor(self.features_list)
        self.metrics_calculator = MetricsCalculator()
        self.mlflow_manager = MLflowManager()
        self.model_io = ModelIO()

        # Build neural network layers
        self._build_network(dropout)

        # Move to device
        self.to(self.device)

    def _build_network(self, dropout: float):
        """Build the neural network architecture"""
        input_size = len(self.features_list)

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size, self.hidden_sizes[0], batch_first=True, dropout=dropout
        )
        self.lstm2 = nn.LSTM(
            self.hidden_sizes[0],
            self.hidden_sizes[1],
            batch_first=True,
            dropout=dropout,
        )

        # Dropout for LSTM
        self.lstm_dropout = nn.Dropout(dropout)

        # Dense layers using Sequential
        self.dense_layers = nn.Sequential(
            nn.Linear(self.hidden_sizes[1], 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        """Forward pass through the network"""
        # First LSTM layer
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.lstm_dropout(lstm_out)

        # Second LSTM layer
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)

        # Take the last output from the sequence
        lstm_out = lstm_out[:, -1, :]

        # Pass through dense layers (Sequential)
        output = self.dense_layers(lstm_out)

        return output

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        val_size: float = 0.2,
    ):
        """Prepare data for training using TimeSeriesSplit"""
        return self.data_processor.prepare_data(
            data,
            target_column,
            self.sequence_length,
            val_size,
        )

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: int = 1,
    ) -> Dict:
        """Train the LSTM model"""

        # Setup optimizer and loss if not already set
        if not hasattr(self, "optimizer"):
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            self.mlflow_manager.log_params(
                {
                    "sequence_length": self.sequence_length,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "features_count": len(self.features_list),
                    "model_type": "LSTM_PyTorch",
                }
            )

            # Convert to tensors with explicit float32
            X_train_tensor = torch.FloatTensor(X_train.astype(np.float32)).to(
                self.device
            )
            y_train_tensor = torch.FloatTensor(y_train.astype(np.float32)).to(
                self.device
            )

            # Create DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            # Validation data if provided
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val.astype(np.float32)).to(
                    self.device
                )
                y_val_tensor = torch.FloatTensor(y_val.astype(np.float32)).to(
                    self.device
                )

            # Training loop
            history = {"loss": [], "val_loss": []}
            best_val_loss = float("inf")
            patience_counter = 0
            patience = 15

            for epoch in range(epochs):
                # Training phase
                self.train()
                epoch_loss = 0.0

                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = self.criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(train_loader)
                history["loss"].append(avg_loss)

                # Validation phase
                val_loss = 0.0
                if X_val is not None:
                    self.eval()
                    with torch.no_grad():
                        val_outputs = self(X_val_tensor)
                        val_loss = self.criterion(
                            val_outputs.squeeze(), y_val_tensor
                        ).item()
                        history["val_loss"].append(val_loss)

                # Logging
                if verbose and (epoch + 1) % 10 == 0:
                    if X_val is not None:
                        logger.info(
                            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
                        )
                    else:
                        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

                # Log metrics every 10 epochs
                if (epoch + 1) % 10 == 0:
                    metrics_to_log = {"train_loss": avg_loss}
                    if X_val is not None:
                        metrics_to_log["val_loss"] = val_loss
                    self.mlflow_manager.log_metrics(metrics_to_log, step=epoch)

                # Early stopping
                if X_val is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        self.best_model_state = self.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            # Restore best model
                            self.load_state_dict(self.best_model_state)
                            break

            # Final metrics
            final_metrics = {
                "final_train_loss": history["loss"][-1],
                "epochs_trained": len(history["loss"]),
            }
            if history["val_loss"]:
                final_metrics["final_val_loss"] = history["val_loss"][-1]
            self.mlflow_manager.log_metrics(final_metrics)

            # Log model
            input_example = X_train[:1].astype(np.float32)
            self.mlflow_manager.log_model(self, input_example)

            self.is_trained = True
            return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.eval()
        with torch.no_grad():
            # Ensure input is float32 for consistency
            X_float32 = X.astype(np.float32)
            X_tensor = torch.FloatTensor(X_float32).to(self.device)
            predictions_scaled = self(X_tensor).cpu().numpy()

        # Inverse transform predictions to original scale
        target_index = self.features_list.index("Close")
        return self.data_processor.inverse_scale_predictions(
            predictions_scaled, target_index
        )

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance on test data"""
        predictions = self.predict(X_test)
        target_index = self.features_list.index("Close")
        y_test_actual = self.data_processor.inverse_scale_predictions(
            y_test.reshape(-1, 1), target_index
        )

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_test_actual, predictions)

        # Log metrics to MLflow if run is active
        if mlflow.active_run():
            mlflow_metrics = {f"test_{k}": v for k, v in metrics.items()}
            self.mlflow_manager.log_metrics(mlflow_metrics)

        return metrics

    def predict_future(self, last_sequence: np.ndarray, days: int = 30) -> np.ndarray:
        """Predict future stock prices"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = []
        current_sequence = last_sequence.copy().astype(np.float32)
        target_index = self.features_list.index("Close")

        self.eval()
        with torch.no_grad():
            for _ in range(days):
                # Convert to tensor with explicit float32
                sequence_tensor = (
                    torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                )

                # Predict next value
                pred = self(sequence_tensor).cpu().numpy()
                predictions.append(pred[0, 0])

                # Update sequence
                new_row = current_sequence[-1].copy()
                new_row[target_index] = pred[0, 0]

                # Slide window
                current_sequence = np.vstack([current_sequence[1:], new_row])

        return self.data_processor.inverse_scale_predictions(
            np.array(predictions).reshape(-1, 1), target_index
        )

    def save_model(
        self,
        model_path: str = "models/lstm_model.pth",
        scaler_path: str = "models/scaler.pkl",
    ):
        """Save model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        self.model_io.save_model(
            self,
            self.data_processor.scaler,
            self.features_list,
            self.sequence_length,
            self.hidden_sizes,
            model_path,
            scaler_path,
        )

    def load_model(
        self,
        model_path: str = "models/lstm_model.pth",
        scaler_path: str = "models/scaler.pkl",
    ):
        """Load model and scaler"""
        # Load model checkpoint
        checkpoint = self.model_io.load_model_checkpoint(model_path, self.device)

        # Set model attributes from checkpoint
        self.features_list = checkpoint["features_list"]
        self.sequence_length = checkpoint["sequence_length"]
        self.hidden_sizes = checkpoint["hidden_sizes"]

        # Rebuild the network layers
        self._build_network(dropout=0.2)
        self.to(self.device)

        # Load model state
        self.load_state_dict(checkpoint["model_state_dict"])

        # Load scaler
        self.data_processor.scaler = self.model_io.load_scaler(scaler_path)
        self.data_processor.scaler_fitted = True
        self.is_trained = True

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
        logger.info(
            f"Model architecture: hidden_sizes={self.hidden_sizes}, sequence_length={self.sequence_length}"
        )

    @property
    def scaler(self):
        """Access to the scaler"""
        return self.data_processor.scaler
