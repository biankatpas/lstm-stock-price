import os
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Silence Git warnings from MLflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# MLflow imports
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


class LSTMStockPrice(nn.Module):
    """LSTM Stock Price Predictor"""

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

        # Scaler for data normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fitted = False

        # Build neural network layers
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

        # Dropout para LSTM
        self.lstm_dropout = nn.Dropout(dropout)

        # Dense layers using Sequential
        self.dense_layers = nn.Sequential(
            nn.Linear(self.hidden_sizes[1], 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(25, 1),
        )

        # Move to device
        self.to(self.device)

        # Setup MLflow
        self.experiment_name = "lstm_stock_prediction"
        self._setup_mlflow()

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

    def _setup_mlflow(self):
        """Setup Experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            print(f"Setup warning: {e}")

    def _create_sequences(
        self, data: np.ndarray, target_column_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series data

        Args:
            data: Scaled feature data
            target_column_index: Index of target column

        Returns:
            Tuple of (X, y) sequences
        """
        if len(data) <= self.sequence_length:
            raise ValueError(
                f"Data length {len(data)} must be greater than sequence length {self.sequence_length}"
            )

        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            # Feature sequence: look back sequence_length steps
            X.append(data[i - self.sequence_length : i])
            # Target: next value of target column
            y.append(data[i, target_column_index])

        return np.array(X), np.array(y)

    def _split_sequences(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split sequences into train and test sets

        Args:
            X: Feature sequences
            y: Target sequences
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if not 0 < test_size < 1:
            raise ValueError("Test size must be between 0 and 1")

        # Use temporal split (no shuffling for time series)
        split_index = int(len(X) * (1 - test_size))

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        test_size: float = 0.3,
        val_size: float = 0.2,
        use_time_series_split: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and scale data for training with train/validation/test splits

        Args:
            data: Input DataFrame
            target_column: Target column name
            test_size: Proportion for test set
            val_size: Proportion of remaining data for validation
            use_time_series_split: Whether to use sklearn's TimeSeriesSplit

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """

        # Select and clean features
        feature_data = data[self.features_list].copy()
        feature_data = feature_data.dropna()

        # Scale the features
        scaled_data = self.scaler.fit_transform(feature_data.values)
        self.scaler_fitted = True

        # Get target column index
        target_index = self.features_list.index(target_column)

        # Create sequences using internal methods
        X, y = self._create_sequences(scaled_data, target_index)

        if use_time_series_split:
            # Using sklearn's TimeSeriesSplit (more robust)
            tscv = TimeSeriesSplit(n_splits=3)
            splits = list(tscv.split(X))

            # Get the last split which gives us the largest training set
            train_idx, test_idx = splits[-1]

            # Further split training into train/validation
            val_split_point = int(len(train_idx) * (1 - val_size))

            final_train_idx = train_idx[:val_split_point]
            val_idx = train_idx[val_split_point:]

            X_train = X[final_train_idx]
            y_train = y[final_train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

        else:
            # Custom temporal split (current implementation)
            # First split: separate test set
            test_split_idx = int(len(X) * (1 - test_size))
            X_temp = X[:test_split_idx]
            y_temp = y[:test_split_idx]
            X_test = X[test_split_idx:]
            y_test = y[test_split_idx:]

            # Second split: separate train and validation from remaining data
            val_split_idx = int(len(X_temp) * (1 - val_size))
            X_train = X_temp[:val_split_idx]
            y_train = y_temp[:val_split_idx]
            X_val = X_temp[val_split_idx:]
            y_val = y_temp[val_split_idx:]

        print(
            f"Split method: {'TimeSeriesSplit' if use_time_series_split else 'Custom temporal split'}"
        )
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

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
            mlflow.log_params(
                {
                    "sequence_length": self.sequence_length,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "features_count": len(self.features_list),
                    "model_type": "LSTM_PyTorch",
                }
            )

            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)

            # Create DataLoader
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            # Validation data if provided
            if X_val is not None and y_val is not None:
                X_val_tensor = torch.FloatTensor(X_val).to(self.device)
                y_val_tensor = torch.FloatTensor(y_val).to(self.device)

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
                        print(
                            f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
                        )
                    else:
                        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

                # Logging every 10 epochs
                if (epoch + 1) % 10 == 0:
                    metrics_to_log = {"train_loss": avg_loss}
                    if X_val is not None:
                        metrics_to_log["val_loss"] = val_loss
                    mlflow.log_metrics(metrics_to_log, step=epoch)

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
                            print(f"Early stopping at epoch {epoch+1}")
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
            mlflow.log_metrics(final_metrics)

            # Create input example for model signature
            input_example = X_train[:1]

            # Create model signature by making a prediction
            self.eval()
            with torch.no_grad():
                example_input = torch.FloatTensor(input_example).to(self.device)
                example_output = self(example_input).cpu().numpy()

            # Infer signature from input/output
            signature = infer_signature(input_example, example_output)

            # Log model with signature and input example
            mlflow.pytorch.log_model(
                pytorch_model=self,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            self.is_trained = True
            return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions_scaled = self(X_tensor).cpu().numpy()

        # Inverse transform predictions to original scale
        target_index = self.features_list.index("Close")
        return self._inverse_scale_predictions(predictions_scaled, target_index)

    def _inverse_scale_predictions(
        self, scaled_values: np.ndarray, target_index: int
    ) -> np.ndarray:
        """Inverse transform scaled predictions to original scale"""
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")

        # Create dummy array with same shape as original features
        dummy_array = np.zeros((scaled_values.shape[0], len(self.features_list)))
        dummy_array[:, target_index] = scaled_values.flatten()

        # Inverse transform and extract target column
        inverse_transformed = self.scaler.inverse_transform(dummy_array)
        return inverse_transformed[:, target_index]

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE calculation with zero handling
        mape = (
            np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        )

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        }

    def _print_evaluation_report(
        self, metrics: Dict[str, float], title: str = "Model Evaluation"
    ):
        """Print evaluation report with performance interpretation"""
        print(f"\n{'='*50}")
        print(f"{title:^50}")
        print(f"{'='*50}")

        print(f"RMSE:       {metrics.get('rmse', 0):.4f}")
        print(f"MAE:        {metrics.get('mae', 0):.4f}")
        print(f"R^2 Score:   {metrics.get('r2', 0):.4f}")
        print(f"MAPE:       {metrics.get('mape', 0):.2f}%")
        print(f"MSE:        {metrics.get('mse', 0):.4f}")

        # Performance interpretation
        r2 = metrics.get("r2", 0)
        if r2 >= 0.8:
            performance = "Excellent"
        elif r2 >= 0.6:
            performance = "Good"
        elif r2 >= 0.4:
            performance = "Fair"
        else:
            performance = "Poor"

        print(f"Performance: {performance} (R^2 = {r2:.4f})")
        print(f"{'='*50}\n")

    def _log_training_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "train",
        step: int = None,
    ):
        """Log training metrics to MLflow"""
        if mlflow.active_run():
            metrics = self._calculate_metrics(y_true, y_pred)
            mlflow_metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
            if step is not None:
                mlflow.log_metrics(mlflow_metrics, step=step)
            else:
                mlflow.log_metrics(mlflow_metrics)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance on test data"""
        predictions = self.predict(X_test)
        target_index = self.features_list.index("Close")
        y_test_actual = self._inverse_scale_predictions(
            y_test.reshape(-1, 1), target_index
        )

        # Calculate metrics
        metrics = self._calculate_metrics(y_test_actual, predictions)

        # Log metrics to MLflow if run is active
        if mlflow.active_run():
            mlflow_metrics = {f"test_{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(mlflow_metrics)

        return metrics

    def predict_future(self, last_sequence: np.ndarray, days: int = 30) -> np.ndarray:
        """Predict future stock prices"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = []
        current_sequence = last_sequence.copy()
        target_index = self.features_list.index("Close")

        self.eval()
        with torch.no_grad():
            for _ in range(days):
                # Convert to tensor
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

        return self._inverse_scale_predictions(
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

        # Create directories
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        # Save model
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "features_list": self.features_list,
                "sequence_length": self.sequence_length,
                "input_size": len(self.features_list),
                "hidden_sizes": self.hidden_sizes,
            },
            model_path,
        )

        # Save scaler
        joblib.dump(self.scaler, scaler_path)

        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

    def load_model(
        self,
        model_path: str = "models/lstm_model.pth",
        scaler_path: str = "models/scaler.pkl",
    ):
        """Load model and scaler"""
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # Set model attributes
            self.features_list = checkpoint["features_list"]
            self.sequence_length = checkpoint["sequence_length"]
            self.hidden_sizes = checkpoint["hidden_sizes"]

            # Rebuild the network layers if not already built
            if not hasattr(self, "lstm1"):
                input_size = len(self.features_list)
                dropout = 0.2

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

                # Move to device
                self.to(self.device)

            # Load model state
            self.load_state_dict(checkpoint["model_state_dict"])

            # Load scaler
            self.scaler = joblib.load(scaler_path)
            self.scaler_fitted = True
            self.is_trained = True

            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
        else:
            raise FileNotFoundError("Model or scaler file not found")
