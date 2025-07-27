"""
Data processing utilities for LSTM stock prediction
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from utils.logger_config import get_logger

# Get logger
logger = get_logger(__name__)


class DataProcessor:
    """Handles data preprocessing and sequence creation for LSTM models"""

    def __init__(self, features_list: list):
        self.features_list = features_list
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_fitted = False

    def create_sequences(
        self, data: np.ndarray, target_column_index: int, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series data

        Args:
            data: Scaled feature data
            target_column_index: Index of target column
            sequence_length: Length of input sequences

        Returns:
            Tuple of (X, y) sequences
        """
        if len(data) <= sequence_length:
            raise ValueError(
                f"Data length {len(data)} must be greater than sequence length {sequence_length}"
            )

        X, y = [], []

        for i in range(sequence_length, len(data)):
            # Feature sequence: look back sequence_length steps
            X.append(data[i - sequence_length : i])
            # Target: next value of target column
            y.append(data[i, target_column_index])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        sequence_length: int = 30,
        val_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and scale data for training with train/validation/test splits
        Uses sklearn's TimeSeriesSplit for proper temporal data splitting

        Args:
            data: Input DataFrame
            target_column: Target column name
            sequence_length: Length of input sequences
            val_size: Proportion of remaining data for validation

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

        # Create sequences
        X, y = self.create_sequences(scaled_data, target_index, sequence_length)

        # Using sklearn's TimeSeriesSplit for proper temporal data splitting
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

        logger.info("Split method: TimeSeriesSplit")
        logger.info(f"Train: {len(X_train)} samples")
        logger.info(f"Validation: {len(X_val)} samples")
        logger.info(f"Test: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def inverse_scale_predictions(
        self, scaled_values: np.ndarray, target_index: int
    ) -> np.ndarray:
        """Inverse transform scaled predictions to original scale"""
        if not self.scaler_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")

        # Create array with same shape as original features
        dummy_array = np.zeros((scaled_values.shape[0], len(self.features_list)))
        dummy_array[:, target_index] = scaled_values.flatten()

        # Inverse transform and extract target column
        inverse_transformed = self.scaler.inverse_transform(dummy_array)
        return inverse_transformed[:, target_index]
