from typing import Tuple

import numpy as np


class PrepareSequence:

    def __init__(self, sequence_length: int = 60):
        """
        Args:
            sequence_length (int): Number of time steps to look back
        """
        self.sequence_length = sequence_length
        self._validate_sequence_length()

    def _validate_sequence_length(self):
        """Validate sequence length parameter."""
        if self.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")
        if self.sequence_length > 252:  # More than 1 trading year
            print(f"Warning: Sequence length {self.sequence_length} is very long")

    def create_sequences(
        self, data: np.ndarray, target_column_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            data (np.ndarray): Scaled feature data
            target_column_index (int): Index of target column

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) sequences
        """
        if len(data) <= self.sequence_length:
            raise ValueError(
                f"Data length {len(data)} must be greater than sequence length {self.sequence_length}"
            )

        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(data[i, target_column_index])

        return np.array(X), np.array(y)

    def split_sequences(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            X (np.ndarray): Feature sequences
            y (np.ndarray): Target sequences
            test_size (float): Proportion of data for testing

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        if not 0 < test_size < 1:
            raise ValueError("Test size must be between 0 and 1")

        split_index = int(len(X) * (1 - test_size))

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test

    def create_train_test_sequences(
        self, data: np.ndarray, target_column_index: int = 0, test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            data (np.ndarray): Scaled feature data
            target_column_index (int): Index of target column
            test_size (float): Proportion of data for testing

        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        X, y = self.create_sequences(data, target_column_index)
        return self.split_sequences(X, y, test_size)
