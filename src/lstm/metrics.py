"""
Metrics calculation utilities for LSTM prediction
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricsCalculator:
    """Handles calculation of evaluation metrics for model performance"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary with calculated metrics
        """
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
