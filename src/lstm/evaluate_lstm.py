from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class EvaluateLSTM:

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict[str, float]: Statistical metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error
        mape = (
            np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        )

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

    def evaluate_model(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        return self.calculate_metrics(y_true, y_pred)

    @staticmethod
    def print_evaluation_report(
        metrics: Dict[str, float], title: str = "Model Evaluation"
    ):
        """
        Args:
            metrics (Dict[str, float]): Evaluation metrics
            title (str): Report title
        """
        print(f"\n{'='*40}")
        print(f"{title:^40}")
        print(f"{'='*40}")

        print(f"RMSE:      {metrics.get('rmse', 0):.4f}")
        print(f"MAE:       {metrics.get('mae', 0):.4f}")
        print(f"R^2 Score:  {metrics.get('r2', 0):.4f}")
        print(f"MAPE:      {metrics.get('mape', 0):.2f}%")

        print(f"{'='*40}\n")
