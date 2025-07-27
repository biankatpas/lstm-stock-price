"""
MLflow management utilities
"""

import os
from typing import Any, Dict

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

from utils.logger_config import get_logger

# Get logger
logger = get_logger(__name__)

# Silence Git warnings from MLflow
os.environ["GIT_PYTHON_REFRESH"] = "quiet"


class MLflowManager:
    """Handles MLflow experiment tracking and logging"""

    def __init__(self, experiment_name: str = "lstm_stock_prediction"):
        self.experiment_name = experiment_name
        self.setup_experiment()

    def setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow setup warning: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        if mlflow.active_run():
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        """Log metrics to MLflow"""
        if mlflow.active_run():
            if step is not None:
                mlflow.log_metrics(metrics, step=step)
            else:
                mlflow.log_metrics(metrics)

    def log_model(self, model, input_example, signature=None):
        """Log PyTorch model to MLflow"""
        if mlflow.active_run():
            if signature is None and input_example is not None:
                # Create signature automatically if not provided
                import torch

                model.eval()
                with torch.no_grad():
                    device = next(model.parameters()).device
                    example_input = torch.FloatTensor(input_example).to(device)
                    example_output = model(example_input).cpu().numpy()
                    signature = infer_signature(input_example, example_output)

            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
            )
