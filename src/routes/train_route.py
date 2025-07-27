import os
import sys
import threading
import uuid
from datetime import datetime

import pandas as pd
from flask import Blueprint, jsonify, request

from lstm.lstm_stock_price import LSTMStockPrice

from .mock_payload import get_train_mock_payload
from .training_status import training_status

# Create blueprint
train_bp = Blueprint("train", __name__)


def background_training(task_id, params):
    """Execute training in background thread"""

    try:
        training_status.mark_running()

        print(f"[{task_id}] Starting LSTM training with params: {params}")

        # Load data
        print(f"[{task_id}] Loading data...")
        try:
            # TODO: replace with filepath provided in request
            df = pd.read_csv("data/AAPL_2018-01-01_to_2025-07-01.csv")
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df.set_index("Date", inplace=True)
        except FileNotFoundError:
            raise Exception(
                "Data file not found: data/AAPL_2018-01-01_to_2025-07-01.csv"
            )
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

        # Initialize LSTM model with parameters
        print(f"[{task_id}] Initializing LSTM model...")
        model = LSTMStockPrice(
            sequence_length=params.get("sequence_length", 90),
            hidden_sizes=params.get("hidden_sizes", [128, 64]),
            dropout=params.get("dropout", 0.2),
        )

        # Prepare data with parameters
        print(f"[{task_id}] Preparing data...")
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            df,
            test_size=params.get("test_size", 0.3),
            val_size=params.get("val_size", 0.2),
        )

        print(f"[{task_id}] Training samples: {len(X_train)}")
        print(f"[{task_id}] Validation samples: {len(X_val)}")
        print(f"[{task_id}] Test samples: {len(X_test)}")

        # Train model with parameters
        print(f"[{task_id}] Training model...")
        model.train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=params.get("epochs", 50),
            batch_size=params.get("batch_size", 32),
            learning_rate=params.get("learning_rate", 0.001),
            verbose=1,
        )

        # Evaluate model
        print(f"[{task_id}] Evaluating model...")
        results = model.evaluate(X_test, y_test)

        # Save model
        print(f"[{task_id}] Saving model...")
        model.save_model()

        # Mark training as completed with results
        results_data = {
            "mse": results.get("mse"),
            "rmse": results.get("rmse"),
            "r2": results.get("r2"),
            "mae": results.get("mae"),
            "mape": results.get("mape"),
        }
        training_status.mark_completed(results_data)

        print(f"[{task_id}] Training completed successfully")

    except Exception as e:
        # Mark training as failed
        training_status.mark_failed(str(e))
        print(f"[{task_id}] Training failed: {str(e)}")


@train_bp.route("/train", methods=["POST"])
def start_training():
    """
    Start LSTM model training

    JSON Parameters (all optional):
    - epochs: int (default: 50) - Number of training epochs
    - sequence_length: int (default: 90) - Length of input sequences
    - test_size: float (default: 0.3) - Proportion of data for testing (0-1)
    - val_size: float (default: 0.2) - Proportion of remaining data for validation (0-1)
    - batch_size: int (default: 32) - Training batch size
    - learning_rate: float (default: 0.001) - Learning rate for optimizer
    - hidden_sizes: list (default: [128, 64]) - Hidden layer sizes for LSTM
    - dropout: float (default: 0.2) - Dropout rate (0-1)

    Example request:
    {
        "epochs": 100,
        "sequence_length": 60,
        "batch_size": 64,
        "learning_rate": 0.001,
        "hidden_sizes": [256, 128],
        "dropout": 0.3
    }
    """

    # Check if training is already running
    if training_status.is_running():
        current_status = training_status.get_status()
        return (
            jsonify(
                {
                    "error": "Training already in progress",
                    "message": "Please wait for current training to complete",
                    "current_task_id": current_status["task_id"],
                    "status": current_status["status"],
                }
            ),
            409,
        )

    # Get training parameters from request (optional)
    try:
        data = request.get_json(force=True, silent=True) or {}

        # Use mock payload if no data provided for testing purposes
        if not data:
            data = get_train_mock_payload()
            print(f"Using mock payload for testing: {data}")
    except:
        data = get_train_mock_payload()
        print(f"Using mock payload for testing: {data}")

    # Validate and set parameters with defaults
    try:
        params = {
            "epochs": max(1, int(data.get("epochs", 50))),
            "sequence_length": max(10, int(data.get("sequence_length", 90))),
            "test_size": max(0.1, min(0.5, float(data.get("test_size", 0.3)))),
            "val_size": max(0.1, min(0.5, float(data.get("val_size", 0.2)))),
            "batch_size": max(1, int(data.get("batch_size", 32))),
            "learning_rate": max(
                0.00001, min(0.1, float(data.get("learning_rate", 0.001)))
            ),
            "hidden_sizes": data.get("hidden_sizes", [128, 64]),
            "dropout": max(0.0, min(0.8, float(data.get("dropout", 0.2)))),
        }

        # Validate hidden_sizes
        if (
            not isinstance(params["hidden_sizes"], list)
            or len(params["hidden_sizes"]) < 1
        ):
            params["hidden_sizes"] = [128, 64]

        # Ensure hidden sizes are positive integers
        params["hidden_sizes"] = [max(1, int(size)) for size in params["hidden_sizes"]]

    except (ValueError, TypeError) as e:
        return (
            jsonify(
                {
                    "error": "Invalid parameter values",
                    "message": f"Parameter validation failed: {str(e)}",
                    "valid_ranges": {
                        "epochs": "positive integer",
                        "sequence_length": "integer >= 10",
                        "test_size": "float between 0.1 and 0.5",
                        "val_size": "float between 0.1 and 0.5",
                        "batch_size": "positive integer",
                        "learning_rate": "float between 0.00001 and 0.1",
                        "hidden_sizes": "list of positive integers",
                        "dropout": "float between 0.0 and 0.8",
                    },
                }
            ),
            400,
        )

    # Generate new task ID
    task_id = str(uuid.uuid4())

    # Initialize training status using singleton
    training_status.start_training(task_id, params)

    # Start background thread
    thread = threading.Thread(
        target=background_training, args=(task_id, params), daemon=True
    )
    thread.start()

    return (
        jsonify(
            {
                "message": "Training started successfully",
                "task_id": task_id,
                "status": "started",
                "parameters": params,
                "estimated_duration": "5-15 minutes",
            }
        ),
        202,
    )
