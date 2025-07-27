import os

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from lstm.model import LSTMStockPrice
from mocks.mock_data import generate_mock_stock_data
from mocks.mock_payload import get_predict_future_mock_payload

# Create blueprint
predict_bp = Blueprint("predict", __name__)

# Global model instance
_model = None


def _load_model():
    """Load the trained model if not already loaded"""
    global _model

    if _model is None:
        model_path = "models/lstm_model.pth"
        scaler_path = "models/scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                "Trained model not found. Please train the model first."
            )

        # Initialize model and load weights
        _model = LSTMStockPrice()
        _model.load_model(model_path, scaler_path)

    return _model


@predict_bp.route("/predict", methods=["POST"])
def predict_stock_price():
    """
    Make stock price predictions using the trained model

    JSON Parameters:
    - data: list of lists (required) - Historical stock data for prediction
      Each inner list should contain [Open, High, Low, Close, Volume] values
      Must provide at least 'sequence_length' number of data points (default: 90)

    Example request:
    {
        "data": [
            [150.0, 155.0, 149.0, 152.0, 1000000],
            [152.0, 156.0, 151.0, 154.0, 1100000],
            ...
            (at least 90 rows for default sequence_length)
        ]
    }
    """

    try:
        # Load model
        model = _load_model()

        # Get request data
        data = request.get_json(force=True, silent=True) or {}

        # Use mock data if no data provided for testing purposes
        if not data or "data" not in data:
            input_data = generate_mock_stock_data(90)
            print("Using mock data for testing (90 data points)")
        else:
            input_data = data["data"]
        if not isinstance(input_data, list):
            return (
                jsonify(
                    {
                        "error": "Invalid data format",
                        "message": "Data must be a list of lists",
                    }
                ),
                400,
            )

        if len(input_data) < model.sequence_length:
            return (
                jsonify(
                    {
                        "error": "Insufficient data",
                        "message": f"Need at least {model.sequence_length} data points for prediction",
                    }
                ),
                400,
            )

        # Convert to DataFrame for processing
        df = pd.DataFrame(input_data, columns=model.features_list)

        # Validate columns
        if not all(col in df.columns for col in model.features_list):
            return (
                jsonify(
                    {
                        "error": "Invalid data columns",
                        "message": f"Data must contain columns: {model.features_list}",
                    }
                ),
                400,
            )

        # Scale the data using the fitted scaler
        scaled_data = model.scaler.transform(df[model.features_list].values)

        # Prepare input sequence (last sequence_length points)
        X_input = scaled_data[-model.sequence_length :].reshape(
            1, model.sequence_length, -1
        )

        # Make prediction
        prediction = model.predict(X_input)

        return (
            jsonify(
                {
                    "message": "Prediction successful",
                    "predicted_price": float(prediction[0]),
                    "input_data_points": len(input_data),
                    "model_features": model.features_list,
                    "sequence_length_used": model.sequence_length,
                    "using_mock_data": not bool(data and "data" in data),
                }
            ),
            200,
        )

    except FileNotFoundError as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404

    except ValueError as e:
        return jsonify({"error": "Data validation error", "message": str(e)}), 400

    except Exception as e:
        return (
            jsonify(
                {
                    "error": "Prediction failed",
                    "message": f"An error occurred during prediction: {str(e)}",
                }
            ),
            500,
        )


@predict_bp.route("/predict/future", methods=["POST"])
def predict_future_prices():
    """
    Predict future stock prices for specified number of days

    JSON Parameters:
    - data: list of lists (required) - Historical stock data for prediction
      Each inner list should contain [Open, High, Low, Close, Volume] values
      Must provide at least 'sequence_length' number of data points (default: 90)
    - days: int (optional, default: 30) - Number of future days to predict

    Example request:
    {
        "data": [
            [150.0, 155.0, 149.0, 152.0, 1000000],
            [152.0, 156.0, 151.0, 154.0, 1100000],
            ...
            (at least 90 rows for default sequence_length)
        ],
        "days": 7
    }
    """

    try:
        # Load model
        model = _load_model()

        # Get request data
        data = request.get_json(force=True, silent=True) or {}

        # Use mock data if no data provided for testing purposes
        if not data or "data" not in data:
            input_data = generate_mock_stock_data(90)
            # Check if days parameter was provided, if not use mock payload
            if not data:
                mock_payload = get_predict_future_mock_payload()
                days = mock_payload["days"]
                print(
                    f"Using full mock payload for testing (90 data points, predicting {days} days)"
                )
            else:
                days = data.get("days", 7)  # Default to 7 days for testing
                print(
                    f"Using mock data for testing (90 data points, predicting {days} days)"
                )
        else:
            input_data = data["data"]
            days = data.get("days", 30)

        # Get prediction parameters

        # Validate parameters
        if not isinstance(input_data, list):
            return (
                jsonify(
                    {
                        "error": "Invalid data format",
                        "message": "Data must be a list of lists",
                    }
                ),
                400,
            )

        if len(input_data) < model.sequence_length:
            return (
                jsonify(
                    {
                        "error": "Insufficient data",
                        "message": f"Need at least {model.sequence_length} data points for prediction",
                    }
                ),
                400,
            )

        if not isinstance(days, int) or days < 1 or days > 365:
            return (
                jsonify(
                    {
                        "error": "Invalid days parameter",
                        "message": "Days must be an integer between 1 and 365",
                    }
                ),
                400,
            )

        # Convert to DataFrame for processing
        df = pd.DataFrame(input_data, columns=model.features_list)

        # Validate columns
        if not all(col in df.columns for col in model.features_list):
            return (
                jsonify(
                    {
                        "error": "Invalid data columns",
                        "message": f"Data must contain columns: {model.features_list}",
                    }
                ),
                400,
            )

        # Scale the data using the fitted scaler
        scaled_data = model.scaler.transform(df[model.features_list].values)

        # Prepare input sequence (last sequence_length points)
        last_sequence = scaled_data[-model.sequence_length :]

        # Make future predictions
        future_predictions = model.predict_future(last_sequence, days)

        # Convert to list for JSON serialization
        predictions_list = [float(pred) for pred in future_predictions]

        return (
            jsonify(
                {
                    "message": "Future predictions successful",
                    "predicted_prices": predictions_list,
                    "prediction_days": days,
                    "input_data_points": len(input_data),
                    "model_features": model.features_list,
                    "sequence_length_used": model.sequence_length,
                    "using_mock_data": not bool(data and "data" in data),
                    "using_mock_payload": not bool(
                        request.get_json(force=True, silent=True)
                    ),
                }
            ),
            200,
        )

    except FileNotFoundError as e:
        return jsonify({"error": "Model not found", "message": str(e)}), 404

    except ValueError as e:
        return jsonify({"error": "Data validation error", "message": str(e)}), 400

    except Exception as e:
        return (
            jsonify(
                {
                    "error": "Prediction failed",
                    "message": f"An error occurred during prediction: {str(e)}",
                }
            ),
            500,
        )
