from datetime import datetime

from flask import Blueprint, jsonify

# Create blueprint
health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return (
        jsonify(
            {
                "status": "ok",
                "message": "LSTM API is running!",
                "timestamp": datetime.now().isoformat(),
                "service": "LSTM Stock Price Prediction API",
            }
        ),
        200,
    )
