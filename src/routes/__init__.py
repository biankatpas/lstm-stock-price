"""
Routes package for LSTM Stock Price Prediction API

This package contains the route handlers for training, status, health, scrape, and predict endpoints.
"""

from .health_route import health_bp
from .predict_route import predict_bp
from .scrape_route import scrape_bp
from .status_route import status_bp
from .train_route import train_bp

__all__ = ["train_bp", "status_bp", "health_bp", "scrape_bp", "predict_bp"]
