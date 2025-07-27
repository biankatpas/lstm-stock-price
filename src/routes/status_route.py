import os
import sys
from datetime import datetime

from flask import Blueprint, jsonify

from utils.training_status import training_status

# Create blueprint
status_bp = Blueprint("status", __name__)


@status_bp.route("/train/status", methods=["GET"])
def get_training_status():
    """Get current training status"""

    status_data = training_status.get_status()
    response_data = {
        **status_data,
        "timestamp": datetime.now().isoformat(),
    }

    if status_data["status"] == "failed":
        return jsonify(response_data), 500
    elif status_data["status"] == "running":
        return jsonify(response_data), 202
    else:
        return jsonify(response_data), 200
