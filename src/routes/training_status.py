"""
Training Status Singleton
Manages the global training status across all routes
"""

import threading
from datetime import datetime
from typing import Any, Dict, Optional


class TrainingStatusSingleton:
    """Singleton class to manage training status across the application"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._status = {
                "is_running": False,
                "task_id": None,
                "started_at": None,
                "completed_at": None,
                "status": "idle",
                "results": None,
            }
            self._lock = threading.Lock()
            self._initialized = True

    def get_status(self) -> Dict[str, Any]:
        """Get current training status"""
        with self._lock:
            return self._status.copy()

    def update_status(self, updates: Dict[str, Any]) -> None:
        """Update training status with new values"""
        with self._lock:
            self._status.update(updates)

    def start_training(self, task_id: str, params: Dict[str, Any]) -> None:
        """Initialize status for new training session"""
        with self._lock:
            self._status.update(
                {
                    "is_running": True,
                    "task_id": task_id,
                    "status": "starting",
                    "started_at": None,
                    "completed_at": None,
                    "results": None,
                }
            )

    def mark_running(self) -> None:
        """Mark training as actively running"""
        with self._lock:
            self._status.update(
                {
                    "status": "running",
                    "started_at": datetime.now().isoformat(),
                }
            )

    def mark_completed(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Mark training as completed"""
        with self._lock:
            self._status.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "is_running": False,
                    "results": results,
                }
            )

    def mark_failed(self, error: str) -> None:
        """Mark training as failed"""
        with self._lock:
            self._status.update(
                {
                    "status": "failed",
                    "completed_at": datetime.now().isoformat(),
                    "is_running": False,
                    "results": {"error": error},
                }
            )

    def reset(self) -> None:
        """Reset to idle state"""
        with self._lock:
            self._status = {
                "is_running": False,
                "task_id": None,
                "started_at": None,
                "completed_at": None,
                "status": "idle",
                "results": None,
            }

    def is_running(self) -> bool:
        """Check if training is currently running"""
        with self._lock:
            return self._status["is_running"]


# Global instance
training_status = TrainingStatusSingleton()
