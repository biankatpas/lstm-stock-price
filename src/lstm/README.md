# LSTM Package Structure

This package provides modular components for data processing, neural network training, evaluation metrics, and experiment tracking.

## File Structure

### `model.py`
- **Main Class**: `LSTMStockPrice`
- **Responsibility**: LSTM neural network architecture definition and training logic

### `data_processor.py`
- **Class**: `DataProcessor`
- **Responsibility**: Data processing and preparation

### `metrics.py`
- **Class**: `MetricsCalculator`
- **Responsibility**: Evaluation metrics calculation
- **Metrics**:
  - MSE, RMSE, MAE, R², MAPE

### `mlflow_manager.py`
- **Class**: `MLflowManager`
- **Responsibility**: MLflow management for experiment tracking

### `model_io.py`
- **Class**: `ModelIO`
- **Responsibility**: Model persistence
