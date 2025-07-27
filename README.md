# LSTM Stock Price

A Flask-based API for stock price prediction using LSTM neural networks with real-time data from Yahoo Finance. This API provides a complete pipeline with data collection, model training, evaluation, and prediction capabilities.

## Key Features

- **📈 Data Collection**: Stock data scraping from Yahoo Finance
- **🧠 LSTM Neural Networks**: Uses LSTM model for time series prediction
- **🔄 Asynchronous Training**: Background model training
- **🎯 Prediction**: Single price prediction and multi-day future forecasting
- **🧪 Mock Data Support**: Built-in test data for easy API testing
- **🐳 Docker Ready**: Containerized deployment with Docker Compose

## Project Structure

```
lstm-stock-price/
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── requirements.txt           # Python dependencies
├── data/                      # CSV data storage
├── models/                    # Trained models
├── docs/                      # Documentation files
├── mlruns/                    # MLflow tracking data
└── src/                       # Source code
    ├── app.py                 # Flask application entry point
    ├── lstm/                  # LSTM model implementation (modular)
        ├── __init__.py        # Package initialization
        ├── model.py           # Main LSTM model class
        ├── data_processor.py  # Data processing and preparation
        ├── metrics.py         # Evaluation metrics
        ├── mlflow_manager.py  # MLflow experiment tracking
        ├── model_io.py        # Model persistence
        └── README.md          # LSTM package documentation
    ├── routes/                # API routes
        ├── __init__.py
        ├── health_route.py    # Health check endpoint
        ├── scrape_route.py    # Data scraping endpoint
        ├── train_route.py     # Model training endpoint
        ├── status_route.py    # Training status endpoint
        └── predict_route.py   # Prediction endpoint
    ├── scrape/                # Data collection
        ├── __init__.py
        └── yahoo_finance.py   # Yahoo Finance data downloader
    ├── mocks/                 # Mock data for testing
        ├── __init__.py
        ├── mock_data.py       # Mock data generators
        └── mock_payload.py    # Mock API payloads
    └── utils/                 # Utility functions
        ├── __init__.py
        └── training_status.py # Training status management
```

## Installation & Setup

### Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/biankatpas/lstm-stock-price.git
   cd lstm-stock-price
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker compose up
   ```

3. **Test the containerized API**
   ```bash
   curl http://localhost:5000/health
   ```

### Postman Collection

A complete Postman collection is available in the `docs/` folder for easy API testing:

- **File:** `docs/lstm-stock-price-api.postman_collection.json`
- **Import:** Import this file into Postman to get pre-configured requests for all endpoints

## Usage

### API Endpoints

#### Health Check
- **`GET /health`** - Health check endpoint
  - Returns API status and timestamp
  - Response: `200 OK`

#### Data Collection
- **`POST /scrape`** - Download stock data from Yahoo Finance
  - **Parameters:**
    - `symbol` (required): Stock symbol (e.g., 'AAPL', 'GOOGL', 'PETR4.SA')
    - `start_date` (optional): Start date (format 'YYYY-MM-DD', default: '2018-01-01')
    - `end_date` (optional): End date (format 'YYYY-MM-DD', default: '2025-07-01')
    - `interval` (optional): Data interval ('1d', '1wk', '1mo', default: '1d')
    - `output_dir` (optional): Directory to save CSV files (default: 'data')
  - **Example payload:**
    ```json
    {
      "symbol": "AAPL",
      "start_date": "2020-01-01",
      "end_date": "2025-01-01",
      "interval": "1d",
      "output_dir": "data"
    }
    ```

#### Model Training
- **`POST /train`** - Start LSTM model training
  - **Parameters (all optional):**
    - `epochs` (default: 50): Number of training epochs
    - `sequence_length` (default: 90): Length of input sequences
    - `val_size` (default: 0.2): Proportion of data for validation (0-1)
    - `batch_size` (default: 32): Training batch size
    - `learning_rate` (default: 0.001): Learning rate for optimizer
    - `hidden_sizes` (default: [128, 64]): Hidden layer sizes for LSTM
    - `dropout` (default: 0.2): Dropout rate (0-1)
    - `filepath` (default: "data/AAPL_2018-01-01_to_2025-07-01.csv"): Path to CSV data file
  - **Response:** `202 Accepted` (training runs in background)
  - **Mock data available:** Can be called without payload for testing (uses default AAPL data)
  - **Example payload:**
    ```json
    {
      "epochs": 100,
      "sequence_length": 60,
      "val_size": 0.2,
      "batch_size": 64,
      "learning_rate": 0.001,
      "hidden_sizes": [256, 128],
      "dropout": 0.3,
      "filepath": "data/GOOGL_2018-01-01_to_2025-07-01.csv"
    }
    ```

- **`GET /train/status`** - Get current training status
  - Returns training progress, status, and results
  - **Response codes:**
    - `200 OK`: Training completed successfully
    - `202 Accepted`: Training in progress
    - `500 Internal Server Error`: Training failed

#### Predictions
- **`POST /predict`** - Make single stock price prediction
  - **Parameters:**
    - `data` (required): Array of historical stock data
      - Format: `[[Open, High, Low, Close, Volume], ...]`
      - Must provide at least `sequence_length` data points (default: 90)
  - **Response:** Single predicted price value
  - **Mock data available:** Can be called without payload for testing (uses generated mock data)
  - **Example payload (minimal with 5 data points for demo):**
    ```json
    {
      "data": [
        [150.0, 155.0, 149.0, 152.0, 1000000],
        [152.0, 156.0, 151.0, 154.0, 1100000],
        [154.0, 158.0, 153.0, 157.0, 1200000],
        [157.0, 161.0, 156.0, 159.0, 1300000],
        [159.0, 163.0, 158.0, 162.0, 1400000]
      ]
    }
    ```
  - **Note:** For production use, provide at least 90 data points (default sequence_length)

- **`POST /predict/future`** - Predict future stock prices
  - **Parameters:**
    - `data` (required): Array of historical stock data (same format as `/predict`)
    - `days` (optional, default: 30): Number of future days to predict
  - **Response:** Array of predicted prices for specified number of days
  - **Mock data available:** Can be called without payload for testing (uses generated mock data with 30 days prediction)
  - **Example payload (minimal with 5 data points for demo):**
    ```json
    {
      "data": [
        [150.0, 155.0, 149.0, 152.0, 1000000],
        [152.0, 156.0, 151.0, 154.0, 1100000],
        [154.0, 158.0, 153.0, 157.0, 1200000],
        [157.0, 161.0, 156.0, 159.0, 1300000],
        [159.0, 163.0, 158.0, 162.0, 1400000]
      ],
      "days": 7
    }
    ```
  - **Note:** For production use, provide at least 90 data points (default sequence_length)

### Example Usage

```bash
# Health check
curl http://localhost:5000/health

# Download Apple stock data
curl -X POST http://localhost:5000/scrape \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2025-01-01",
    "interval": "1d"
  }'

# Start training with custom parameters
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs": 100,
    "sequence_length": 60,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_sizes": [256, 128],
    "dropout": 0.3,
    "filepath": "data/GOOGL_2018-01-01_to_2025-07-01.csv"
  }'

# Start training with default parameters (uses mock data)
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{}'

# Check training status
curl http://localhost:5000/train/status

# Make a prediction with sample data
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [150.0, 155.0, 149.0, 152.0, 1000000],
      [152.0, 156.0, 151.0, 154.0, 1100000],
      [154.0, 158.0, 153.0, 157.0, 1200000],
      [157.0, 161.0, 156.0, 159.0, 1300000],
      [159.0, 163.0, 158.0, 162.0, 1400000]
    ]
  }'

# Make a prediction using mock data (no payload needed)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{}'

# Predict future prices for 7 days
curl -X POST http://localhost:5000/predict/future \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [150.0, 155.0, 149.0, 152.0, 1000000],
      [152.0, 156.0, 151.0, 154.0, 1100000],
      [154.0, 158.0, 153.0, 157.0, 1200000],
      [157.0, 161.0, 156.0, 159.0, 1300000],
      [159.0, 163.0, 158.0, 162.0, 1400000]
    ],
    "days": 7
  }'

# Predict future prices using mock data (no payload needed)
curl -X POST http://localhost:5000/predict/future \
  -H "Content-Type: application/json" \
  -d '{}'
```

## License

This project is open source and available under the MIT License.
