# LSTM Stock Price

A Flask-based API for stock price prediction using LSTM neural networks with data from Yahoo Finance.

## Project Structure

```
lstm-stock-price/
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── requirements.txt         # Python dependencies
├── data/                    # CSV data storage
├── models/                  # Trained models
├── docs/                    # Documentation files
└── src/                     # Source code
    ├── app.py               # Flask application entry point
    ├── lstm/                # LSTM model implementation
        ├── __init__.py
        └── lstm_stock_price.py    
    ├── routes/              # API routes
        ├── __init__.py
        ├── health_route.py
        ├── scrape_route.py
        ├── train_route.py
        ├── status_route.py
        └── predict_route.py
    └── scrape/              # Data collection
        ├── __init__.py
        └── yahoo_finance.py # Yahoo Finance data downloader
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
   docker-compose up --build
   ```

3. **Test the containerized API**
   ```bash
   curl http://localhost:5000/health
   ```

## Usage

### API Endpoints

- `GET /health` - Health check endpoint

## License

This project is open source and available under the MIT License.
