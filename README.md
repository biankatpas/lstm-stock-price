# LSTM Stock Price

A Flask-based API for stock price prediction using LSTM neural networks with data from Yahoo Finance.

## Project Structure

```
lstm-stock-price/
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── requirements.txt         # Python dependencies
├── data/                    # CSV data storage
├── docs/                    # Documentation files
└── src/                     # Source code
    ├── app.py               # Flask application entry point
    ├── models/              # ML models
    │   ├── __init__.py
    │   └── lstm_model.py    # LSTM model implementation
    ├── routes/              # API routes
    │   └── __init__.py
    └── scrape/              # Data collection
        ├── __init__.py
        └── yahoo_finance.py # Yahoo Finance data downloader
```

## Installation & Setup

### Option 1: Local Development with Virtual Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/biankatpas/lstm-stock-price.git
   cd lstm-stock-price
   ```

2. **Create and activate Python virtual environment**
      
   **On Windows:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```
   
   **On macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**
   ```bash
   python src/app.py
   ```

5. **Test the API**
   ```bash
   curl http://localhost:5000/test
   ```

### Option 2: Docker Deployment

1. **Clone the repository**
   ```bash
   git clone https://github.com/biankatpas/lstm-stock-price.git
   cd lstm-stock-price
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker compose up --build
   ```

3. **Test the containerized API**
   ```bash
   curl http://localhost:5000/test
   ```

## Usage

### Download Stock Data

Run the Yahoo Finance data downloader:

```bash
python src/scrape/yahoo_finance.py
```

This will download stock data from 2018-01-01 to 2025-07-01 and save it as CSV in the `data/` directory.

### API Endpoints

- `GET /test` - Health check endpoint

## Development

### Adding New Features

1. **Data Sources**: Add new data collectors in `src/scrape/`
2. **Models**: Implement ML models in `src/models/`
3. **API Routes**: Create new endpoints in `src/routes/`

### Environment Variables

Create a `.env` file for configuration.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
