import os

from flask import Blueprint, jsonify, request

from scrape.yahoo_finance import YahooFinanceDownloader

from .mock_payload import get_scrape_mock_payload

# Create blueprint
scrape_bp = Blueprint("scrape", __name__)


@scrape_bp.route("/scrape", methods=["POST"])
def download_stock_data():
    """
    Download stock data from Yahoo Finance

    JSON Parameters:
    - symbol: str (required) - Stock symbol (e.g., 'AAPL', 'GOOGL', 'PETR4.SA')
    - start_date: str (optional, default: '2018-01-01') - Start date (format 'YYYY-MM-DD')
    - end_date: str (optional, default: '2025-07-01') - End date (format 'YYYY-MM-DD')
    - interval: str (optional, default: '1d') - Data interval ('1d', '1wk', '1mo')
    - output_dir: str (optional, default: 'data') - Directory to save CSV files

    Example request:
    {
        "symbol": "AAPL",
        "start_date": "2020-01-01",
        "end_date": "2025-01-01",
        "interval": "1d"
    }
    """

    # Get request data
    data = request.get_json(force=True, silent=True) or {}

    # Use mock payload if no data provided for testing purposes
    if not data:
        data = get_scrape_mock_payload()
        print(f"Using mock payload for testing: {data}")

    # Validate required parameters
    symbol = data.get("symbol")
    if not symbol:
        return (
            jsonify(
                {
                    "error": "Missing required parameter",
                    "message": "Parameter 'symbol' is required",
                }
            ),
            400,
        )

    # Get optional parameters with defaults
    start_date = data.get("start_date", "2018-01-01")
    end_date = data.get("end_date", "2025-07-01")
    interval = data.get("interval", "1d")
    output_dir = data.get("output_dir", "data")

    # Validate interval
    valid_intervals = ["1d", "1wk", "1mo", "3mo"]
    if interval not in valid_intervals:
        return (
            jsonify(
                {
                    "error": "Invalid interval",
                    "message": f"Interval must be one of: {', '.join(valid_intervals)}",
                }
            ),
            400,
        )

    try:
        # Initialize downloader
        downloader = YahooFinanceDownloader()

        # Download data
        df = downloader.run(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            output_dir=output_dir,
        )

        if df is None:
            return (
                jsonify(
                    {
                        "error": "Download failed",
                        "message": f"Failed to download data for symbol {symbol}",
                    }
                ),
                500,
            )

        # Generate filename for reference
        filename = f"{symbol}_{start_date}_to_{end_date}.csv"
        filepath = os.path.join(output_dir, filename)

        return (
            jsonify(
                {
                    "message": "Data downloaded successfully",
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                    "interval": interval,
                    "filename": filename,
                    "filepath": filepath,
                    "data_shape": {"rows": len(df), "columns": len(df.columns)},
                    "date_range": {
                        "first_date": str(df.index[0].date()),
                        "last_date": str(df.index[-1].date()),
                    },
                    "columns": list(df.columns),
                }
            ),
            200,
        )

    except ValueError as e:
        return jsonify({"error": "Data validation error", "message": str(e)}), 400

    except Exception as e:
        return (
            jsonify(
                {
                    "error": "Unexpected error",
                    "message": f"An error occurred while downloading data: {str(e)}",
                }
            ),
            500,
        )
