import os

import pandas as pd
import yfinance as yf

from utils.logger_config import get_logger

# Get logger
logger = get_logger(__name__)


class YahooFinanceDownloader:

    def __init__(self):
        pass

    def run(
        self,
        symbol,
        start_date="2018-01-01",
        end_date="2025-07-01",
        interval="1d",
        output_dir="data",
    ):
        """
        Downloads stock data for a specific date range and saves to CSV

        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'PETR4.SA')
            start_date (str): Start date (format 'YYYY-MM-DD')
            end_date (str): End date (format 'YYYY-MM-DD')
            interval (str): Data interval ('1d', '1wk', '1mo')
            output_dir (str): Directory to save CSV files

        Returns:
            pandas.DataFrame: DataFrame with stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data.empty:
                raise ValueError(
                    f"No data found for symbol {symbol} in the specified period"
                )

            os.makedirs(output_dir, exist_ok=True)

            filename = f"{symbol}_{start_date}_to_{end_date}.csv"
            filepath = os.path.join(output_dir, filename)
            data.to_csv(filepath)

            logger.info(f"Data for {symbol} saved to {filepath}")
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

            return data

        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            return None
