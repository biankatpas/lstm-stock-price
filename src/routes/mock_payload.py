"""
Mock payloads for testing API routes
"""


def get_scrape_mock_payload():
    """
    Get mock payload for scrape route

    Returns:
        dict: Mock payload for /scrape endpoint
    """
    return {
        "symbol": "AAPL",
        "start_date": "2018-01-01",
        "end_date": "2025-07-01",
        "interval": "1d",
    }


def get_train_mock_payload():
    """
    Get mock payload for train route

    Returns:
        dict: Mock payload for /train endpoint
    """
    return {
        "epochs": 10,
        "sequence_length": 60,
        "batch_size": 32,
        "learning_rate": 0.001,
        "hidden_sizes": [128, 64],
        "dropout": 0.2,
        "test_size": 0.3,
        "val_size": 0.2,
    }


def get_predict_future_mock_payload():
    """
    Get mock payload for predict future route

    Returns:
        dict: Mock payload for /predict/future endpoint
    """
    return {"days": 7}
