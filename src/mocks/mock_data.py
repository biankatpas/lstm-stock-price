"""
Mock data for testing predict routes
"""


def generate_mock_stock_data(num_points=90, base_price=150.0):
    """
    Generate realistic mock stock data for testing

    Args:
        num_points (int): Number of data points to generate
        base_price (float): Base price to start from

    Returns:
        list: List of lists containing [Open, High, Low, Close, Volume] data
    """
    mock_data = []

    for i in range(num_points):
        # Generate stock data with slight variations
        open_price = base_price + (i * 0.5) + ((-1) ** i * 0.2)
        high_price = open_price + abs(hash(str(i)) % 100) / 100 * 2
        low_price = open_price - abs(hash(str(i + 1)) % 100) / 100 * 2
        close_price = (high_price + low_price) / 2 + ((-1) ** i * 0.1)
        volume = 1000000 + abs(hash(str(i + 2)) % 500000)

        mock_data.append([open_price, high_price, low_price, close_price, volume])

    return mock_data
