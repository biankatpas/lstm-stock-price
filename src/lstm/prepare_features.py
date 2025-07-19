import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class PrepareFeatures:
    """
    Handles data preprocessing for stock market analysis.
    
    Responsibilities:
    - Data cleaning and validation
    - Feature engineering
    - Technical indicators calculation
    - Data scaling and normalization
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """  
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.dropna()
        
        # Ensure positive values
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Price-based indicators
        df['MA_7'] = df['Close'].rolling(7).mean()
        df['MA_21'] = df['Close'].rolling(21).mean() 
        df['MA_50'] = df['Close'].rolling(50).mean()
        
        # Price range (volatility proxy)
        df['Price_Range'] = df['High'] - df['Low']
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(21).std()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(21).mean()
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        RSI Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Interpretation:
        - RSI > 70: Potentially overbought (sell signal)
        - RSI < 30: Potentially oversold (buy signal)
        - RSI 30-70: Normal trading range
        
        Args:
            prices (pd.Series): Price series
            window (int): Calculation window
            
        Returns:
            pd.Series: RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def scale_features(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Args:
            data (np.ndarray): Feature data
            fit (bool): Whether to fit the scaler
            
        Returns:
            np.ndarray: Scaled data
        """
        if fit:
            scaled_data = self.scaler.fit_transform(data)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transform")
            scaled_data = self.scaler.transform(data)
        
        return scaled_data
    
    def inverse_scale_predictions(self, predictions: np.ndarray, target_index: int = 0) -> np.ndarray:
        """
        Args:
            predictions (np.ndarray): Scaled predictions
            target_index (int): Index of target feature
            
        Returns:
            np.ndarray: Original scale predictions
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        
        # Create dummy array for inverse transform
        dummy_array = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy_array[:, target_index] = predictions.flatten()
        
        # Inverse transform and return only target column
        return self.scaler.inverse_transform(dummy_array)[:, target_index]
    
    def prepare_features(self, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Args:
            df (pd.DataFrame): Stock data with indicators
            features (list): List of feature column names
            
        Returns:
            pd.DataFrame: Prepared feature data
        """
        # Clean data
        df = self.clean_data(df)
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Select features
        available_features = [f for f in features if f in df.columns]
        if len(available_features) != len(features):
            missing = set(features) - set(available_features)
            print(f"Warning: Missing features {missing}")
        
        return df[available_features]
