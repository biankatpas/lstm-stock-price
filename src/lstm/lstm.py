import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import os
from typing import Tuple, Optional, List

from .prepare_features import PrepareFeatures
from .prepare_sequence import PrepareSequence
from .evaluate_lstm import EvaluateLSTM


class LSTM:
    
    def __init__(self, sequence_length: int = 30, features: List[str] = None):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.is_trained = False
        
        self.features = PrepareFeatures()
        self.sequence = PrepareSequence(sequence_length)
        self.evaluator = EvaluateLSTM()
    
    def prepare_data(self, data: pd.DataFrame, target_column: str = 'Close', 
                    test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # Prepare features
        feature_data = self.features.prepare_features(data, self.features)
        
        # Scale data
        scaled_data = self.features.scale_features(feature_data.values, fit=True)
        
        # Get target column index
        target_index = self.features.index(target_column)
        
        # Create sequences
        return self.sequence.create_train_test_sequences(
            scaled_data, target_index, test_size
        )
    
    def build_model(self, units: List[int] = None, dropout: float = 0.3, 
                   learning_rate: float = 0.001) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            units (List[int]): Number of units in each LSTM layer
            dropout (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
            
        Returns:
            Sequential: Compiled Keras model
        """
        if units is None:
            units = [100, 50]
        
        self.model = Sequential([
            LSTM(units[0], return_sequences=True, 
                 input_shape=(self.sequence_length, len(self.features))),
            Dropout(dropout),
            
            LSTM(units[1], return_sequences=False),
            Dropout(dropout),
            
            Dense(25, activation='relu'),
            Dropout(dropout/2),
            
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             epochs: int = 100, batch_size: int = 32, verbose: int = 1):
        """
        Train the LSTM model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray, optional): Validation features
            y_val (np.ndarray, optional): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (int): Verbosity level
            
        Returns:
            History: Training history
        """
        if self.model is None:
            self.build_model()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss', 
                patience=15, 
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss', 
                factor=0.5, 
                patience=10, 
                min_lr=1e-6
            )
        ]
        
        # Training
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Input sequences
            
        Returns:
            np.ndarray: Predictions in original scale
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions_scaled = self.model.predict(X)
        target_index = self.features.index('Close')
        return self.features.inverse_scale_predictions(predictions_scaled, target_index)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        target_index = self.features.index('Close')
        y_test_actual = self.features.inverse_scale_predictions(
            y_test.reshape(-1, 1), target_index
        )
        
        return self.evaluator.evaluate_model(y_test_actual, predictions)
    
    def predict_future(self, last_sequence: np.ndarray, days: int = 30) -> np.ndarray:
        """
        Predict future stock prices.
        
        Args:
            last_sequence (np.ndarray): Last sequence from historical data
            days (int): Number of days to predict
            
        Returns:
            np.ndarray: Future predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_sequence = last_sequence.copy()
        target_index = self.features.index('Close')
        
        for _ in range(days):
            # Predict next value
            pred = self.model.predict(
                current_sequence.reshape(1, *current_sequence.shape), 
                verbose=0
            )
            predictions.append(pred[0, 0])
            
            # Update sequence
            new_row = current_sequence[-1].copy()
            new_row[target_index] = pred[0, 0]
            
            # Slide window
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return self.features.inverse_scale_predictions(
            np.array(predictions).reshape(-1, 1), target_index
        )
    
    def save_model(self, model_path: str = 'models/lstm_model.h5', 
                  scaler_path: str = 'models/scaler.pkl'):
        """
        Save trained model and scaler.
        
        Args:
            model_path (str): Path to save model
            scaler_path (str): Path to save scaler
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directories
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        # Save model and scaler
        self.model.save(model_path)
        joblib.dump(self.features.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str = 'models/lstm_model.h5', 
                  scaler_path: str = 'models/scaler.pkl'):
        """
        Load trained model and scaler.
        
        Args:
            model_path (str): Path to model file
            scaler_path (str): Path to scaler file
        """
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = tf.keras.models.load_model(model_path)
            self.features.scaler = joblib.load(scaler_path)
            self.features.is_fitted = True
            self.is_trained = True
            
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
        else:
            raise FileNotFoundError("Model or scaler file not found")


if __name__ == "__main__":
    def train_model():
        print("Starting LSTM Training")
        
        print("Loading data...")
        df = pd.read_csv('../../data/AAPL_2018-01-01_to_2025-07-01.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        features = ['Close', 'Volume', 'MA_7', 'MA_21', 'RSI', 'Volatility']
        
        print("Initializing LSTM model...")
        model = LSTM(sequence_length=60, features=features)
   
        print("Preparing data...")
        X_train, X_test, y_train, y_test = model.prepare_data(df, test_size=0.3)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        print("Training model...")
        history = model.train(X_train, y_train, epochs=50, verbose=1)
        
        print("Evaluating model...")
        metrics = model.evaluate(X_test, y_test)
        model.model_evaluator.print_evaluation_report(metrics)
      
        print("Saving model...")
        model.save_model()
        
        print("Training completed successfully!")
        return model, metrics
    
    # Run training
    trained_model, results = train_model()