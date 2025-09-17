"""
Anomaly Detection Models

Implements various anomaly detection algorithms for sensor time-series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM models will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAnomalyDetector:
    """Base class for anomaly detection models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'BaseAnomalyDetector':
        """Fit the anomaly detection model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        raise NotImplementedError
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for samples."""
        raise NotImplementedError
    
    def save_model(self, filepath: str):
        """Save the model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        logger.info(f"Model loaded from {filepath}")


class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        super().__init__("IsolationForest")
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model."""
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower scores indicate more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)


class OneClassSVMDetector(BaseAnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: str = 'scale'):
        super().__init__("OneClassSVM")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the One-Class SVM model."""
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (lower scores indicate more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """LSTM Autoencoder for anomaly detection."""
    
    def __init__(self, sequence_length: int = 60, encoding_dim: int = 32, 
                 learning_rate: float = 0.001, epochs: int = 100):
        super().__init__("LSTMAutoencoder")
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = None
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM autoencoder model."""
        input_dim = 1  # Single feature for now
        
        # Encoder
        encoder_input = Input(shape=(self.sequence_length, input_dim))
        encoder_lstm1 = LSTM(64, return_sequences=True)(encoder_input)
        encoder_lstm2 = LSTM(32, return_sequences=False)(encoder_lstm1)
        encoder_output = Dense(self.encoding_dim)(encoder_lstm2)
        
        # Decoder
        decoder_input = Input(shape=(self.encoding_dim,))
        decoder_dense = Dense(32)(decoder_input)
        decoder_lstm1 = LSTM(32, return_sequences=True)(decoder_dense)
        decoder_lstm2 = LSTM(64, return_sequences=True)(decoder_lstm1)
        decoder_output = Dense(input_dim)(decoder_lstm2)
        
        # Autoencoder
        self.encoder = Model(encoder_input, encoder_output)
        self.decoder = Model(decoder_input, decoder_output)
        
        autoencoder_input = Input(shape=(self.sequence_length, input_dim))
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        
        self.model = Model(autoencoder_input, decoded)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                          loss='mse')
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X: np.ndarray) -> 'LSTMAutoencoderDetector':
        """Fit the LSTM autoencoder model."""
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences = self._create_sequences(X_scaled)
        X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        # Train the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        self.model.fit(
            X_sequences, X_sequences,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate reconstruction threshold
        reconstructions = self.model.predict(X_sequences)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile as threshold
        
        self.is_fitted = True
        logger.info(f"{self.name} training completed. Threshold: {self.threshold:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        X_sequences = self._create_sequences(X_scaled)
        X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        reconstructions = self.model.predict(X_sequences)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=1)
        
        # Convert to binary predictions
        predictions = np.where(mse > self.threshold, -1, 1)
        
        # Pad with normal predictions for the beginning
        padded_predictions = np.ones(len(X))
        padded_predictions[self.sequence_length-1:] = predictions
        
        return padded_predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        X_sequences = self._create_sequences(X_scaled)
        X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        reconstructions = self.model.predict(X_sequences)
        mse = np.mean(np.power(X_sequences - reconstructions, 2), axis=1)
        
        # Pad with low scores for the beginning
        padded_scores = np.full(len(X), 0.0)
        padded_scores[self.sequence_length-1:] = mse
        
        return padded_scores


class AnomalyDetectionEnsemble:
    """Ensemble of anomaly detection models."""
    
    def __init__(self, models: List[BaseAnomalyDetector], voting_method: str = 'average'):
        """
        Initialize the ensemble.
        
        Args:
            models: List of anomaly detection models
            voting_method: Method for combining predictions ('average', 'majority', 'weighted')
        """
        self.models = models
        self.voting_method = voting_method
        self.weights = None
    
    def fit(self, X: np.ndarray) -> 'AnomalyDetectionEnsemble':
        """Fit all models in the ensemble."""
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        for model in self.models:
            model.fit(X)
        
        # Calculate weights based on individual model performance
        self._calculate_weights(X)
        
        logger.info("Ensemble training completed")
        return self
    
    def _calculate_weights(self, X: np.ndarray):
        """Calculate weights for ensemble voting."""
        if self.voting_method == 'weighted':
            # Use reconstruction error as weights (for autoencoders)
            weights = []
            for model in self.models:
                if hasattr(model, 'score_samples'):
                    scores = model.score_samples(X)
                    # Lower scores (more anomalous) should have higher weight
                    weight = 1.0 / (np.mean(scores) + 1e-8)
                    weights.append(weight)
                else:
                    weights.append(1.0)
            
            # Normalize weights
            self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using ensemble voting."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            # Convert to 0/1 for voting
            pred_binary = (pred == 1).astype(int)
            predictions.append(pred_binary)
        
        predictions = np.array(predictions)
        
        if self.voting_method == 'majority':
            # Majority voting
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
        elif self.voting_method == 'average':
            # Average voting
            ensemble_pred = np.mean(predictions, axis=0) > 0.5
        elif self.voting_method == 'weighted':
            # Weighted voting
            weighted_pred = np.average(predictions, axis=0, weights=self.weights)
            ensemble_pred = weighted_pred > 0.5
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        # Convert back to -1/1 format
        return np.where(ensemble_pred, 1, -1)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble anomaly scores."""
        scores = []
        
        for model in self.models:
            if hasattr(model, 'score_samples'):
                score = model.score_samples(X)
                scores.append(score)
        
        if not scores:
            return np.zeros(len(X))
        
        scores = np.array(scores)
        
        if self.voting_method == 'weighted' and self.weights is not None:
            # Weighted average
            return np.average(scores, axis=0, weights=self.weights)
        else:
            # Simple average
            return np.mean(scores, axis=0)


class AnomalyDetectionPipeline:
    """Complete pipeline for anomaly detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the anomaly detection pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.ensemble = None
        self.results = {}
    
    def create_models(self) -> Dict[str, BaseAnomalyDetector]:
        """Create anomaly detection models."""
        models = {}
        
        # Isolation Forest
        models['isolation_forest'] = IsolationForestDetector(
            contamination=self.config.get('contamination', 0.1)
        )
        
        # One-Class SVM
        models['one_class_svm'] = OneClassSVMDetector(
            nu=self.config.get('nu', 0.1)
        )
        
        # LSTM Autoencoder (if TensorFlow is available)
        if TF_AVAILABLE:
            models['lstm_autoencoder'] = LSTMAutoencoderDetector(
                sequence_length=self.config.get('sequence_length', 60),
                encoding_dim=self.config.get('encoding_dim', 32)
            )
        
        self.models = models
        return models
    
    def train_models(self, X: np.ndarray, y: np.ndarray = None):
        """
        Train all anomaly detection models.
        
        Args:
            X: Training features
            y: Training labels (optional, for evaluation)
        """
        logger.info("Training anomaly detection models")
        
        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            model.fit(X)
        
        # Create and train ensemble
        model_list = list(self.models.values())
        self.ensemble = AnomalyDetectionEnsemble(
            models=model_list,
            voting_method=self.config.get('voting_method', 'average')
        )
        self.ensemble.fit(X)
        
        # Evaluate models if labels are provided
        if y is not None:
            self._evaluate_models(X, y)
    
    def _evaluate_models(self, X: np.ndarray, y: np.ndarray):
        """Evaluate model performance."""
        logger.info("Evaluating anomaly detection models")
        
        for name, model in self.models.items():
            predictions = model.predict(X)
            # Convert to binary (0 for normal, 1 for anomaly)
            pred_binary = (predictions == -1).astype(int)
            
            # Calculate metrics
            accuracy = np.mean(pred_binary == y)
            precision = np.sum((pred_binary == 1) & (y == 1)) / (np.sum(pred_binary == 1) + 1e-8)
            recall = np.sum((pred_binary == 1) & (y == 1)) / (np.sum(y == 1) + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    def predict_anomalies(self, X: np.ndarray, use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Predict anomalies using the best model or ensemble.
        
        Args:
            X: Features to predict
            use_ensemble: Whether to use ensemble or individual models
            
        Returns:
            Dictionary with predictions and scores
        """
        if use_ensemble and self.ensemble:
            predictions = self.ensemble.predict(X)
            scores = self.ensemble.score_samples(X)
            model_name = 'ensemble'
        else:
            # Use the best individual model
            best_model_name = max(self.results.keys(), 
                                key=lambda k: self.results[k]['f1_score'])
            model = self.models[best_model_name]
            predictions = model.predict(X)
            scores = model.score_samples(X)
            model_name = best_model_name
        
        return {
            'predictions': predictions,
            'scores': scores,
            'model_used': model_name,
            'anomaly_indices': np.where(predictions == -1)[0]
        }
    
    def save_pipeline(self, filepath: str):
        """Save the entire pipeline."""
        pipeline_data = {
            'config': self.config,
            'results': self.results,
            'ensemble_weights': self.ensemble.weights if self.ensemble else None
        }
        
        # Save individual models
        for name, model in self.models.items():
            model.save_model(f"{filepath}_{name}.joblib")
        
        # Save pipeline metadata
        joblib.dump(pipeline_data, f"{filepath}_pipeline.joblib")
        logger.info(f"Pipeline saved to {filepath}")


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Normal data
    X_normal = np.random.normal(0, 1, (n_samples, n_features))
    
    # Anomalous data
    X_anomaly = np.random.normal(3, 0.5, (50, n_features))
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_samples), np.ones(50)])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Initialize pipeline
    config = {
        'contamination': 0.1,
        'nu': 0.1,
        'voting_method': 'weighted'
    }
    
    pipeline = AnomalyDetectionPipeline(config)
    pipeline.create_models()
    
    # Train models
    pipeline.train_models(X, y)
    
    # Make predictions
    results = pipeline.predict_anomalies(X)
    
    print(f"Detected {len(results['anomaly_indices'])} anomalies")
    print(f"Model used: {results['model_used']}")
    
    # Save pipeline
    pipeline.save_pipeline("models/anomaly_detection")
