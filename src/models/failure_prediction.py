"""
Failure Prediction Models

Implements various models for predicting equipment failures and maintenance needs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Traditional ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Time series imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Time series forecasting will be limited.")

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM models will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseFailurePredictor:
    """Base class for failure prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.target_columns = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseFailurePredictor':
        """Fit the failure prediction model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict failures."""
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict failure probabilities."""
        raise NotImplementedError
    
    def save_model(self, filepath: str):
        """Save the model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model from file."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        logger.info(f"Model loaded from {filepath}")


class RandomForestFailurePredictor(BaseFailurePredictor):
    """Random Forest for failure prediction."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 random_state: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestFailurePredictor':
        """Fit the Random Forest model."""
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict failures."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict failure probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.feature_importances_


class GradientBoostingFailurePredictor(BaseFailurePredictor):
    """Gradient Boosting for failure prediction."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, random_state: int = 42):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingFailurePredictor':
        """Fit the Gradient Boosting model."""
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict failures."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict failure probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.feature_importances_


class LSTMFailurePredictor(BaseFailurePredictor):
    """LSTM for failure prediction."""
    
    def __init__(self, sequence_length: int = 60, lstm_units: int = 50,
                 learning_rate: float = 0.001, epochs: int = 100):
        super().__init__("LSTM")
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM model."""
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X_sequences = []
        y_sequences = []
        
        for i in range(len(X) - self.sequence_length + 1):
            X_sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                y_sequences.append(y[i + self.sequence_length - 1])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences) if y is not None else None
        
        return X_sequences, y_sequences
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LSTMFailurePredictor':
        """Fit the LSTM model."""
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X_scaled, y)
        X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_sequences, test_size=0.2, random_state=42
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        self.is_fitted = True
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict failures."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        X_sequences, _ = self._create_sequences(X_scaled)
        X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        predictions = self.model.predict(X_sequences)
        
        # Pad with predictions for the beginning
        padded_predictions = np.zeros(len(X))
        padded_predictions[self.sequence_length-1:] = (predictions.flatten() > 0.5).astype(int)
        
        return padded_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict failure probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).flatten()
        X_sequences, _ = self._create_sequences(X_scaled)
        X_sequences = X_sequences.reshape(X_sequences.shape[0], X_sequences.shape[1], 1)
        
        probabilities = self.model.predict(X_sequences)
        
        # Pad with probabilities for the beginning
        padded_probs = np.zeros(len(X))
        padded_probs[self.sequence_length-1:] = probabilities.flatten()
        
        return padded_probs


class ProphetFailurePredictor:
    """Prophet for time series failure prediction."""
    
    def __init__(self, name: str = "Prophet"):
        self.name = name
        self.model = None
        self.is_fitted = False
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for time series forecasting")
    
    def fit(self, df: pd.DataFrame, target_column: str = 'value') -> 'ProphetFailurePredictor':
        """
        Fit the Prophet model.
        
        Args:
            df: DataFrame with 'ds' (datetime) and target column
            target_column: Name of the target column
        """
        logger.info(f"Training {self.name} with {len(df)} samples")
        
        # Prepare data for Prophet
        prophet_df = df[['ds', target_column]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Initialize and fit Prophet
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Make predictions for future periods.
        
        Args:
            periods: Number of periods to predict
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return forecast
    
    def predict_failure_probability(self, df: pd.DataFrame, 
                                  failure_threshold: float = 0.8) -> pd.DataFrame:
        """
        Predict failure probability based on forecasted values.
        
        Args:
            df: Historical data
            failure_threshold: Threshold for considering a failure
            
        Returns:
            DataFrame with failure probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get forecast
        forecast = self.predict(periods=0)
        
        # Calculate failure probability based on trend and seasonality
        failure_prob = np.where(
            forecast['yhat'] > failure_threshold,
            (forecast['yhat'] - failure_threshold) / (1 - failure_threshold),
            0
        )
        
        forecast['failure_probability'] = np.clip(failure_prob, 0, 1)
        
        return forecast


class FailurePredictionEnsemble:
    """Ensemble of failure prediction models."""
    
    def __init__(self, models: List[BaseFailurePredictor], voting_method: str = 'soft'):
        """
        Initialize the ensemble.
        
        Args:
            models: List of failure prediction models
            voting_method: Method for combining predictions ('soft', 'hard', 'weighted')
        """
        self.models = models
        self.voting_method = voting_method
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'FailurePredictionEnsemble':
        """Fit all models in the ensemble."""
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        for model in self.models:
            model.fit(X, y)
        
        # Calculate weights based on individual model performance
        self._calculate_weights(X, y)
        
        logger.info("Ensemble training completed")
        return self
    
    def _calculate_weights(self, X: np.ndarray, y: np.ndarray):
        """Calculate weights for ensemble voting."""
        if self.voting_method == 'weighted':
            # Use cross-validation scores as weights
            weights = []
            for model in self.models:
                scores = cross_val_score(model.model, X, y, cv=5, scoring='roc_auc')
                weight = np.mean(scores)
                weights.append(weight)
            
            # Normalize weights
            self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict failures using ensemble voting."""
        if self.voting_method == 'soft':
            # Soft voting using probabilities
            probabilities = []
            for model in self.models:
                proba = model.predict_proba(X)
                probabilities.append(proba[:, 1])  # Probability of positive class
            
            probabilities = np.array(probabilities)
            
            if self.weights is not None:
                ensemble_proba = np.average(probabilities, axis=0, weights=self.weights)
            else:
                ensemble_proba = np.mean(probabilities, axis=0)
            
            return (ensemble_proba > 0.5).astype(int)
        
        elif self.voting_method == 'hard':
            # Hard voting using predictions
            predictions = []
            for model in self.models:
                pred = model.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            if self.weights is not None:
                ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
            else:
                ensemble_pred = np.mean(predictions, axis=0)
            
            return (ensemble_pred > 0.5).astype(int)
        
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict failure probabilities using ensemble."""
        probabilities = []
        for model in self.models:
            proba = model.predict_proba(X)
            probabilities.append(proba)
        
        probabilities = np.array(probabilities)
        
        if self.weights is not None:
            ensemble_proba = np.average(probabilities, axis=0, weights=self.weights)
        else:
            ensemble_proba = np.mean(probabilities, axis=0)
        
        return ensemble_proba


class FailurePredictionPipeline:
    """Complete pipeline for failure prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the failure prediction pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.ensemble = None
        self.results = {}
        self.feature_columns = []
        self.target_columns = []
    
    def create_models(self) -> Dict[str, BaseFailurePredictor]:
        """Create failure prediction models."""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestFailurePredictor(
            n_estimators=self.config.get('rf_n_estimators', 100),
            max_depth=self.config.get('rf_max_depth', 10)
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingFailurePredictor(
            n_estimators=self.config.get('gb_n_estimators', 100),
            learning_rate=self.config.get('gb_learning_rate', 0.1)
        )
        
        # LSTM (if TensorFlow is available)
        if TF_AVAILABLE:
            models['lstm'] = LSTMFailurePredictor(
                sequence_length=self.config.get('lstm_sequence_length', 60),
                lstm_units=self.config.get('lstm_units', 50)
            )
        
        self.models = models
        return models
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train all failure prediction models.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Training failure prediction models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
        
        # Create and train ensemble
        model_list = list(self.models.values())
        self.ensemble = FailurePredictionEnsemble(
            models=model_list,
            voting_method=self.config.get('voting_method', 'soft')
        )
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate models
        self._evaluate_models(X_test, y_test)
    
    def _evaluate_models(self, X: np.ndarray, y: np.ndarray):
        """Evaluate model performance."""
        logger.info("Evaluating failure prediction models")
        
        for name, model in self.models.items():
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Calculate metrics
            accuracy = np.mean(predictions == y)
            auc = roc_auc_score(y, probabilities[:, 1])
            
            # Classification report
            report = classification_report(y, predictions, output_dict=True)
            
            self.results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1_score': report['1']['f1-score']
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
        
        # Evaluate ensemble
        if self.ensemble:
            ensemble_pred = self.ensemble.predict(X)
            ensemble_proba = self.ensemble.predict_proba(X)
            
            accuracy = np.mean(ensemble_pred == y)
            auc = roc_auc_score(y, ensemble_proba[:, 1])
            
            self.results['ensemble'] = {
                'accuracy': accuracy,
                'auc': auc
            }
            
            logger.info(f"Ensemble - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
    
    def predict_failures(self, X: np.ndarray, use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Predict failures using the best model or ensemble.
        
        Args:
            X: Features to predict
            use_ensemble: Whether to use ensemble or individual models
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if use_ensemble and self.ensemble:
            predictions = self.ensemble.predict(X)
            probabilities = self.ensemble.predict_proba(X)
            model_name = 'ensemble'
        else:
            # Use the best individual model
            best_model_name = max(self.results.keys(), 
                                key=lambda k: self.results[k]['auc'])
            model = self.models[best_model_name]
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            model_name = best_model_name
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_used': model_name,
            'failure_indices': np.where(predictions == 1)[0]
        }
    
    def get_maintenance_recommendations(self, X: np.ndarray, 
                                      urgency_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get maintenance recommendations based on failure predictions.
        
        Args:
            X: Features to predict
            urgency_threshold: Threshold for urgent maintenance
            
        Returns:
            List of maintenance recommendations
        """
        results = self.predict_failures(X)
        recommendations = []
        
        for i, (pred, prob) in enumerate(zip(results['predictions'], results['probabilities'][:, 1])):
            if pred == 1:  # Failure predicted
                urgency = 'high' if prob > urgency_threshold else 'medium'
                recommendations.append({
                    'index': i,
                    'failure_probability': prob,
                    'urgency': urgency,
                    'recommendation': f"Schedule maintenance - {urgency} priority",
                    'estimated_failure_time': self._estimate_failure_time(prob)
                })
        
        return recommendations
    
    def _estimate_failure_time(self, probability: float) -> str:
        """Estimate time to failure based on probability."""
        if probability > 0.9:
            return "Within 24 hours"
        elif probability > 0.7:
            return "Within 1 week"
        elif probability > 0.5:
            return "Within 1 month"
        else:
            return "More than 1 month"
    
    def save_pipeline(self, filepath: str):
        """Save the entire pipeline."""
        pipeline_data = {
            'config': self.config,
            'results': self.results,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
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
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target (failure) with some correlation to features
    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.5 > 1).astype(int)
    
    # Initialize pipeline
    config = {
        'rf_n_estimators': 100,
        'gb_n_estimators': 100,
        'voting_method': 'soft'
    }
    
    pipeline = FailurePredictionPipeline(config)
    pipeline.create_models()
    
    # Train models
    pipeline.train_models(X, y)
    
    # Make predictions
    results = pipeline.predict_failures(X)
    
    print(f"Predicted {len(results['failure_indices'])} failures")
    print(f"Model used: {results['model_used']}")
    
    # Get maintenance recommendations
    recommendations = pipeline.get_maintenance_recommendations(X)
    print(f"Generated {len(recommendations)} maintenance recommendations")
    
    # Save pipeline
    pipeline.save_pipeline("models/failure_prediction")
