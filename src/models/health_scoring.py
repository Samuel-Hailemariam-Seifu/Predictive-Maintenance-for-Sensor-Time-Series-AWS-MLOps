"""
Health Scoring Models

Implements various models for calculating equipment health scores and maintenance urgency.
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network models will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseHealthScorer:
    """Base class for health scoring models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_columns = []
        self.health_range = (0, 1)  # Health score range
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseHealthScorer':
        """Fit the health scoring model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict health scores."""
        raise NotImplementedError
    
    def get_health_status(self, health_score: float) -> str:
        """Convert health score to status string."""
        if health_score >= 0.8:
            return 'excellent'
        elif health_score >= 0.6:
            return 'good'
        elif health_score >= 0.4:
            return 'fair'
        elif health_score >= 0.2:
            return 'poor'
        else:
            return 'critical'
    
    def get_maintenance_urgency(self, health_score: float) -> str:
        """Get maintenance urgency based on health score."""
        if health_score <= 0.2:
            return 'critical'
        elif health_score <= 0.4:
            return 'high'
        elif health_score <= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def save_model(self, filepath: str):
        """Save the model to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'health_range': self.health_range
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
        self.health_range = model_data['health_range']
        logger.info(f"Model loaded from {filepath}")


class RandomForestHealthScorer(BaseHealthScorer):
    """Random Forest for health scoring."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 random_state: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestHealthScorer':
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
        """Predict health scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Ensure predictions are in valid range
        return np.clip(predictions, self.health_range[0], self.health_range[1])
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.feature_importances_


class GradientBoostingHealthScorer(BaseHealthScorer):
    """Gradient Boosting for health scoring."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 6, random_state: int = 42):
        super().__init__("GradientBoosting")
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingHealthScorer':
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
        """Predict health scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Ensure predictions are in valid range
        return np.clip(predictions, self.health_range[0], self.health_range[1])
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.model.feature_importances_


class NeuralNetworkHealthScorer(BaseHealthScorer):
    """Neural Network for health scoring."""
    
    def __init__(self, hidden_layers: List[int] = [64, 32], 
                 learning_rate: float = 0.001, epochs: int = 100,
                 dropout_rate: float = 0.2):
        super().__init__("NeuralNetwork")
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models")
        
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self._build_model()
    
    def _build_model(self):
        """Build the neural network model."""
        self.model = Sequential()
        
        # Input layer
        self.model.add(Dense(self.hidden_layers[0], activation='relu', input_shape=(None,)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralNetworkHealthScorer':
        """Fit the neural network model."""
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
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
        """Predict health scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled).flatten()
        
        # Ensure predictions are in valid range
        return np.clip(predictions, self.health_range[0], self.health_range[1])


class ClusteringHealthScorer:
    """Health scoring based on clustering of normal operating conditions."""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.name = "Clustering"
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ClusteringHealthScorer':
        """
        Fit the clustering model.
        
        Args:
            X: Training features
            y: Training labels (not used for clustering)
        """
        logger.info(f"Training {self.name} with {len(X)} samples")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit clustering
        self.kmeans.fit(X_scaled)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        self.is_fitted = True
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict health scores based on distance to cluster centers."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Calculate distances to all cluster centers
        distances = np.linalg.norm(X_scaled[:, np.newaxis] - self.cluster_centers, axis=2)
        
        # Health score is inverse of minimum distance (normalized)
        min_distances = np.min(distances, axis=1)
        max_distance = np.max(min_distances)
        
        # Normalize to 0-1 range (higher is better)
        health_scores = 1 - (min_distances / max_distance)
        
        return np.clip(health_scores, 0, 1)
    
    def get_cluster_labels(self, X: np.ndarray) -> np.ndarray:
        """Get cluster labels for data points."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)


class MultiSensorHealthScorer:
    """Health scoring that combines multiple sensor readings."""
    
    def __init__(self, sensor_weights: Dict[str, float] = None):
        """
        Initialize the multi-sensor health scorer.
        
        Args:
            sensor_weights: Weights for different sensor types
        """
        self.name = "MultiSensor"
        self.sensor_weights = sensor_weights or {
            'vibration': 0.4,
            'temperature': 0.3,
            'pressure': 0.2,
            'rpm': 0.1
        }
        self.normal_ranges = {
            'vibration': (0.0, 1.0),
            'temperature': (20, 80),
            'pressure': (0, 100),
            'rpm': (1500, 2000)
        }
        self.is_fitted = True  # No training needed for rule-based approach
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'MultiSensorHealthScorer':
        """Fit the multi-sensor health scorer (no training needed)."""
        logger.info(f"Multi-sensor health scorer initialized")
        return self
    
    def predict(self, X: np.ndarray, sensor_types: List[str] = None) -> np.ndarray:
        """
        Predict health scores based on multiple sensor readings.
        
        Args:
            X: Sensor readings (n_samples, n_sensors)
            sensor_types: List of sensor types for each column
            
        Returns:
            Health scores
        """
        if sensor_types is None:
            sensor_types = ['vibration'] * X.shape[1]
        
        health_scores = []
        
        for i in range(X.shape[0]):
            sensor_scores = []
            total_weight = 0
            
            for j, sensor_type in enumerate(sensor_types):
                if j < X.shape[1] and sensor_type in self.sensor_weights:
                    value = X[i, j]
                    weight = self.sensor_weights[sensor_type]
                    
                    # Calculate health score for this sensor
                    if sensor_type in self.normal_ranges:
                        normal_min, normal_max = self.normal_ranges[sensor_type]
                        # Normalize value to 0-1 scale
                        normalized_value = np.clip((value - normal_min) / (normal_max - normal_min), 0, 1)
                        # Health score (1 - normalized value, so higher is better)
                        sensor_score = 1 - normalized_value
                    else:
                        # Default health score
                        sensor_score = 0.5
                    
                    sensor_scores.append(sensor_score * weight)
                    total_weight += weight
            
            # Calculate weighted average health score
            if total_weight > 0:
                overall_health = sum(sensor_scores) / total_weight
            else:
                overall_health = 0.5  # Default neutral score
            
            health_scores.append(overall_health)
        
        return np.array(health_scores)


class HealthScoringEnsemble:
    """Ensemble of health scoring models."""
    
    def __init__(self, models: List[BaseHealthScorer], voting_method: str = 'average'):
        """
        Initialize the ensemble.
        
        Args:
            models: List of health scoring models
            voting_method: Method for combining predictions ('average', 'weighted', 'median')
        """
        self.models = models
        self.voting_method = voting_method
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HealthScoringEnsemble':
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
            # Use R² scores as weights
            weights = []
            for model in self.models:
                predictions = model.predict(X)
                r2 = r2_score(y, predictions)
                weights.append(max(r2, 0))  # Ensure non-negative weights
            
            # Normalize weights
            if sum(weights) > 0:
                self.weights = np.array(weights) / sum(weights)
            else:
                self.weights = np.ones(len(self.models)) / len(self.models)
        else:
            self.weights = np.ones(len(self.models)) / len(self.models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict health scores using ensemble voting."""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.voting_method == 'average':
            # Simple average
            ensemble_pred = np.mean(predictions, axis=0)
        elif self.voting_method == 'weighted':
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        elif self.voting_method == 'median':
            # Median
            ensemble_pred = np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown voting method: {self.voting_method}")
        
        return np.clip(ensemble_pred, 0, 1)


class HealthScoringPipeline:
    """Complete pipeline for health scoring."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the health scoring pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.ensemble = None
        self.results = {}
        self.feature_columns = []
    
    def create_models(self) -> Dict[str, BaseHealthScorer]:
        """Create health scoring models."""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestHealthScorer(
            n_estimators=self.config.get('rf_n_estimators', 100),
            max_depth=self.config.get('rf_max_depth', 10)
        )
        
        # Gradient Boosting
        models['gradient_boosting'] = GradientBoostingHealthScorer(
            n_estimators=self.config.get('gb_n_estimators', 100),
            learning_rate=self.config.get('gb_learning_rate', 0.1)
        )
        
        # Neural Network (if TensorFlow is available)
        if TF_AVAILABLE:
            models['neural_network'] = NeuralNetworkHealthScorer(
                hidden_layers=self.config.get('nn_hidden_layers', [64, 32]),
                learning_rate=self.config.get('nn_learning_rate', 0.001)
            )
        
        # Clustering
        models['clustering'] = ClusteringHealthScorer(
            n_clusters=self.config.get('n_clusters', 5)
        )
        
        # Multi-sensor
        models['multi_sensor'] = MultiSensorHealthScorer(
            sensor_weights=self.config.get('sensor_weights')
        )
        
        self.models = models
        return models
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """
        Train all health scoring models.
        
        Args:
            X: Training features
            y: Training health scores
        """
        logger.info("Training health scoring models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            model.fit(X_train, y_train)
        
        # Create and train ensemble
        model_list = [model for model in self.models.values() 
                     if hasattr(model, 'predict') and not isinstance(model, MultiSensorHealthScorer)]
        
        if model_list:
            self.ensemble = HealthScoringEnsemble(
                models=model_list,
                voting_method=self.config.get('voting_method', 'average')
            )
            self.ensemble.fit(X_train, y_train)
        
        # Evaluate models
        self._evaluate_models(X_test, y_test)
    
    def _evaluate_models(self, X: np.ndarray, y: np.ndarray):
        """Evaluate model performance."""
        logger.info("Evaluating health scoring models")
        
        for name, model in self.models.items():
            predictions = model.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            self.results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"{name} - RMSE: {rmse:.3f}, R²: {r2:.3f}")
        
        # Evaluate ensemble
        if self.ensemble:
            ensemble_pred = self.ensemble.predict(X)
            
            mse = mean_squared_error(y, ensemble_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, ensemble_pred)
            r2 = r2_score(y, ensemble_pred)
            
            self.results['ensemble'] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"Ensemble - RMSE: {rmse:.3f}, R²: {r2:.3f}")
    
    def predict_health_scores(self, X: np.ndarray, use_ensemble: bool = True) -> Dict[str, Any]:
        """
        Predict health scores using the best model or ensemble.
        
        Args:
            X: Features to predict
            use_ensemble: Whether to use ensemble or individual models
            
        Returns:
            Dictionary with predictions and additional information
        """
        if use_ensemble and self.ensemble:
            predictions = self.ensemble.predict(X)
            model_name = 'ensemble'
        else:
            # Use the best individual model
            best_model_name = max(self.results.keys(), 
                                key=lambda k: self.results[k]['r2'])
            model = self.models[best_model_name]
            predictions = model.predict(X)
            model_name = best_model_name
        
        # Convert to health status and urgency
        health_statuses = [self.models[model_name].get_health_status(score) 
                          for score in predictions]
        urgencies = [self.models[model_name].get_maintenance_urgency(score) 
                    for score in predictions]
        
        return {
            'health_scores': predictions,
            'health_statuses': health_statuses,
            'maintenance_urgencies': urgencies,
            'model_used': model_name,
            'critical_indices': np.where(predictions <= 0.2)[0]
        }
    
    def get_maintenance_schedule(self, X: np.ndarray, 
                               days_ahead: int = 30) -> List[Dict[str, Any]]:
        """
        Get maintenance schedule based on health scores.
        
        Args:
            X: Features to predict
            days_ahead: Number of days to look ahead
            
        Returns:
            List of maintenance recommendations
        """
        results = self.predict_health_scores(X)
        schedule = []
        
        for i, (score, status, urgency) in enumerate(zip(
            results['health_scores'], 
            results['health_statuses'], 
            results['maintenance_urgencies']
        )):
            if urgency in ['critical', 'high']:
                # Immediate maintenance needed
                maintenance_date = datetime.now() + timedelta(days=1)
            elif urgency == 'medium':
                # Schedule within a week
                maintenance_date = datetime.now() + timedelta(days=7)
            else:
                # Schedule within the month
                maintenance_date = datetime.now() + timedelta(days=30)
            
            schedule.append({
                'index': i,
                'health_score': score,
                'status': status,
                'urgency': urgency,
                'recommended_date': maintenance_date,
                'description': f"Health score: {score:.2f} - {status} condition"
            })
        
        # Sort by urgency and date
        urgency_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        schedule.sort(key=lambda x: (urgency_order[x['urgency']], x['recommended_date']))
        
        return schedule
    
    def save_pipeline(self, filepath: str):
        """Save the entire pipeline."""
        pipeline_data = {
            'config': self.config,
            'results': self.results,
            'feature_columns': self.feature_columns,
            'ensemble_weights': self.ensemble.weights if self.ensemble else None
        }
        
        # Save individual models
        for name, model in self.models.items():
            if hasattr(model, 'save_model'):
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
    
    # Generate target health scores (0-1 range)
    y = np.random.beta(2, 2, n_samples)  # Beta distribution for health scores
    
    # Initialize pipeline
    config = {
        'rf_n_estimators': 100,
        'gb_n_estimators': 100,
        'voting_method': 'average'
    }
    
    pipeline = HealthScoringPipeline(config)
    pipeline.create_models()
    
    # Train models
    pipeline.train_models(X, y)
    
    # Make predictions
    results = pipeline.predict_health_scores(X)
    
    print(f"Predicted health scores for {len(results['health_scores'])} samples")
    print(f"Model used: {results['model_used']}")
    print(f"Critical equipment: {len(results['critical_indices'])}")
    
    # Get maintenance schedule
    schedule = pipeline.get_maintenance_schedule(X)
    print(f"Generated {len(schedule)} maintenance recommendations")
    
    # Save pipeline
    pipeline.save_pipeline("models/health_scoring")
