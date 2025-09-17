"""
Model Retraining Pipeline

Automated model retraining pipeline for the predictive maintenance system
with data drift detection and model performance monitoring.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import boto3
import mlflow
from mlflow.tracking import MlflowClient

# Import our modules
from src.models.anomaly_detection import AnomalyDetectionPipeline
from src.models.failure_prediction import FailurePredictionPipeline
from src.models.health_scoring import HealthScoringPipeline
from src.aws.s3_integration import S3DataLake
from src.aws.sagemaker_integration import SageMakerModelManager
from mlops.mlflow.mlflow_config import MLflowManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detects data drift in sensor data."""
    
    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference dataset for drift detection
        """
        self.reference_data = reference_data
        self.drift_threshold = 0.1  # 10% drift threshold
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset
            
        Returns:
            Drift detection results
        """
        try:
            drift_results = {
                'has_drift': False,
                'drift_score': 0.0,
                'feature_drifts': {},
                'recommendation': 'No retraining needed'
            }
            
            # Check for drift in each numeric feature
            numeric_features = self.reference_data.select_dtypes(include=[np.number]).columns
            
            for feature in numeric_features:
                if feature in current_data.columns:
                    # Calculate statistical drift
                    ref_mean = self.reference_data[feature].mean()
                    ref_std = self.reference_data[feature].std()
                    curr_mean = current_data[feature].mean()
                    curr_std = current_data[feature].std()
                    
                    # Calculate drift score (normalized difference)
                    mean_drift = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
                    std_drift = abs(curr_std - ref_std) / (ref_std + 1e-8)
                    
                    feature_drift = (mean_drift + std_drift) / 2
                    drift_results['feature_drifts'][feature] = feature_drift
                    
                    if feature_drift > self.drift_threshold:
                        drift_results['has_drift'] = True
                        drift_results['drift_score'] = max(drift_results['drift_score'], feature_drift)
            
            # Overall drift score
            if drift_results['feature_drifts']:
                drift_results['drift_score'] = np.mean(list(drift_results['feature_drifts'].values()))
            
            # Recommendation
            if drift_results['has_drift']:
                if drift_results['drift_score'] > 0.3:
                    drift_results['recommendation'] = 'Immediate retraining required'
                elif drift_results['drift_score'] > 0.2:
                    drift_results['recommendation'] = 'Retraining recommended within 24 hours'
                else:
                    drift_results['recommendation'] = 'Retraining recommended within 1 week'
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {'has_drift': False, 'drift_score': 0.0, 'error': str(e)}
    
    def update_reference_data(self, new_data: pd.DataFrame):
        """Update reference data with new data."""
        try:
            # Combine reference and new data
            combined_data = pd.concat([self.reference_data, new_data], ignore_index=True)
            
            # Keep only recent data (e.g., last 30 days)
            if 'timestamp' in combined_data.columns:
                cutoff_date = datetime.now() - timedelta(days=30)
                combined_data = combined_data[combined_data['timestamp'] >= cutoff_date]
            
            self.reference_data = combined_data
            logger.info("Reference data updated")
            
        except Exception as e:
            logger.error(f"Error updating reference data: {e}")


class ModelPerformanceMonitor:
    """Monitors model performance and triggers retraining."""
    
    def __init__(self, performance_threshold: float = 0.8):
        """
        Initialize performance monitor.
        
        Args:
            performance_threshold: Minimum acceptable performance threshold
        """
        self.performance_threshold = performance_threshold
        self.performance_history = []
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Performance metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Add to history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'model_name': model_name,
                'metrics': metrics
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {}
    
    def should_retrain(self, model_name: str) -> bool:
        """
        Determine if model should be retrained.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if retraining is needed
        """
        try:
            # Get recent performance for this model
            recent_performance = [
                p for p in self.performance_history 
                if p['model_name'] == model_name and 
                (datetime.now() - p['timestamp']).days <= 7
            ]
            
            if not recent_performance:
                return False
            
            # Check if performance is below threshold
            latest_performance = recent_performance[-1]['metrics']
            overall_performance = np.mean(list(latest_performance.values()))
            
            return overall_performance < self.performance_threshold
            
        except Exception as e:
            logger.error(f"Error checking retraining need: {e}")
            return False


class ModelRetrainingPipeline:
    """Main pipeline for model retraining."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize retraining pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.s3_data_lake = S3DataLake(config['s3_manager'])
        self.mlflow_manager = MLflowManager(config['mlflow_tracking_uri'])
        self.sagemaker_manager = SageMakerModelManager(config['sagemaker_role_arn'])
        
        # Initialize pipelines
        self.anomaly_pipeline = AnomalyDetectionPipeline()
        self.failure_pipeline = FailurePredictionPipeline()
        self.health_pipeline = HealthScoringPipeline()
        
        # Initialize monitoring
        self.drift_detector = None
        self.performance_monitor = ModelPerformanceMonitor(
            performance_threshold=config.get('performance_threshold', 0.8)
        )
    
    def load_training_data(self, start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training data from S3.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            # Load processed data from S3
            data_summary = self.s3_data_lake.get_data_summary('processed')
            
            # Get data for the specified date range
            features_data = []
            labels_data = []
            
            # This is a simplified version - in practice, you'd query S3 for specific date ranges
            for obj in self.s3_data_lake.s3_manager.list_objects('processed-data/'):
                if start_date <= obj['last_modified'] <= end_date:
                    data = self.s3_data_lake.s3_manager.download_data(obj['key'], 'parquet')
                    if data is not None:
                        features_data.append(data)
            
            if not features_data:
                raise ValueError("No training data found for the specified date range")
            
            # Combine all data
            combined_data = pd.concat(features_data, ignore_index=True)
            
            # Separate features and labels
            feature_columns = [col for col in combined_data.columns 
                             if col not in ['is_anomaly', 'will_fail', 'health_score', 'maintenance_urgency']]
            
            features = combined_data[feature_columns]
            labels = combined_data[['is_anomaly', 'will_fail', 'health_score']]
            
            return features, labels
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> bool:
        """
        Detect data drift and determine if retraining is needed.
        
        Args:
            current_data: Current data to check for drift
            
        Returns:
            True if drift is detected
        """
        try:
            if self.drift_detector is None:
                # Load reference data
                reference_data = self.load_training_data(
                    datetime.now() - timedelta(days=30),
                    datetime.now() - timedelta(days=1)
                )[0]
                self.drift_detector = DataDriftDetector(reference_data)
            
            # Detect drift
            drift_results = self.drift_detector.detect_drift(current_data)
            
            logger.info(f"Drift detection results: {drift_results}")
            
            return drift_results['has_drift']
            
        except Exception as e:
            logger.error(f"Error detecting data drift: {e}")
            return False
    
    def retrain_anomaly_detection_model(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Retrain anomaly detection model.
        
        Args:
            features: Training features
            labels: Training labels
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting anomaly detection model retraining")
            
            # Prepare data
            X = features.values
            y = labels['is_anomaly'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.anomaly_pipeline.create_models()
            self.anomaly_pipeline.train_models(X_train, y_train)
            
            # Evaluate model
            predictions = self.anomaly_pipeline.predict_anomalies(X_test)
            y_pred = predictions['predictions']
            
            # Calculate metrics
            metrics = self.performance_monitor.evaluate_model_performance(y_test, y_pred, 'anomaly_detection')
            
            # Log to MLflow
            with self.mlflow_manager.start_run(run_name="anomaly-detection-retraining"):
                self.mlflow_manager.log_parameters(self.config.get('anomaly_params', {}))
                self.mlflow_manager.log_metrics(metrics)
                self.mlflow_manager.log_model(self.anomaly_pipeline, "anomaly-detection", "sklearn")
            
            # Save model
            model_path = f"models/anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            self.anomaly_pipeline.save_pipeline(model_path)
            
            # Upload to S3
            self.s3_data_lake.store_model(self.anomaly_pipeline, "anomaly-detection", "latest")
            
            logger.info("Anomaly detection model retraining completed")
            
            return {
                'model_name': 'anomaly_detection',
                'metrics': metrics,
                'model_path': model_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error retraining anomaly detection model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def retrain_failure_prediction_model(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Retrain failure prediction model.
        
        Args:
            features: Training features
            labels: Training labels
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting failure prediction model retraining")
            
            # Prepare data
            X = features.values
            y = labels['will_fail'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.failure_pipeline.create_models()
            self.failure_pipeline.train_models(X_train, y_train)
            
            # Evaluate model
            predictions = self.failure_pipeline.predict_failures(X_test)
            y_pred = predictions['predictions']
            
            # Calculate metrics
            metrics = self.performance_monitor.evaluate_model_performance(y_test, y_pred, 'failure_prediction')
            
            # Log to MLflow
            with self.mlflow_manager.start_run(run_name="failure-prediction-retraining"):
                self.mlflow_manager.log_parameters(self.config.get('failure_params', {}))
                self.mlflow_manager.log_metrics(metrics)
                self.mlflow_manager.log_model(self.failure_pipeline, "failure-prediction", "sklearn")
            
            # Save model
            model_path = f"models/failure_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            self.failure_pipeline.save_pipeline(model_path)
            
            # Upload to S3
            self.s3_data_lake.store_model(self.failure_pipeline, "failure-prediction", "latest")
            
            logger.info("Failure prediction model retraining completed")
            
            return {
                'model_name': 'failure_prediction',
                'metrics': metrics,
                'model_path': model_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error retraining failure prediction model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def retrain_health_scoring_model(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict[str, Any]:
        """
        Retrain health scoring model.
        
        Args:
            features: Training features
            labels: Training labels
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting health scoring model retraining")
            
            # Prepare data
            X = features.values
            y = labels['health_score'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.health_pipeline.create_models()
            self.health_pipeline.train_models(X_train, y_train)
            
            # Evaluate model
            predictions = self.health_pipeline.predict_health_scores(X_test)
            y_pred = predictions['health_scores']
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, r2_score
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Log to MLflow
            with self.mlflow_manager.start_run(run_name="health-scoring-retraining"):
                self.mlflow_manager.log_parameters(self.config.get('health_params', {}))
                self.mlflow_manager.log_metrics(metrics)
                self.mlflow_manager.log_model(self.health_pipeline, "health-scoring", "sklearn")
            
            # Save model
            model_path = f"models/health_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            self.health_pipeline.save_pipeline(model_path)
            
            # Upload to S3
            self.s3_data_lake.store_model(self.health_pipeline, "health-scoring", "latest")
            
            logger.info("Health scoring model retraining completed")
            
            return {
                'model_name': 'health_scoring',
                'metrics': metrics,
                'model_path': model_path,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error retraining health scoring model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_retraining_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete retraining pipeline.
        
        Returns:
            Pipeline results
        """
        try:
            logger.info("Starting model retraining pipeline")
            
            # Load recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last 7 days
            
            features, labels = self.load_training_data(start_date, end_date)
            
            # Check for data drift
            if not self.detect_data_drift(features):
                logger.info("No data drift detected, skipping retraining")
                return {'status': 'skipped', 'reason': 'No data drift detected'}
            
            # Retrain models
            results = {}
            
            # Anomaly detection
            anomaly_result = self.retrain_anomaly_detection_model(features, labels)
            results['anomaly_detection'] = anomaly_result
            
            # Failure prediction
            failure_result = self.retrain_failure_prediction_model(features, labels)
            results['failure_prediction'] = failure_result
            
            # Health scoring
            health_result = self.retrain_health_scoring_model(features, labels)
            results['health_scoring'] = health_result
            
            # Deploy new models to SageMaker
            self.deploy_models_to_sagemaker(results)
            
            logger.info("Model retraining pipeline completed")
            
            return {
                'status': 'completed',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in retraining pipeline: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def deploy_models_to_sagemaker(self, results: Dict[str, Any]):
        """
        Deploy retrained models to SageMaker.
        
        Args:
            results: Retraining results
        """
        try:
            for model_name, result in results.items():
                if result['status'] == 'success':
                    # Deploy to SageMaker
                    endpoint_name = f"{model_name}-endpoint"
                    
                    # This would typically involve uploading model artifacts to S3
                    # and creating/updating SageMaker endpoints
                    logger.info(f"Deploying {model_name} to SageMaker endpoint: {endpoint_name}")
                    
        except Exception as e:
            logger.error(f"Error deploying models to SageMaker: {e}")


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        's3_manager': None,  # Would be initialized with actual S3 manager
        'mlflow_tracking_uri': 'http://localhost:5000',
        'sagemaker_role_arn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
        'performance_threshold': 0.8,
        'anomaly_params': {'contamination': 0.1, 'nu': 0.1},
        'failure_params': {'n_estimators': 100, 'max_depth': 10},
        'health_params': {'n_estimators': 100, 'learning_rate': 0.1}
    }
    
    # Initialize retraining pipeline
    pipeline = ModelRetrainingPipeline(config)
    
    # Run retraining pipeline
    results = pipeline.run_retraining_pipeline()
    print(f"Retraining results: {results}")
