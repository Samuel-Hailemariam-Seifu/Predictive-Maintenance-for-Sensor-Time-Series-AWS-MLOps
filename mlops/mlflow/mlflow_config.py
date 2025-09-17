"""
MLflow Configuration and Setup

Configures MLflow for experiment tracking, model registry, and deployment
in the predictive maintenance system.
"""

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import os
import json
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowManager:
    """Manages MLflow experiments and model registry."""
    
    def __init__(self, tracking_uri: str = None, experiment_name: str = "predictive-maintenance"):
        """
        Initialize MLflow manager.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
        """
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        self.experiment_name = experiment_name
        self.client = None
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Initialize client
        try:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            logger.info(f"Connected to MLflow tracking server: {self.tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")
            raise
    
    def create_experiment(self, experiment_name: str = None) -> str:
        """
        Create or get experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Experiment ID
        """
        if experiment_name is None:
            experiment_name = self.experiment_name
        
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {e}")
            raise
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name of the run
            tags: Tags for the run
            
        Returns:
            Active MLflow run
        """
        try:
            # Create experiment if it doesn't exist
            experiment_id = self.create_experiment()
            
            # Start run
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags=tags
            )
            
            logger.info(f"Started run: {run_name}")
            return run
            
        except Exception as e:
            logger.error(f"Error starting run: {e}")
            raise
    
    def log_parameters(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params: Parameters to log
        """
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logger.info(f"Logged {len(params)} parameters")
            
        except Exception as e:
            logger.error(f"Error logging parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Metrics to log
            step: Step number for the metrics
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.info(f"Logged {len(metrics)} metrics")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn"):
        """
        Log model to MLflow.
        
        Args:
            model: Trained model
            model_name: Name of the model
            model_type: Type of model ('sklearn', 'tensorflow', 'pytorch')
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, model_name)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, model_name)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Logged {model_type} model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def log_artifacts(self, artifacts_path: str, artifact_path: str = None):
        """
        Log artifacts to the current run.
        
        Args:
            artifacts_path: Path to artifacts
            artifact_path: Path within the run to store artifacts
        """
        try:
            mlflow.log_artifacts(artifacts_path, artifact_path)
            logger.info(f"Logged artifacts from: {artifacts_path}")
            
        except Exception as e:
            logger.error(f"Error logging artifacts: {e}")
    
    def register_model(self, model_name: str, model_version: str = None, 
                      description: str = None) -> str:
        """
        Register model in the model registry.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            description: Description of the model
            
        Returns:
            Model version
        """
        try:
            # Get the current run
            run_id = mlflow.active_run().info.run_id
            
            # Register model
            model_uri = f"runs:/{run_id}/{model_name}"
            
            if model_version:
                model_version = self.client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    description=description
                )
            else:
                model_version = self.client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name} (Version: {model_version.version})")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def get_best_model(self, experiment_name: str, metric_name: str, 
                      ascending: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best model from an experiment.
        
        Args:
            experiment_name: Name of the experiment
            metric_name: Name of the metric to optimize
            ascending: Whether to sort in ascending order
            
        Returns:
            Best model information
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                return None
            
            # Get all runs
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            if not runs:
                return None
            
            best_run = runs[0]
            
            return {
                'run_id': best_run.info.run_id,
                'run_name': best_run.data.tags.get('mlflow.runName'),
                'metrics': best_run.data.metrics,
                'parameters': best_run.data.params,
                'model_uri': f"runs:/{best_run.info.run_id}/model"
            }
            
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return None
    
    def list_models(self, experiment_name: str = None) -> List[Dict[str, Any]]:
        """
        List all models in the registry.
        
        Args:
            experiment_name: Name of the experiment to filter by
            
        Returns:
            List of model information
        """
        try:
            models = self.client.search_registered_models()
            model_list = []
            
            for model in models:
                model_info = {
                    'name': model.name,
                    'latest_versions': [],
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp
                }
                
                for version in model.latest_versions:
                    model_info['latest_versions'].append({
                        'version': version.version,
                        'stage': version.current_stage,
                        'description': version.description,
                        'creation_timestamp': version.creation_timestamp
                    })
                
                model_list.append(model_info)
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """
        Promote model to a specific stage.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            
            logger.info(f"Promoted model {model_name} version {version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
    
    def get_model_uri(self, model_name: str, version: str = None, stage: str = None) -> str:
        """
        Get model URI for deployment.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            stage: Stage of the model
            
        Returns:
            Model URI
        """
        try:
            if stage:
                return f"models:/{model_name}/{stage}"
            elif version:
                return f"models:/{model_name}/{version}"
            else:
                return f"models:/{model_name}/latest"
                
        except Exception as e:
            logger.error(f"Error getting model URI: {e}")
            raise


class ModelRegistry:
    """Manages model registry operations."""
    
    def __init__(self, mlflow_manager: MLflowManager):
        """
        Initialize model registry.
        
        Args:
            mlflow_manager: MLflowManager instance
        """
        self.mlflow_manager = mlflow_manager
        self.client = mlflow_manager.client
    
    def register_anomaly_detection_model(self, model, metrics: Dict[str, float], 
                                       parameters: Dict[str, Any]) -> str:
        """
        Register anomaly detection model.
        
        Args:
            model: Trained model
            metrics: Model metrics
            parameters: Model parameters
            
        Returns:
            Model version
        """
        with self.mlflow_manager.start_run(run_name="anomaly-detection-training"):
            # Log parameters
            self.mlflow_manager.log_parameters(parameters)
            
            # Log metrics
            self.mlflow_manager.log_metrics(metrics)
            
            # Log model
            self.mlflow_manager.log_model(model, "anomaly-detection", "sklearn")
            
            # Register model
            version = self.mlflow_manager.register_model(
                "anomaly-detection",
                description="Anomaly detection model for sensor data"
            )
            
            return version
    
    def register_failure_prediction_model(self, model, metrics: Dict[str, float], 
                                        parameters: Dict[str, Any]) -> str:
        """
        Register failure prediction model.
        
        Args:
            model: Trained model
            metrics: Model metrics
            parameters: Model parameters
            
        Returns:
            Model version
        """
        with self.mlflow_manager.start_run(run_name="failure-prediction-training"):
            # Log parameters
            self.mlflow_manager.log_parameters(parameters)
            
            # Log metrics
            self.mlflow_manager.log_metrics(metrics)
            
            # Log model
            self.mlflow_manager.log_model(model, "failure-prediction", "sklearn")
            
            # Register model
            version = self.mlflow_manager.register_model(
                "failure-prediction",
                description="Failure prediction model for equipment maintenance"
            )
            
            return version
    
    def register_health_scoring_model(self, model, metrics: Dict[str, float], 
                                     parameters: Dict[str, Any]) -> str:
        """
        Register health scoring model.
        
        Args:
            model: Trained model
            metrics: Model metrics
            parameters: Model parameters
            
        Returns:
            Model version
        """
        with self.mlflow_manager.start_run(run_name="health-scoring-training"):
            # Log parameters
            self.mlflow_manager.log_parameters(parameters)
            
            # Log metrics
            self.mlflow_manager.log_metrics(metrics)
            
            # Log model
            self.mlflow_manager.log_model(model, "health-scoring", "sklearn")
            
            # Register model
            version = self.mlflow_manager.register_model(
                "health-scoring",
                description="Health scoring model for equipment condition assessment"
            )
            
            return version
    
    def get_production_models(self) -> Dict[str, str]:
        """
        Get production model URIs.
        
        Returns:
            Dictionary of model names to URIs
        """
        try:
            models = {}
            
            # Get anomaly detection model
            anomaly_uri = self.mlflow_manager.get_model_uri("anomaly-detection", stage="Production")
            models["anomaly_detection"] = anomaly_uri
            
            # Get failure prediction model
            failure_uri = self.mlflow_manager.get_model_uri("failure-prediction", stage="Production")
            models["failure_prediction"] = failure_uri
            
            # Get health scoring model
            health_uri = self.mlflow_manager.get_model_uri("health-scoring", stage="Production")
            models["health_scoring"] = health_uri
            
            return models
            
        except Exception as e:
            logger.error(f"Error getting production models: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Initialize MLflow manager
    mlflow_manager = MLflowManager(
        tracking_uri="http://localhost:5000",
        experiment_name="predictive-maintenance"
    )
    
    # Create experiment
    experiment_id = mlflow_manager.create_experiment()
    print(f"Experiment ID: {experiment_id}")
    
    # Start a run
    with mlflow_manager.start_run(run_name="test-run"):
        # Log parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42
        }
        mlflow_manager.log_parameters(params)
        
        # Log metrics
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.94,
            "f1_score": 0.935
        }
        mlflow_manager.log_metrics(metrics)
        
        # Log artifacts
        mlflow_manager.log_artifacts("models/", "model_artifacts")
    
    # List models
    models = mlflow_manager.list_models()
    print(f"Registered models: {models}")
    
    # Get best model
    best_model = mlflow_manager.get_best_model("predictive-maintenance", "accuracy")
    if best_model:
        print(f"Best model: {best_model}")
    
    # Model registry
    registry = ModelRegistry(mlflow_manager)
    
    # Get production models
    production_models = registry.get_production_models()
    print(f"Production models: {production_models}")
