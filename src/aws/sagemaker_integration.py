"""
SageMaker Integration Module

Handles model training, deployment, and inference using Amazon SageMaker
for the predictive maintenance system.
"""

import boto3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import logging
import joblib
import os
import tarfile
import io
from sagemaker.sklearn import SKLearn
from sagemaker.tensorflow import TensorFlow
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerModelManager:
    """Manages SageMaker model training and deployment."""
    
    def __init__(self, role_arn: str, region_name: str = 'us-east-1'):
        """
        Initialize SageMaker model manager.
        
        Args:
            role_arn: SageMaker execution role ARN
            region_name: AWS region name
        """
        self.role_arn = role_arn
        self.region_name = region_name
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.session = boto3.Session(region_name=region_name)
        
        # SageMaker session
        from sagemaker.session import Session
        self.sagemaker_session = Session(boto_session=self.session)
    
    def create_training_job(self, model_type: str, training_data_path: str,
                          model_name: str, instance_type: str = 'ml.m5.large',
                          instance_count: int = 1) -> str:
        """
        Create a SageMaker training job.
        
        Args:
            model_type: Type of model ('sklearn', 'tensorflow', 'pytorch')
            training_data_path: S3 path to training data
            model_name: Name for the model
            instance_type: EC2 instance type for training
            instance_count: Number of instances
            
        Returns:
            Training job name
        """
        try:
            job_name = f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            if model_type == 'sklearn':
                estimator = self._create_sklearn_estimator(job_name, instance_type, instance_count)
            elif model_type == 'tensorflow':
                estimator = self._create_tensorflow_estimator(job_name, instance_type, instance_count)
            elif model_type == 'pytorch':
                estimator = self._create_pytorch_estimator(job_name, instance_type, instance_count)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Start training job
            estimator.fit({'training': training_data_path})
            
            logger.info(f"Training job {job_name} completed successfully")
            return job_name
            
        except Exception as e:
            logger.error(f"Error creating training job: {e}")
            raise
    
    def _create_sklearn_estimator(self, job_name: str, instance_type: str, instance_count: int):
        """Create SKLearn estimator."""
        return SKLearn(
            entry_point='train_sklearn.py',
            role=self.role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='0.23-1',
            py_version='py3',
            sagemaker_session=self.sagemaker_session,
            job_name=job_name
        )
    
    def _create_tensorflow_estimator(self, job_name: str, instance_type: str, instance_count: int):
        """Create TensorFlow estimator."""
        return TensorFlow(
            entry_point='train_tensorflow.py',
            role=self.role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='2.8.0',
            py_version='py3',
            sagemaker_session=self.sagemaker_session,
            job_name=job_name
        )
    
    def _create_pytorch_estimator(self, job_name: str, instance_type: str, instance_count: int):
        """Create PyTorch estimator."""
        return PyTorch(
            entry_point='train_pytorch.py',
            role=self.role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            framework_version='1.12.0',
            py_version='py3',
            sagemaker_session=self.sagemaker_session,
            job_name=job_name
        )
    
    def deploy_model(self, model_name: str, model_artifact_path: str,
                    instance_type: str = 'ml.m5.large', instance_count: int = 1) -> str:
        """
        Deploy a trained model to SageMaker endpoint.
        
        Args:
            model_name: Name for the deployed model
            instance_type: EC2 instance type for inference
            instance_count: Number of instances
            
        Returns:
            Endpoint name
        """
        try:
            # Create model
            model = Model(
                image_uri=self._get_model_image_uri(),
                model_data=model_artifact_path,
                role=self.role_arn,
                sagemaker_session=self.sagemaker_session
            )
            
            # Deploy model
            predictor = model.deploy(
                initial_instance_count=instance_count,
                instance_type=instance_type,
                endpoint_name=model_name
            )
            
            logger.info(f"Model {model_name} deployed successfully")
            return model_name
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def _get_model_image_uri(self) -> str:
        """Get the appropriate model image URI."""
        # This would typically be determined based on the model type
        return "763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:0.23-1-cpu-py3"
    
    def create_batch_transform_job(self, model_name: str, input_data_path: str,
                                 output_data_path: str, instance_type: str = 'ml.m5.large',
                                 instance_count: int = 1) -> str:
        """
        Create a batch transform job for batch inference.
        
        Args:
            model_name: Name of the model
            input_data_path: S3 path to input data
            output_data_path: S3 path for output data
            instance_type: EC2 instance type
            instance_count: Number of instances
            
        Returns:
            Transform job name
        """
        try:
            job_name = f"batch-transform-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Create transform job
            response = self.sagemaker_client.create_transform_job(
                TransformJobName=job_name,
                ModelName=model_name,
                MaxConcurrentTransforms=instance_count,
                MaxPayloadInMB=100,
                BatchStrategy='MultiRecord',
                TransformInput={
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': input_data_path
                        }
                    },
                    'ContentType': 'application/json',
                    'SplitType': 'Line'
                },
                TransformOutput={
                    'S3OutputPath': output_data_path
                },
                TransformResources={
                    'InstanceType': instance_type,
                    'InstanceCount': instance_count
                }
            )
            
            logger.info(f"Batch transform job {job_name} created successfully")
            return job_name
            
        except Exception as e:
            logger.error(f"Error creating batch transform job: {e}")
            raise
    
    def get_training_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of a training job."""
        try:
            response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)
            return {
                'status': response['TrainingJobStatus'],
                'creation_time': response['CreationTime'],
                'end_time': response.get('TrainingEndTime'),
                'metrics': response.get('FinalMetricDataList', [])
            }
        except Exception as e:
            logger.error(f"Error getting training job status: {e}")
            return {'status': 'Unknown', 'error': str(e)}
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """Get status of an endpoint."""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            return {
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified': response['LastModifiedTime']
            }
        except Exception as e:
            logger.error(f"Error getting endpoint status: {e}")
            return {'status': 'Unknown', 'error': str(e)}
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a SageMaker endpoint."""
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting endpoint: {e}")
            return False


class SageMakerInferenceManager:
    """Manages SageMaker model inference."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize SageMaker inference manager.
        
        Args:
            region_name: AWS region name
        """
        self.region_name = region_name
        self.session = boto3.Session(region_name=region_name)
        self.sagemaker_session = None
        self.predictors = {}
    
    def create_predictor(self, endpoint_name: str, model_type: str = 'sklearn') -> Predictor:
        """
        Create a predictor for real-time inference.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            model_type: Type of model ('sklearn', 'tensorflow', 'pytorch')
            
        Returns:
            SageMaker predictor
        """
        try:
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=self.sagemaker_session,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            self.predictors[endpoint_name] = predictor
            logger.info(f"Predictor created for endpoint {endpoint_name}")
            return predictor
            
        except Exception as e:
            logger.error(f"Error creating predictor: {e}")
            raise
    
    def predict_anomaly(self, features: List[float], endpoint_name: str) -> Dict[str, Any]:
        """
        Predict anomaly using SageMaker endpoint.
        
        Args:
            features: Input features
            endpoint_name: Name of the endpoint
            
        Returns:
            Prediction results
        """
        try:
            if endpoint_name not in self.predictors:
                self.create_predictor(endpoint_name)
            
            predictor = self.predictors[endpoint_name]
            
            # Prepare input data
            input_data = {
                'features': features,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Make prediction
            response = predictor.predict(input_data)
            
            return {
                'prediction': response.get('prediction', 0),
                'confidence': response.get('confidence', 0),
                'is_anomaly': response.get('is_anomaly', False),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'prediction': 0,
                'confidence': 0,
                'is_anomaly': False,
                'error': str(e)
            }
    
    def predict_failure(self, features: List[float], endpoint_name: str) -> Dict[str, Any]:
        """
        Predict failure using SageMaker endpoint.
        
        Args:
            features: Input features
            endpoint_name: Name of the endpoint
            
        Returns:
            Prediction results
        """
        try:
            if endpoint_name not in self.predictors:
                self.create_predictor(endpoint_name)
            
            predictor = self.predictors[endpoint_name]
            
            # Prepare input data
            input_data = {
                'features': features,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Make prediction
            response = predictor.predict(input_data)
            
            return {
                'failure_probability': response.get('failure_probability', 0),
                'time_to_failure': response.get('time_to_failure', 0),
                'maintenance_urgency': response.get('maintenance_urgency', 'low'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'failure_probability': 0,
                'time_to_failure': 0,
                'maintenance_urgency': 'low',
                'error': str(e)
            }
    
    def predict_health_score(self, features: List[float], endpoint_name: str) -> Dict[str, Any]:
        """
        Predict health score using SageMaker endpoint.
        
        Args:
            features: Input features
            endpoint_name: Name of the endpoint
            
        Returns:
            Prediction results
        """
        try:
            if endpoint_name not in self.predictors:
                self.create_predictor(endpoint_name)
            
            predictor = self.predictors[endpoint_name]
            
            # Prepare input data
            input_data = {
                'features': features,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Make prediction
            response = predictor.predict(input_data)
            
            return {
                'health_score': response.get('health_score', 0.5),
                'health_status': response.get('health_status', 'unknown'),
                'maintenance_urgency': response.get('maintenance_urgency', 'low'),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'health_score': 0.5,
                'health_status': 'unknown',
                'maintenance_urgency': 'low',
                'error': str(e)
            }


class SageMakerModelRegistry:
    """Manages SageMaker model registry and versioning."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize SageMaker model registry.
        
        Args:
            region_name: AWS region name
        """
        self.region_name = region_name
        self.sagemaker_client = boto3.client('sagemaker', region_name=region_name)
    
    def create_model_package_group(self, group_name: str, description: str = '') -> bool:
        """
        Create a model package group for organizing models.
        
        Args:
            group_name: Name of the model package group
            description: Description of the group
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.sagemaker_client.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription=description
            )
            
            logger.info(f"Model package group {group_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model package group: {e}")
            return False
    
    def register_model(self, model_name: str, model_artifact_path: str,
                      model_package_group_name: str, description: str = '') -> str:
        """
        Register a model in the model registry.
        
        Args:
            model_name: Name of the model
            model_artifact_path: S3 path to model artifacts
            model_package_group_name: Name of the model package group
            description: Description of the model
            
        Returns:
            Model package ARN
        """
        try:
            response = self.sagemaker_client.create_model_package(
                ModelPackageName=model_name,
                ModelPackageGroupName=model_package_group_name,
                ModelPackageDescription=description,
                InferenceSpecification={
                    'Containers': [
                        {
                            'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:0.23-1-cpu-py3',
                            'ModelDataUrl': model_artifact_path
                        }
                    ],
                    'SupportedContentTypes': ['application/json'],
                    'SupportedResponseMIMETypes': ['application/json']
                }
            )
            
            model_package_arn = response['ModelPackageArn']
            logger.info(f"Model {model_name} registered successfully: {model_package_arn}")
            return model_package_arn
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def approve_model(self, model_package_arn: str) -> bool:
        """
        Approve a model for deployment.
        
        Args:
            model_package_arn: ARN of the model package
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.sagemaker_client.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus='Approved'
            )
            
            logger.info(f"Model {model_package_arn} approved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error approving model: {e}")
            return False
    
    def list_models(self, model_package_group_name: str) -> List[Dict[str, Any]]:
        """
        List models in a model package group.
        
        Args:
            model_package_group_name: Name of the model package group
            
        Returns:
            List of model information
        """
        try:
            response = self.sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name
            )
            
            models = []
            for model in response['ModelPackageSummaryList']:
                models.append({
                    'model_package_arn': model['ModelPackageArn'],
                    'model_package_name': model['ModelPackageName'],
                    'model_package_status': model['ModelPackageStatus'],
                    'creation_time': model['CreationTime'],
                    'model_approval_status': model.get('ModelApprovalStatus', 'Unknown')
                })
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize SageMaker manager
    role_arn = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
    model_manager = SageMakerModelManager(role_arn)
    
    # Create training job
    training_data_path = "s3://predictive-maintenance-data/training-data/"
    job_name = model_manager.create_training_job(
        model_type='sklearn',
        training_data_path=training_data_path,
        model_name='anomaly-detection',
        instance_type='ml.m5.large'
    )
    
    print(f"Training job created: {job_name}")
    
    # Check training job status
    status = model_manager.get_training_job_status(job_name)
    print(f"Training job status: {status}")
    
    # Deploy model
    model_artifact_path = f"s3://predictive-maintenance-models/{job_name}/output/model.tar.gz"
    endpoint_name = model_manager.deploy_model(
        model_name='anomaly-detection-endpoint',
        model_artifact_path=model_artifact_path,
        instance_type='ml.m5.large'
    )
    
    print(f"Model deployed to endpoint: {endpoint_name}")
    
    # Create inference manager
    inference_manager = SageMakerInferenceManager()
    
    # Make predictions
    features = [0.5, 75.0, 1800, 85.0]  # Example features
    
    anomaly_prediction = inference_manager.predict_anomaly(features, endpoint_name)
    print(f"Anomaly prediction: {anomaly_prediction}")
    
    failure_prediction = inference_manager.predict_failure(features, endpoint_name)
    print(f"Failure prediction: {failure_prediction}")
    
    health_prediction = inference_manager.predict_health_score(features, endpoint_name)
    print(f"Health prediction: {health_prediction}")
    
    # Model registry
    registry = SageMakerModelRegistry()
    
    # Create model package group
    registry.create_model_package_group(
        group_name='predictive-maintenance-models',
        description='Models for predictive maintenance system'
    )
    
    # Register model
    model_package_arn = registry.register_model(
        model_name='anomaly-detection-v1',
        model_artifact_path=model_artifact_path,
        model_package_group_name='predictive-maintenance-models',
        description='Anomaly detection model v1.0'
    )
    
    print(f"Model registered: {model_package_arn}")
    
    # Approve model
    registry.approve_model(model_package_arn)
    print("Model approved for deployment")
