"""
AWS Lambda Functions

Contains Lambda function implementations for real-time processing, 
data ingestion, and model inference in the predictive maintenance system.
"""

import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging
import base64
import gzip
import io

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class LambdaDataProcessor:
    """Base class for Lambda data processing functions."""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.kinesis_client = boto3.client('kinesis')
        self.sns_client = boto3.client('sns')
    
    def process_kinesis_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single Kinesis record.
        
        Args:
            record: Kinesis record
            
        Returns:
            Processed data or None if failed
        """
        try:
            # Decode the data
            if 'kinesis' in record:
                data = base64.b64decode(record['kinesis']['data'])
            else:
                data = record['data']
            
            # Parse JSON
            sensor_data = json.loads(data.decode('utf-8'))
            
            # Process the data
            processed_data = self.process_sensor_data(sensor_data)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing Kinesis record: {e}")
            return None
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sensor data. Override in subclasses.
        
        Args:
            sensor_data: Raw sensor data
            
        Returns:
            Processed sensor data
        """
        return sensor_data


def lambda_handler(event, context):
    """
    Main Lambda handler for real-time data processing.
    
    Args:
        event: Lambda event (Kinesis records)
        context: Lambda context
        
    Returns:
        Processing results
    """
    processor = RealTimeDataProcessor()
    
    try:
        # Process Kinesis records
        processed_records = []
        
        for record in event['Records']:
            processed_data = processor.process_kinesis_record(record)
            if processed_data:
                processed_records.append(processed_data)
        
        # Store processed data
        if processed_records:
            processor.store_processed_data(processed_records)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Processed {len(processed_records)} records',
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Error in Lambda handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }


class RealTimeDataProcessor(LambdaDataProcessor):
    """Real-time data processor for sensor data."""
    
    def __init__(self):
        super().__init__()
        self.bucket_name = 'predictive-maintenance-data'
        self.anomaly_threshold = 0.8
        self.health_threshold = 0.3
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sensor data with anomaly detection and health scoring.
        
        Args:
            sensor_data: Raw sensor data
            
        Returns:
            Processed sensor data with analysis
        """
        try:
            # Add processing timestamp
            sensor_data['processed_at'] = datetime.now(timezone.utc).isoformat()
            
            # Basic anomaly detection (simplified)
            sensor_data['is_anomaly'] = self.detect_anomaly(sensor_data)
            
            # Health scoring
            sensor_data['health_score'] = self.calculate_health_score(sensor_data)
            sensor_data['health_status'] = self.get_health_status(sensor_data['health_score'])
            
            # Maintenance urgency
            sensor_data['maintenance_urgency'] = self.get_maintenance_urgency(sensor_data['health_score'])
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            return sensor_data
    
    def detect_anomaly(self, sensor_data: Dict[str, Any]) -> bool:
        """
        Simple anomaly detection based on value thresholds.
        
        Args:
            sensor_data: Sensor data
            
        Returns:
            True if anomaly detected, False otherwise
        """
        try:
            value = sensor_data.get('value', 0)
            sensor_type = sensor_data.get('sensor_type', '')
            
            # Define thresholds for different sensor types
            thresholds = {
                'vibration': 1.0,
                'temperature': 100.0,
                'pressure': 150.0,
                'rpm': 2500.0
            }
            
            threshold = thresholds.get(sensor_type, 1.0)
            return value > threshold
            
        except Exception:
            return False
    
    def calculate_health_score(self, sensor_data: Dict[str, Any]) -> float:
        """
        Calculate health score for equipment.
        
        Args:
            sensor_data: Sensor data
            
        Returns:
            Health score (0-1, higher is better)
        """
        try:
            # Simple health scoring based on value ranges
            value = sensor_data.get('value', 0)
            sensor_type = sensor_data.get('sensor_type', '')
            
            # Normalize value to 0-1 scale (simplified)
            if sensor_type == 'vibration':
                health_score = max(0, 1 - value)  # Lower vibration is better
            elif sensor_type == 'temperature':
                health_score = max(0, 1 - (value - 20) / 60)  # 20-80°C is good
            elif sensor_type == 'pressure':
                health_score = max(0, 1 - abs(value - 50) / 50)  # 50 is optimal
            else:
                health_score = 0.5  # Default neutral score
            
            return min(1.0, max(0.0, health_score))
            
        except Exception:
            return 0.5
    
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
    
    def store_processed_data(self, processed_records: List[Dict[str, Any]]):
        """
        Store processed data in S3.
        
        Args:
            processed_records: List of processed records
        """
        try:
            # Convert to DataFrame
            df = pd.DataFrame(processed_records)
            
            # Create S3 key with timestamp
            timestamp = datetime.now(timezone.utc)
            key = f"processed-data/{timestamp.strftime('%Y/%m/%d/%H')}/processed_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
            
            # Convert to parquet
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Stored {len(processed_records)} processed records to S3")
            
        except Exception as e:
            logger.error(f"Error storing processed data: {e}")
    
    def send_alert(self, sensor_data: Dict[str, Any], alert_type: str):
        """
        Send alert via SNS.
        
        Args:
            sensor_data: Sensor data that triggered alert
            alert_type: Type of alert
        """
        try:
            message = {
                'equipment_id': sensor_data.get('equipment_id', 'unknown'),
                'sensor_type': sensor_data.get('sensor_type', 'unknown'),
                'value': sensor_data.get('value', 0),
                'health_score': sensor_data.get('health_score', 0),
                'alert_type': alert_type,
                'timestamp': sensor_data.get('processed_at', datetime.now(timezone.utc).isoformat())
            }
            
            # Send to SNS topic
            self.sns_client.publish(
                TopicArn='arn:aws:sns:us-east-1:123456789012:predictive-maintenance-alerts',
                Message=json.dumps(message),
                Subject=f'Predictive Maintenance Alert - {alert_type}'
            )
            
            logger.info(f"Sent {alert_type} alert for equipment {sensor_data.get('equipment_id')}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")


def anomaly_detection_lambda(event, context):
    """
    Lambda function for anomaly detection.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Anomaly detection results
    """
    processor = AnomalyDetectionProcessor()
    
    try:
        results = []
        
        for record in event['Records']:
            processed_data = processor.process_kinesis_record(record)
            if processed_data and processed_data.get('is_anomaly', False):
                results.append(processed_data)
                
                # Send alert for anomalies
                processor.send_alert(processed_data, 'anomaly_detected')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'anomalies_detected': len(results),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection Lambda: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


class AnomalyDetectionProcessor(LambdaDataProcessor):
    """Anomaly detection processor for Lambda."""
    
    def __init__(self):
        super().__init__()
        self.anomaly_threshold = 0.8
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data with anomaly detection."""
        # Add anomaly detection logic here
        sensor_data['is_anomaly'] = self.detect_anomaly(sensor_data)
        sensor_data['anomaly_confidence'] = self.calculate_anomaly_confidence(sensor_data)
        
        return sensor_data
    
    def detect_anomaly(self, sensor_data: Dict[str, Any]) -> bool:
        """Detect anomalies in sensor data."""
        # Implement your anomaly detection logic here
        value = sensor_data.get('value', 0)
        return value > self.anomaly_threshold
    
    def calculate_anomaly_confidence(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate confidence score for anomaly detection."""
        # Implement confidence calculation here
        return 0.8


def health_scoring_lambda(event, context):
    """
    Lambda function for health scoring.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Health scoring results
    """
    processor = HealthScoringProcessor()
    
    try:
        results = []
        
        for record in event['Records']:
            processed_data = processor.process_kinesis_record(record)
            if processed_data:
                results.append(processed_data)
                
                # Send alert for critical health
                if processed_data.get('health_score', 1) < 0.2:
                    processor.send_alert(processed_data, 'critical_health')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'records_processed': len(results),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Error in health scoring Lambda: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


class HealthScoringProcessor(LambdaDataProcessor):
    """Health scoring processor for Lambda."""
    
    def __init__(self):
        super().__init__()
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data with health scoring."""
        sensor_data['health_score'] = self.calculate_health_score(sensor_data)
        sensor_data['health_status'] = self.get_health_status(sensor_data['health_score'])
        sensor_data['maintenance_urgency'] = self.get_maintenance_urgency(sensor_data['health_score'])
        
        return sensor_data
    
    def calculate_health_score(self, sensor_data: Dict[str, Any]) -> float:
        """Calculate health score for equipment."""
        # Implement health scoring logic here
        return 0.7
    
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


def model_inference_lambda(event, context):
    """
    Lambda function for model inference.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Model inference results
    """
    processor = ModelInferenceProcessor()
    
    try:
        results = []
        
        for record in event['Records']:
            processed_data = processor.process_kinesis_record(record)
            if processed_data:
                results.append(processed_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'predictions_made': len(results),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        }
        
    except Exception as e:
        logger.error(f"Error in model inference Lambda: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


class ModelInferenceProcessor(LambdaDataProcessor):
    """Model inference processor for Lambda."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from S3."""
        try:
            # Load model from S3
            response = self.s3_client.get_object(
                Bucket='predictive-maintenance-models',
                Key='models/anomaly_detection/latest/model.pkl'
            )
            
            # Deserialize model
            import pickle
            self.model = pickle.loads(response['Body'].read())
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensor data with model inference."""
        if self.model:
            # Prepare features for model
            features = self.prepare_features(sensor_data)
            
            # Make prediction
            prediction = self.model.predict([features])
            sensor_data['model_prediction'] = prediction[0]
            sensor_data['model_confidence'] = self.calculate_confidence(features)
        
        return sensor_data
    
    def prepare_features(self, sensor_data: Dict[str, Any]) -> List[float]:
        """Prepare features for model inference."""
        # Extract features from sensor data
        features = [
            sensor_data.get('value', 0),
            sensor_data.get('temperature', 0),
            sensor_data.get('rpm', 0),
            sensor_data.get('load', 0)
        ]
        return features
    
    def calculate_confidence(self, features: List[float]) -> float:
        """Calculate prediction confidence."""
        # Implement confidence calculation
        return 0.8


# Example usage and testing
if __name__ == "__main__":
    # Test the processors
    processor = RealTimeDataProcessor()
    
    # Sample sensor data
    sample_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'equipment_id': 'MOTOR_001',
        'sensor_type': 'vibration',
        'value': 0.8,
        'unit': 'g',
        'location': 'bearing_1'
    }
    
    # Process data
    processed_data = processor.process_sensor_data(sample_data)
    print(f"Processed data: {processed_data}")
    
    # Test anomaly detection
    anomaly_processor = AnomalyDetectionProcessor()
    anomaly_data = anomaly_processor.process_sensor_data(sample_data)
    print(f"Anomaly detection: {anomaly_data}")
    
    # Test health scoring
    health_processor = HealthScoringProcessor()
    health_data = health_processor.process_sensor_data(sample_data)
    print(f"Health scoring: {health_data}")
