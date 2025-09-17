"""
S3 Integration Module

Handles data storage, retrieval, and management in Amazon S3 for the predictive maintenance system.
"""

import boto3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import logging
from botocore.exceptions import ClientError, NoCredentialsError
import io
import gzip
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DataManager:
    """Manages data operations with Amazon S3."""
    
    def __init__(self, bucket_name: str, region_name: str = 'us-east-1', 
                 aws_access_key_id: str = None, aws_secret_access_key: str = None):
        """
        Initialize S3 data manager.
        
        Args:
            bucket_name: Name of the S3 bucket
            region_name: AWS region name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.bucket_name = bucket_name
        self.region_name = region_name
        
        # Initialize S3 client
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key
                )
            else:
                # Use default credentials (IAM role, environment variables, etc.)
                self.s3_client = boto3.client('s3', region_name=region_name)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error connecting to S3 bucket {bucket_name}: {e}")
            raise
    
    def upload_data(self, data: Union[pd.DataFrame, dict, list], 
                   key: str, format: str = 'parquet', compress: bool = True) -> bool:
        """
        Upload data to S3.
        
        Args:
            data: Data to upload (DataFrame, dict, or list)
            key: S3 object key
            format: Data format ('parquet', 'json', 'csv', 'pickle')
            compress: Whether to compress the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert data to appropriate format
            if format == 'parquet':
                buffer = io.BytesIO()
                data.to_parquet(buffer, index=False)
                file_data = buffer.getvalue()
                content_type = 'application/octet-stream'
                
            elif format == 'json':
                if isinstance(data, pd.DataFrame):
                    file_data = data.to_json(orient='records', date_format='iso')
                else:
                    file_data = json.dumps(data, default=str)
                file_data = file_data.encode('utf-8')
                content_type = 'application/json'
                
            elif format == 'csv':
                if isinstance(data, pd.DataFrame):
                    file_data = data.to_csv(index=False).encode('utf-8')
                else:
                    file_data = str(data).encode('utf-8')
                content_type = 'text/csv'
                
            elif format == 'pickle':
                file_data = pickle.dumps(data)
                content_type = 'application/octet-stream'
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Compress if requested
            if compress and format != 'parquet':  # Parquet is already compressed
                file_data = gzip.compress(file_data)
                key += '.gz'
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_data,
                ContentType=content_type
            )
            
            logger.info(f"Successfully uploaded data to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading data to S3: {e}")
            return False
    
    def download_data(self, key: str, format: str = 'parquet') -> Optional[Union[pd.DataFrame, dict, list]]:
        """
        Download data from S3.
        
        Args:
            key: S3 object key
            format: Expected data format ('parquet', 'json', 'csv', 'pickle')
            
        Returns:
            Downloaded data or None if failed
        """
        try:
            # Download from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            file_data = response['Body'].read()
            
            # Check if data is compressed
            is_compressed = key.endswith('.gz')
            if is_compressed:
                file_data = gzip.decompress(file_data)
            
            # Parse data based on format
            if format == 'parquet':
                buffer = io.BytesIO(file_data)
                return pd.read_parquet(buffer)
                
            elif format == 'json':
                data = json.loads(file_data.decode('utf-8'))
                return data
                
            elif format == 'csv':
                buffer = io.StringIO(file_data.decode('utf-8'))
                return pd.read_csv(buffer)
                
            elif format == 'pickle':
                return pickle.loads(file_data)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.warning(f"Object not found: s3://{self.bucket_name}/{key}")
            else:
                logger.error(f"Error downloading data from S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing downloaded data: {e}")
            return None
    
    def list_objects(self, prefix: str = '', max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in the S3 bucket.
        
        Args:
            prefix: Object key prefix to filter
            max_keys: Maximum number of objects to return
            
        Returns:
            List of object metadata
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Error listing objects: {e}")
            return []
    
    def delete_object(self, key: str) -> bool:
        """
        Delete an object from S3.
        
        Args:
            key: S3 object key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Successfully deleted object: s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting object: {e}")
            return False
    
    def create_folder_structure(self, base_path: str = 'predictive-maintenance') -> bool:
        """
        Create a folder structure in S3 for the project.
        
        Args:
            base_path: Base path for the project
            
        Returns:
            True if successful, False otherwise
        """
        try:
            folders = [
                f"{base_path}/raw-data/",
                f"{base_path}/processed-data/",
                f"{base_path}/models/",
                f"{base_path}/logs/",
                f"{base_path}/config/",
                f"{base_path}/backups/"
            ]
            
            for folder in folders:
                # Create empty object to represent folder
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=folder,
                    Body=b'',
                    ContentType='application/x-directory'
                )
            
            logger.info(f"Created folder structure in s3://{self.bucket_name}/{base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating folder structure: {e}")
            return False


class S3DataLake:
    """Manages data lake operations in S3."""
    
    def __init__(self, s3_manager: S3DataManager):
        """
        Initialize S3 data lake.
        
        Args:
            s3_manager: S3DataManager instance
        """
        self.s3_manager = s3_manager
        self.base_path = 'predictive-maintenance'
    
    def store_raw_data(self, data: pd.DataFrame, equipment_id: str, 
                      timestamp: datetime = None) -> bool:
        """
        Store raw sensor data in the data lake.
        
        Args:
            data: Raw sensor data
            equipment_id: Equipment identifier
            timestamp: Data timestamp (defaults to now)
            
        Returns:
            True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Create key with partitioning
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        hour = timestamp.hour
        
        key = f"{self.base_path}/raw-data/year={year}/month={month}/day={day}/hour={hour}/equipment_{equipment_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
        
        return self.s3_manager.upload_data(data, key, format='parquet')
    
    def store_processed_data(self, data: pd.DataFrame, process_type: str,
                           equipment_id: str, timestamp: datetime = None) -> bool:
        """
        Store processed data in the data lake.
        
        Args:
            data: Processed data
            process_type: Type of processing (e.g., 'cleaned', 'features', 'predictions')
            equipment_id: Equipment identifier
            timestamp: Data timestamp (defaults to now)
            
        Returns:
            True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Create key with partitioning
        year = timestamp.year
        month = timestamp.month
        day = timestamp.day
        
        key = f"{self.base_path}/processed-data/type={process_type}/year={year}/month={month}/day={day}/equipment_{equipment_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.parquet"
        
        return self.s3_manager.upload_data(data, key, format='parquet')
    
    def store_model(self, model: Any, model_name: str, version: str = 'latest') -> bool:
        """
        Store trained model in S3.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            version: Model version
            
        Returns:
            True if successful, False otherwise
        """
        key = f"{self.base_path}/models/{model_name}/version={version}/model.pkl"
        
        return self.s3_manager.upload_data(model, key, format='pickle')
    
    def load_model(self, model_name: str, version: str = 'latest') -> Optional[Any]:
        """
        Load trained model from S3.
        
        Args:
            model_name: Name of the model
            version: Model version
            
        Returns:
            Loaded model or None if failed
        """
        key = f"{self.base_path}/models/{model_name}/version={version}/model.pkl"
        
        return self.s3_manager.download_data(key, format='pickle')
    
    def get_data_summary(self, data_type: str = 'raw') -> Dict[str, Any]:
        """
        Get summary of stored data.
        
        Args:
            data_type: Type of data ('raw', 'processed', 'models')
            
        Returns:
            Data summary dictionary
        """
        prefix = f"{self.base_path}/{data_type}-data/"
        objects = self.s3_manager.list_objects(prefix)
        
        summary = {
            'total_objects': len(objects),
            'total_size_bytes': sum(obj['size'] for obj in objects),
            'last_modified': max(obj['last_modified'] for obj in objects) if objects else None,
            'equipment_ids': set(),
            'date_range': {'start': None, 'end': None}
        }
        
        # Extract equipment IDs and date range
        for obj in objects:
            key = obj['key']
            # Extract equipment ID from key
            if 'equipment_' in key:
                equipment_id = key.split('equipment_')[1].split('_')[0]
                summary['equipment_ids'].add(equipment_id)
            
            # Extract date from key
            if 'year=' in key:
                try:
                    year = int(key.split('year=')[1].split('/')[0])
                    month = int(key.split('month=')[1].split('/')[0])
                    day = int(key.split('day=')[1].split('/')[0])
                    date = datetime(year, month, day)
                    
                    if summary['date_range']['start'] is None or date < summary['date_range']['start']:
                        summary['date_range']['start'] = date
                    if summary['date_range']['end'] is None or date > summary['date_range']['end']:
                        summary['date_range']['end'] = date
                except:
                    pass
        
        summary['equipment_ids'] = list(summary['equipment_ids'])
        summary['total_size_mb'] = summary['total_size_bytes'] / (1024 * 1024)
        
        return summary


class S3ConfigManager:
    """Manages configuration files in S3."""
    
    def __init__(self, s3_manager: S3DataManager):
        """
        Initialize S3 config manager.
        
        Args:
            s3_manager: S3DataManager instance
        """
        self.s3_manager = s3_manager
        self.config_path = 'predictive-maintenance/config'
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> bool:
        """
        Save configuration to S3.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the configuration
            
        Returns:
            True if successful, False otherwise
        """
        key = f"{self.config_path}/{config_name}.json"
        return self.s3_manager.upload_data(config, key, format='json')
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from S3.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary or None if failed
        """
        key = f"{self.config_path}/{config_name}.json"
        return self.s3_manager.download_data(key, format='json')
    
    def list_configs(self) -> List[str]:
        """
        List available configurations.
        
        Returns:
            List of configuration names
        """
        objects = self.s3_manager.list_objects(self.config_path)
        configs = []
        
        for obj in objects:
            key = obj['key']
            if key.endswith('.json'):
                config_name = key.split('/')[-1].replace('.json', '')
                configs.append(config_name)
        
        return configs


# Example usage
if __name__ == "__main__":
    # Initialize S3 manager
    s3_manager = S3DataManager(
        bucket_name='your-bucket-name',
        region_name='us-east-1'
    )
    
    # Create data lake
    data_lake = S3DataLake(s3_manager)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'equipment_id': 'MOTOR_001',
        'vibration': np.random.normal(0.5, 0.1, 100),
        'temperature': np.random.normal(50, 10, 100),
        'rpm': np.random.normal(1800, 100, 100)
    })
    
    # Store raw data
    success = data_lake.store_raw_data(sample_data, 'MOTOR_001')
    print(f"Raw data stored: {success}")
    
    # Get data summary
    summary = data_lake.get_data_summary('raw')
    print(f"Data summary: {summary}")
    
    # Create config manager
    config_manager = S3ConfigManager(s3_manager)
    
    # Save configuration
    config = {
        'model_params': {'n_estimators': 100, 'max_depth': 10},
        'data_params': {'sequence_length': 60, 'batch_size': 32}
    }
    success = config_manager.save_config(config, 'model_config')
    print(f"Config saved: {success}")
    
    # Load configuration
    loaded_config = config_manager.load_config('model_config')
    print(f"Loaded config: {loaded_config}")
