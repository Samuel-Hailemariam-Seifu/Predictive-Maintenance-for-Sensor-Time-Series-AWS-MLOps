"""
Data Ingestion Module

Handles real-time and batch data ingestion from various sources including
Kinesis streams, S3 buckets, and direct sensor connections.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Iterator
import boto3
import pandas as pd
from kinesis import KinesisConsumer
from kinesis import KinesisProducer
import asyncio
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Enumeration of supported data sources."""
    KINESIS = "kinesis"
    S3 = "s3"
    KAFKA = "kafka"
    DIRECT = "direct"


@dataclass
class SensorData:
    """Data class for sensor readings."""
    timestamp: datetime
    equipment_id: str
    sensor_type: str
    value: float
    unit: str
    location: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equipment_id": self.equipment_id,
            "sensor_type": self.sensor_type,
            "value": self.value,
            "unit": self.unit,
            "location": self.location,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorData':
        """Create SensorData from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00')),
            equipment_id=data["equipment_id"],
            sensor_type=data["sensor_type"],
            value=data["value"],
            unit=data["unit"],
            location=data["location"],
            metadata=data.get("metadata", {})
        )


class DataIngestionService:
    """Service for ingesting sensor data from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data ingestion service.
        
        Args:
            config: Configuration dictionary containing AWS credentials and settings
        """
        self.config = config
        self.aws_session = boto3.Session(
            aws_access_key_id=config.get('aws_access_key_id'),
            aws_secret_access_key=config.get('aws_secret_access_key'),
            region_name=config.get('aws_region', 'us-east-1')
        )
        self.s3_client = self.aws_session.client('s3')
        self.kinesis_client = self.aws_session.client('kinesis')
        
    def ingest_from_kinesis(self, stream_name: str, shard_id: str = None) -> Iterator[SensorData]:
        """
        Ingest data from Kinesis stream.
        
        Args:
            stream_name: Name of the Kinesis stream
            shard_id: Specific shard ID to read from (optional)
            
        Yields:
            SensorData objects from the stream
        """
        try:
            consumer = KinesisConsumer(
                stream_name=stream_name,
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            
            for record in consumer:
                try:
                    # Parse the record data
                    data = json.loads(record['Data'].decode('utf-8'))
                    sensor_data = SensorData.from_dict(data)
                    yield sensor_data
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Error parsing Kinesis record: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error ingesting from Kinesis: {e}")
            raise
    
    def ingest_from_s3(self, bucket: str, prefix: str) -> Iterator[SensorData]:
        """
        Ingest data from S3 bucket.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 object prefix
            
        Yields:
            SensorData objects from S3 files
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                for obj in page.get('Contents', []):
                    try:
                        # Download and parse the object
                        response = self.s3_client.get_object(Bucket=bucket, Key=obj['Key'])
                        content = response['Body'].read().decode('utf-8')
                        
                        # Parse JSON lines format
                        for line in content.strip().split('\n'):
                            if line:
                                data = json.loads(line)
                                sensor_data = SensorData.from_dict(data)
                                yield sensor_data
                                
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.error(f"Error parsing S3 object {obj['Key']}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error ingesting from S3: {e}")
            raise
    
    def ingest_batch_data(self, file_path: str) -> List[SensorData]:
        """
        Ingest data from a local file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of SensorData objects
        """
        sensor_data_list = []
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            sensor_data = SensorData.from_dict(data)
                            sensor_data_list.append(sensor_data)
                            
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    sensor_data = SensorData(
                        timestamp=pd.to_datetime(row['timestamp']),
                        equipment_id=row['equipment_id'],
                        sensor_type=row['sensor_type'],
                        value=row['value'],
                        unit=row['unit'],
                        location=row['location'],
                        metadata=json.loads(row.get('metadata', '{}'))
                    )
                    sensor_data_list.append(sensor_data)
                    
        except Exception as e:
            logger.error(f"Error ingesting batch data from {file_path}: {e}")
            raise
            
        return sensor_data_list
    
    def publish_to_kinesis(self, stream_name: str, data: List[SensorData]) -> None:
        """
        Publish sensor data to Kinesis stream.
        
        Args:
            stream_name: Name of the Kinesis stream
            data: List of SensorData objects to publish
        """
        try:
            producer = KinesisProducer(
                stream_name=stream_name,
                region_name=self.config.get('aws_region', 'us-east-1')
            )
            
            for sensor_data in data:
                record = {
                    'Data': json.dumps(sensor_data.to_dict()),
                    'PartitionKey': sensor_data.equipment_id
                }
                producer.put_record(record)
                
        except Exception as e:
            logger.error(f"Error publishing to Kinesis: {e}")
            raise
    
    def store_to_s3(self, bucket: str, key: str, data: List[SensorData]) -> None:
        """
        Store sensor data to S3 bucket.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            data: List of SensorData objects to store
        """
        try:
            # Convert to JSON lines format
            json_lines = [json.dumps(sensor_data.to_dict()) for sensor_data in data]
            content = '\n'.join(json_lines)
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode('utf-8'),
                ContentType='application/json'
            )
            
        except Exception as e:
            logger.error(f"Error storing to S3: {e}")
            raise


class RealTimeDataProcessor:
    """Real-time data processor for streaming sensor data."""
    
    def __init__(self, ingestion_service: DataIngestionService):
        """
        Initialize the real-time data processor.
        
        Args:
            ingestion_service: DataIngestionService instance
        """
        self.ingestion_service = ingestion_service
        self.running = False
    
    async def start_processing(self, stream_name: str, callback_func=None):
        """
        Start real-time processing of sensor data.
        
        Args:
            stream_name: Name of the Kinesis stream to process
            callback_func: Optional callback function to process each data point
        """
        self.running = True
        logger.info(f"Starting real-time processing from stream: {stream_name}")
        
        try:
            for sensor_data in self.ingestion_service.ingest_from_kinesis(stream_name):
                if not self.running:
                    break
                    
                # Process the data
                if callback_func:
                    await callback_func(sensor_data)
                else:
                    await self._default_processing(sensor_data)
                    
        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
            raise
        finally:
            self.running = False
    
    async def _default_processing(self, sensor_data: SensorData):
        """
        Default processing for sensor data.
        
        Args:
            sensor_data: SensorData object to process
        """
        logger.info(f"Processing data: {sensor_data.equipment_id} - {sensor_data.sensor_type}")
        # Add your processing logic here
        pass
    
    def stop_processing(self):
        """Stop the real-time processing."""
        self.running = False
        logger.info("Stopping real-time processing")


# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = {
        'aws_access_key_id': 'your_access_key',
        'aws_secret_access_key': 'your_secret_key',
        'aws_region': 'us-east-1'
    }
    
    # Initialize services
    ingestion_service = DataIngestionService(config)
    
    # Example: Process data from Kinesis
    async def process_sensor_data(sensor_data: SensorData):
        print(f"Received: {sensor_data.equipment_id} - {sensor_data.value}")
    
    # Start real-time processing
    processor = RealTimeDataProcessor(ingestion_service)
    # asyncio.run(processor.start_processing("sensor-data-stream", process_sensor_data))
