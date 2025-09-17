"""
Kinesis Integration Module

Handles real-time data streaming using Amazon Kinesis for the predictive maintenance system.
"""

import boto3
import json
import time
import base64
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KinesisDataStream:
    """Manages Kinesis data stream operations."""
    
    def __init__(self, stream_name: str, region_name: str = 'us-east-1',
                 aws_access_key_id: str = None, aws_secret_access_key: str = None):
        """
        Initialize Kinesis data stream.
        
        Args:
            stream_name: Name of the Kinesis stream
            region_name: AWS region name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
        """
        self.stream_name = stream_name
        self.region_name = region_name
        
        # Initialize Kinesis client
        try:
            if aws_access_key_id and aws_secret_access_key:
                self.kinesis_client = boto3.client(
                    'kinesis',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key
                )
            else:
                # Use default credentials
                self.kinesis_client = boto3.client('kinesis', region_name=region_name)
            
            # Test connection
            self.kinesis_client.describe_stream(StreamName=stream_name)
            logger.info(f"Successfully connected to Kinesis stream: {stream_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to Kinesis stream {stream_name}: {e}")
            raise
    
    def create_stream(self, shard_count: int = 1) -> bool:
        """
        Create a Kinesis stream.
        
        Args:
            shard_count: Number of shards for the stream
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.kinesis_client.create_stream(
                StreamName=self.stream_name,
                ShardCount=shard_count
            )
            
            # Wait for stream to become active
            self._wait_for_stream_active()
            
            logger.info(f"Kinesis stream {self.stream_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Kinesis stream: {e}")
            return False
    
    def _wait_for_stream_active(self, timeout: int = 300):
        """Wait for stream to become active."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.kinesis_client.describe_stream(StreamName=self.stream_name)
                status = response['StreamDescription']['StreamStatus']
                
                if status == 'ACTIVE':
                    return
                elif status == 'DELETING':
                    raise Exception(f"Stream {self.stream_name} is being deleted")
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error checking stream status: {e}")
                raise
        
        raise Exception(f"Stream {self.stream_name} did not become active within {timeout} seconds")
    
    def put_record(self, data: Dict[str, Any], partition_key: str = None) -> bool:
        """
        Put a single record to the stream.
        
        Args:
            data: Data to send
            partition_key: Partition key for the record
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Convert to JSON
            json_data = json.dumps(data, default=str)
            
            # Use equipment_id as partition key if available
            if partition_key is None:
                partition_key = data.get('equipment_id', 'default')
            
            # Put record
            response = self.kinesis_client.put_record(
                StreamName=self.stream_name,
                Data=json_data,
                PartitionKey=partition_key
            )
            
            logger.debug(f"Record sent to Kinesis: {response['SequenceNumber']}")
            return True
            
        except Exception as e:
            logger.error(f"Error putting record to Kinesis: {e}")
            return False
    
    def put_records(self, records: List[Dict[str, Any]], partition_key: str = None) -> bool:
        """
        Put multiple records to the stream.
        
        Args:
            records: List of data records
            partition_key: Partition key for the records
            
        Returns:
            True if successful, False otherwise
        """
        try:
            kinesis_records = []
            
            for record in records:
                # Add timestamp if not present
                if 'timestamp' not in record:
                    record['timestamp'] = datetime.now(timezone.utc).isoformat()
                
                # Convert to JSON
                json_data = json.dumps(record, default=str)
                
                # Use equipment_id as partition key if available
                record_partition_key = partition_key or record.get('equipment_id', 'default')
                
                kinesis_records.append({
                    'Data': json_data,
                    'PartitionKey': record_partition_key
                })
            
            # Put records in batches
            batch_size = 500  # Kinesis limit
            for i in range(0, len(kinesis_records), batch_size):
                batch = kinesis_records[i:i + batch_size]
                
                response = self.kinesis_client.put_records(
                    StreamName=self.stream_name,
                    Records=batch
                )
                
                # Check for failed records
                failed_count = response['FailedRecordCount']
                if failed_count > 0:
                    logger.warning(f"{failed_count} records failed to be sent")
                
                logger.debug(f"Batch {i//batch_size + 1} sent to Kinesis")
            
            return True
            
        except Exception as e:
            logger.error(f"Error putting records to Kinesis: {e}")
            return False
    
    def get_records(self, shard_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get records from a specific shard.
        
        Args:
            shard_id: Shard ID to read from
            limit: Maximum number of records to return
            
        Returns:
            List of records
        """
        try:
            # Get shard iterator
            response = self.kinesis_client.get_shard_iterator(
                StreamName=self.stream_name,
                ShardId=shard_id,
                ShardIteratorType='LATEST'
            )
            
            shard_iterator = response['ShardIterator']
            records = []
            
            # Get records
            while len(records) < limit:
                response = self.kinesis_client.get_records(
                    ShardIterator=shard_iterator,
                    Limit=min(100, limit - len(records))
                )
                
                for record in response['Records']:
                    try:
                        data = json.loads(record['Data'].decode('utf-8'))
                        records.append(data)
                    except Exception as e:
                        logger.error(f"Error parsing record: {e}")
                        continue
                
                shard_iterator = response.get('NextShardIterator')
                if not shard_iterator:
                    break
                
                time.sleep(0.1)  # Rate limiting
            
            return records
            
        except Exception as e:
            logger.error(f"Error getting records from Kinesis: {e}")
            return []
    
    def list_shards(self) -> List[Dict[str, Any]]:
        """
        List all shards in the stream.
        
        Returns:
            List of shard information
        """
        try:
            response = self.kinesis_client.describe_stream(StreamName=self.stream_name)
            shards = []
            
            for shard in response['StreamDescription']['Shards']:
                shards.append({
                    'shard_id': shard['ShardId'],
                    'hash_key_range': shard['HashKeyRange'],
                    'sequence_number_range': shard['SequenceNumberRange']
                })
            
            return shards
            
        except Exception as e:
            logger.error(f"Error listing shards: {e}")
            return []
    
    def delete_stream(self) -> bool:
        """
        Delete the Kinesis stream.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.kinesis_client.delete_stream(StreamName=self.stream_name)
            logger.info(f"Kinesis stream {self.stream_name} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting Kinesis stream: {e}")
            return False


class KinesisConsumer:
    """Consumes records from Kinesis stream."""
    
    def __init__(self, stream_name: str, region_name: str = 'us-east-1'):
        """
        Initialize Kinesis consumer.
        
        Args:
            stream_name: Name of the Kinesis stream
            region_name: AWS region name
        """
        self.stream_name = stream_name
        self.region_name = region_name
        self.kinesis_client = boto3.client('kinesis', region_name=region_name)
        self.running = False
        self.consumer_threads = []
    
    def start_consuming(self, shard_ids: List[str] = None, 
                       callback_func=None, max_records: int = 100):
        """
        Start consuming records from the stream.
        
        Args:
            shard_ids: List of shard IDs to consume from (None for all)
            callback_func: Function to call for each record
            max_records: Maximum number of records per batch
        """
        if shard_ids is None:
            shard_ids = self._get_all_shard_ids()
        
        self.running = True
        
        # Start consumer threads for each shard
        for shard_id in shard_ids:
            thread = threading.Thread(
                target=self._consume_shard,
                args=(shard_id, callback_func, max_records)
            )
            thread.start()
            self.consumer_threads.append(thread)
        
        logger.info(f"Started consuming from {len(shard_ids)} shards")
    
    def stop_consuming(self):
        """Stop consuming records."""
        self.running = False
        
        # Wait for all threads to finish
        for thread in self.consumer_threads:
            thread.join()
        
        self.consumer_threads = []
        logger.info("Stopped consuming from Kinesis stream")
    
    def _get_all_shard_ids(self) -> List[str]:
        """Get all shard IDs from the stream."""
        try:
            response = self.kinesis_client.describe_stream(StreamName=self.stream_name)
            shard_ids = []
            
            for shard in response['StreamDescription']['Shards']:
                shard_ids.append(shard['ShardId'])
            
            return shard_ids
            
        except Exception as e:
            logger.error(f"Error getting shard IDs: {e}")
            return []
    
    def _consume_shard(self, shard_id: str, callback_func, max_records: int):
        """Consume records from a specific shard."""
        try:
            # Get shard iterator
            response = self.kinesis_client.get_shard_iterator(
                StreamName=self.stream_name,
                ShardId=shard_id,
                ShardIteratorType='LATEST'
            )
            
            shard_iterator = response['ShardIterator']
            
            while self.running:
                # Get records
                response = self.kinesis_client.get_records(
                    ShardIterator=shard_iterator,
                    Limit=max_records
                )
                
                # Process records
                for record in response['Records']:
                    try:
                        data = json.loads(record['Data'].decode('utf-8'))
                        
                        if callback_func:
                            callback_func(data)
                        else:
                            logger.info(f"Received record: {data}")
                            
                    except Exception as e:
                        logger.error(f"Error processing record: {e}")
                        continue
                
                # Update shard iterator
                shard_iterator = response.get('NextShardIterator')
                if not shard_iterator:
                    break
                
                time.sleep(0.1)  # Rate limiting
                
        except Exception as e:
            logger.error(f"Error consuming from shard {shard_id}: {e}")


class KinesisProducer:
    """Produces records to Kinesis stream."""
    
    def __init__(self, stream_name: str, region_name: str = 'us-east-1'):
        """
        Initialize Kinesis producer.
        
        Args:
            stream_name: Name of the Kinesis stream
            region_name: AWS region name
        """
        self.stream_name = stream_name
        self.region_name = region_name
        self.kinesis_client = boto3.client('kinesis', region_name=region_name)
        self.record_buffer = []
        self.buffer_size = 100
        self.flush_interval = 5  # seconds
        self.last_flush = time.time()
    
    def put_record(self, data: Dict[str, Any], partition_key: str = None):
        """
        Put a record to the stream (buffered).
        
        Args:
            data: Data to send
            partition_key: Partition key for the record
        """
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Use equipment_id as partition key if available
        if partition_key is None:
            partition_key = data.get('equipment_id', 'default')
        
        # Add to buffer
        self.record_buffer.append({
            'Data': json.dumps(data, default=str),
            'PartitionKey': partition_key
        })
        
        # Flush if buffer is full or enough time has passed
        if (len(self.record_buffer) >= self.buffer_size or 
            time.time() - self.last_flush > self.flush_interval):
            self.flush()
    
    def flush(self):
        """Flush the record buffer to Kinesis."""
        if not self.record_buffer:
            return
        
        try:
            # Put records in batches
            batch_size = 500  # Kinesis limit
            for i in range(0, len(self.record_buffer), batch_size):
                batch = self.record_buffer[i:i + batch_size]
                
                response = self.kinesis_client.put_records(
                    StreamName=self.stream_name,
                    Records=batch
                )
                
                # Check for failed records
                failed_count = response['FailedRecordCount']
                if failed_count > 0:
                    logger.warning(f"{failed_count} records failed to be sent")
                
                logger.debug(f"Flushed {len(batch)} records to Kinesis")
            
            # Clear buffer
            self.record_buffer = []
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Error flushing records to Kinesis: {e}")


class KinesisAnalytics:
    """Provides analytics and monitoring for Kinesis streams."""
    
    def __init__(self, stream_name: str, region_name: str = 'us-east-1'):
        """
        Initialize Kinesis analytics.
        
        Args:
            stream_name: Name of the Kinesis stream
            region_name: AWS region name
        """
        self.stream_name = stream_name
        self.region_name = region_name
        self.kinesis_client = boto3.client('kinesis', region_name=region_name)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=region_name)
    
    def get_stream_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get CloudWatch metrics for the stream.
        
        Args:
            start_time: Start time for metrics
            end_time: End time for metrics
            
        Returns:
            Stream metrics
        """
        try:
            metrics = {}
            
            # Get incoming records metric
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/Kinesis',
                MetricName='IncomingRecords',
                Dimensions=[
                    {
                        'Name': 'StreamName',
                        'Value': self.stream_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=['Sum', 'Average', 'Maximum']
            )
            
            metrics['incoming_records'] = response['Datapoints']
            
            # Get incoming bytes metric
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace='AWS/Kinesis',
                MetricName='IncomingBytes',
                Dimensions=[
                    {
                        'Name': 'StreamName',
                        'Value': self.stream_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum', 'Average', 'Maximum']
            )
            
            metrics['incoming_bytes'] = response['Datapoints']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting stream metrics: {e}")
            return {}
    
    def get_stream_info(self) -> Dict[str, Any]:
        """
        Get basic information about the stream.
        
        Returns:
            Stream information
        """
        try:
            response = self.kinesis_client.describe_stream(StreamName=self.stream_name)
            stream_description = response['StreamDescription']
            
            return {
                'stream_name': stream_description['StreamName'],
                'stream_status': stream_description['StreamStatus'],
                'stream_arn': stream_description['StreamARN'],
                'shard_count': len(stream_description['Shards']),
                'retention_period': stream_description['RetentionPeriodHours'],
                'creation_time': stream_description['StreamCreationTimestamp']
            }
            
        except Exception as e:
            logger.error(f"Error getting stream info: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Initialize Kinesis stream
    stream = KinesisDataStream('predictive-maintenance-stream')
    
    # Create stream if it doesn't exist
    stream.create_stream(shard_count=2)
    
    # Send sample data
    sample_data = {
        'equipment_id': 'MOTOR_001',
        'sensor_type': 'vibration',
        'value': 0.5,
        'unit': 'g',
        'location': 'bearing_1',
        'metadata': {
            'temperature': 75.0,
            'rpm': 1800,
            'load': 85.0
        }
    }
    
    # Put single record
    success = stream.put_record(sample_data)
    print(f"Single record sent: {success}")
    
    # Put multiple records
    records = [sample_data.copy() for _ in range(10)]
    for i, record in enumerate(records):
        record['equipment_id'] = f'MOTOR_{i+1:03d}'
        record['value'] = 0.3 + i * 0.1
    
    success = stream.put_records(records)
    print(f"Multiple records sent: {success}")
    
    # List shards
    shards = stream.list_shards()
    print(f"Shards: {shards}")
    
    # Consumer example
    def process_record(data):
        print(f"Processed record: {data['equipment_id']} - {data['value']}")
    
    consumer = KinesisConsumer('predictive-maintenance-stream')
    consumer.start_consuming(callback_func=process_record)
    
    # Let it run for a bit
    time.sleep(10)
    
    # Stop consuming
    consumer.stop_consuming()
    
    # Producer example
    producer = KinesisProducer('predictive-maintenance-stream')
    
    for i in range(20):
        data = {
            'equipment_id': f'PUMP_{i+1:03d}',
            'sensor_type': 'pressure',
            'value': 50 + i * 2,
            'unit': 'psi'
        }
        producer.put_record(data)
    
    # Flush remaining records
    producer.flush()
    
    # Analytics
    analytics = KinesisAnalytics('predictive-maintenance-stream')
    stream_info = analytics.get_stream_info()
    print(f"Stream info: {stream_info}")
    
    # Get metrics for the last hour
    end_time = datetime.now(timezone.utc)
    start_time = end_time.replace(hour=end_time.hour - 1)
    
    metrics = analytics.get_stream_metrics(start_time, end_time)
    print(f"Stream metrics: {metrics}")
