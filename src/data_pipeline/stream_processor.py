"""
Stream Processing Module

Handles real-time processing of sensor data streams using Apache Kafka or AWS Kinesis.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for stream processing."""
    batch_size: int = 100
    processing_interval: float = 1.0  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    enable_parallel_processing: bool = True
    max_workers: int = 4


class StreamProcessor:
    """Real-time stream processor for sensor data."""
    
    def __init__(self, config: ProcessingConfig, data_processor: Callable = None):
        """
        Initialize the stream processor.
        
        Args:
            config: Processing configuration
            data_processor: Function to process batches of data
        """
        self.config = config
        self.data_processor = data_processor or self._default_processor
        self.data_queue = Queue(maxsize=config.batch_size * 2)
        self.running = False
        self.processing_thread = None
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
    def start(self):
        """Start the stream processor."""
        if self.running:
            logger.warning("Stream processor is already running")
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info("Stream processor started")
    
    def stop(self):
        """Stop the stream processor."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("Stream processor stopped")
    
    def add_data(self, data: Dict[str, Any]):
        """
        Add data to the processing queue.
        
        Args:
            data: Sensor data dictionary
        """
        try:
            self.data_queue.put_nowait(data)
        except:
            logger.warning("Data queue is full, dropping data point")
    
    def _processing_loop(self):
        """Main processing loop."""
        batch = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # Try to get data from queue
                try:
                    data = self.data_queue.get(timeout=0.1)
                    batch.append(data)
                except Empty:
                    pass
                
                # Process batch if conditions are met
                current_time = time.time()
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_process_time >= self.config.processing_interval)
                )
                
                if should_process and batch:
                    self._process_batch(batch.copy())
                    batch.clear()
                    last_process_time = current_time
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(self.config.retry_delay)
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """
        Process a batch of data.
        
        Args:
            batch: List of data dictionaries
        """
        if not batch:
            return
        
        logger.info(f"Processing batch of {len(batch)} data points")
        
        try:
            if self.config.enable_parallel_processing:
                # Process in parallel
                future = self.executor.submit(self.data_processor, batch)
                # Don't wait for completion to avoid blocking
            else:
                # Process synchronously
                self.data_processor(batch)
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
    
    def _default_processor(self, batch: List[Dict[str, Any]]):
        """
        Default data processor.
        
        Args:
            batch: List of data dictionaries
        """
        # Convert to DataFrame for processing
        df = pd.DataFrame(batch)
        
        # Basic processing
        logger.info(f"Processing {len(df)} records")
        
        # Add processing timestamp
        df['processed_at'] = datetime.now(timezone.utc)
        
        # Calculate basic statistics
        if 'value' in df.columns:
            stats = {
                'mean': df['value'].mean(),
                'std': df['value'].std(),
                'min': df['value'].min(),
                'max': df['value'].max(),
                'count': len(df)
            }
            logger.info(f"Value statistics: {stats}")
        
        # Store or forward processed data
        self._store_processed_data(df)
    
    def _store_processed_data(self, df: pd.DataFrame):
        """
        Store processed data.
        
        Args:
            df: Processed DataFrame
        """
        # This would typically store to a database or forward to another service
        logger.info(f"Stored {len(df)} processed records")


class AnomalyDetector:
    """Real-time anomaly detection for sensor data."""
    
    def __init__(self, threshold: float = 2.0, window_size: int = 100):
        """
        Initialize the anomaly detector.
        
        Args:
            threshold: Z-score threshold for anomaly detection
            window_size: Rolling window size for statistics
        """
        self.threshold = threshold
        self.window_size = window_size
        self.data_buffer = {}
        self.statistics = {}
    
    def detect_anomalies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the data.
        
        Args:
            data: List of sensor data dictionaries
            
        Returns:
            List of data with anomaly flags
        """
        results = []
        
        for record in data:
            equipment_id = record.get('equipment_id')
            sensor_type = record.get('sensor_type')
            value = record.get('value')
            
            if not all([equipment_id, sensor_type, value is not None]):
                continue
            
            key = f"{equipment_id}_{sensor_type}"
            
            # Initialize buffer for new equipment/sensor
            if key not in self.data_buffer:
                self.data_buffer[key] = []
                self.statistics[key] = {'mean': 0, 'std': 1}
            
            # Add to buffer
            self.data_buffer[key].append(value)
            
            # Maintain window size
            if len(self.data_buffer[key]) > self.window_size:
                self.data_buffer[key].pop(0)
            
            # Calculate statistics
            if len(self.data_buffer[key]) >= 10:  # Minimum samples for statistics
                self.statistics[key] = {
                    'mean': np.mean(self.data_buffer[key]),
                    'std': np.std(self.data_buffer[key])
                }
            
            # Detect anomaly
            is_anomaly = False
            if self.statistics[key]['std'] > 0:
                z_score = abs(value - self.statistics[key]['mean']) / self.statistics[key]['std']
                is_anomaly = z_score > self.threshold
            
            # Add anomaly information to record
            record['is_anomaly'] = is_anomaly
            record['z_score'] = z_score if self.statistics[key]['std'] > 0 else 0
            record['anomaly_confidence'] = min(z_score / self.threshold, 1.0) if is_anomaly else 0.0
            
            results.append(record)
        
        return results


class HealthScoreCalculator:
    """Calculate equipment health scores in real-time."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize the health score calculator.
        
        Args:
            weights: Weights for different sensor types
        """
        self.weights = weights or {
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
    
    def calculate_health_score(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate health scores for equipment.
        
        Args:
            data: List of sensor data dictionaries
            
        Returns:
            List of data with health scores
        """
        # Group by equipment
        equipment_data = {}
        for record in data:
            equipment_id = record.get('equipment_id')
            if equipment_id not in equipment_data:
                equipment_data[equipment_id] = []
            equipment_data[equipment_id].append(record)
        
        results = []
        
        for equipment_id, records in equipment_data.items():
            # Calculate individual sensor health scores
            sensor_scores = {}
            
            for record in records:
                sensor_type = record.get('sensor_type')
                value = record.get('value')
                
                if sensor_type in self.weights and value is not None:
                    # Normalize value to 0-1 scale
                    normal_min, normal_max = self.normal_ranges.get(sensor_type, (0, 1))
                    normalized_value = np.clip((value - normal_min) / (normal_max - normal_min), 0, 1)
                    
                    # Health score (1 - normalized value, so higher is better)
                    health_score = 1 - normalized_value
                    sensor_scores[sensor_type] = health_score
            
            # Calculate weighted overall health score
            overall_health = 0
            total_weight = 0
            
            for sensor_type, score in sensor_scores.items():
                weight = self.weights.get(sensor_type, 0)
                overall_health += score * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_health /= total_weight
            else:
                overall_health = 0.5  # Default neutral score
            
            # Add health scores to all records for this equipment
            for record in records:
                record['sensor_health'] = sensor_scores.get(record.get('sensor_type'), 0.5)
                record['overall_health'] = overall_health
                record['health_status'] = self._get_health_status(overall_health)
                results.append(record)
        
        return results
    
    def _get_health_status(self, health_score: float) -> str:
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


class AlertManager:
    """Manage alerts based on processed data."""
    
    def __init__(self, alert_rules: Dict[str, Any] = None):
        """
        Initialize the alert manager.
        
        Args:
            alert_rules: Rules for generating alerts
        """
        self.alert_rules = alert_rules or {
            'anomaly_threshold': 0.8,
            'health_threshold': 0.3,
            'consecutive_anomalies': 3
        }
        self.alert_history = {}
    
    def check_alerts(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check for alert conditions.
        
        Args:
            data: List of processed data dictionaries
            
        Returns:
            List of alerts
        """
        alerts = []
        
        for record in data:
            equipment_id = record.get('equipment_id')
            is_anomaly = record.get('is_anomaly', False)
            health_score = record.get('overall_health', 0.5)
            anomaly_confidence = record.get('anomaly_confidence', 0)
            
            # Anomaly alert
            if is_anomaly and anomaly_confidence >= self.alert_rules['anomaly_threshold']:
                alert = {
                    'timestamp': datetime.now(timezone.utc),
                    'equipment_id': equipment_id,
                    'alert_type': 'anomaly',
                    'severity': 'high' if anomaly_confidence > 0.9 else 'medium',
                    'message': f"Anomaly detected in {equipment_id} with confidence {anomaly_confidence:.2f}",
                    'data': record
                }
                alerts.append(alert)
            
            # Health alert
            if health_score <= self.alert_rules['health_threshold']:
                alert = {
                    'timestamp': datetime.now(timezone.utc),
                    'equipment_id': equipment_id,
                    'alert_type': 'health',
                    'severity': 'critical' if health_score <= 0.2 else 'high',
                    'message': f"Poor health detected in {equipment_id} (score: {health_score:.2f})",
                    'data': record
                }
                alerts.append(alert)
        
        return alerts


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = [
        {
            'timestamp': datetime.now(timezone.utc),
            'equipment_id': 'MOTOR_001',
            'sensor_type': 'vibration',
            'value': 0.5,
            'unit': 'g',
            'location': 'bearing_1'
        },
        {
            'timestamp': datetime.now(timezone.utc),
            'equipment_id': 'MOTOR_001',
            'sensor_type': 'temperature',
            'value': 75.0,
            'unit': 'C',
            'location': 'housing'
        }
    ]
    
    # Initialize components
    config = ProcessingConfig(batch_size=10, processing_interval=2.0)
    anomaly_detector = AnomalyDetector(threshold=2.0)
    health_calculator = HealthScoreCalculator()
    alert_manager = AlertManager()
    
    # Create stream processor
    def process_data(batch):
        logger.info(f"Processing batch of {len(batch)} records")
        
        # Detect anomalies
        batch_with_anomalies = anomaly_detector.detect_anomalies(batch)
        
        # Calculate health scores
        batch_with_health = health_calculator.calculate_health_score(batch_with_anomalies)
        
        # Check for alerts
        alerts = alert_manager.check_alerts(batch_with_health)
        
        if alerts:
            logger.warning(f"Generated {len(alerts)} alerts")
            for alert in alerts:
                logger.warning(f"Alert: {alert['message']}")
        
        return batch_with_health
    
    processor = StreamProcessor(config, process_data)
    
    # Start processing
    processor.start()
    
    # Add sample data
    for data in sample_data:
        processor.add_data(data)
    
    # Let it process
    time.sleep(5)
    
    # Stop processing
    processor.stop()
