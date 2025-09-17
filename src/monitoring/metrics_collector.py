"""
Metrics Collector

Collects and aggregates metrics from various components of the predictive maintenance system.
"""

import time
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json
import boto3
from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Collects system-level metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.collection_interval = 60  # seconds
        self.running = False
        self.collection_thread = None
    
    def start_collection(self):
        """Start collecting metrics in a background thread."""
        self.running = True
        self.collection_thread = threading.Thread(target=self._collect_metrics_loop)
        self.collection_thread.start()
        logger.info("System metrics collection started")
    
    def stop_collection(self):
        """Stop collecting metrics."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join()
        logger.info("System metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total
            
            # Network metrics
            network = psutil.net_io_counters()
            bytes_sent = network.bytes_sent
            bytes_recv = network.bytes_recv
            
            # Process metrics
            process = psutil.Process()
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info().rss
            
            self.metrics = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'available_bytes': memory_available,
                    'total_bytes': memory_total
                },
                'disk': {
                    'percent': disk_percent,
                    'free_bytes': disk_free,
                    'total_bytes': disk_total
                },
                'network': {
                    'bytes_sent': bytes_sent,
                    'bytes_recv': bytes_recv
                },
                'process': {
                    'cpu_percent': process_cpu,
                    'memory_bytes': process_memory
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()


class ApplicationMetricsCollector:
    """Collects application-specific metrics."""
    
    def __init__(self):
        self.metrics = {
            'data_ingestion': {
                'records_processed': 0,
                'processing_time': 0.0,
                'errors': 0
            },
            'model_inference': {
                'predictions_made': 0,
                'inference_time': 0.0,
                'errors': 0
            },
            'anomaly_detection': {
                'anomalies_detected': 0,
                'false_positives': 0,
                'detection_time': 0.0
            },
            'health_scoring': {
                'scores_calculated': 0,
                'calculation_time': 0.0,
                'critical_alerts': 0
            }
        }
        self.lock = threading.Lock()
    
    def increment_counter(self, component: str, metric: str, value: int = 1):
        """Increment a counter metric."""
        with self.lock:
            if component in self.metrics and metric in self.metrics[component]:
                self.metrics[component][metric] += value
    
    def set_gauge(self, component: str, metric: str, value: float):
        """Set a gauge metric."""
        with self.lock:
            if component in self.metrics and metric in self.metrics[component]:
                self.metrics[component][metric] = value
    
    def record_timing(self, component: str, metric: str, duration: float):
        """Record a timing metric."""
        with self.lock:
            if component in self.metrics and metric in self.metrics[component]:
                self.metrics[component][metric] = duration
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self.lock:
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metrics': self.metrics.copy()
            }


class PrometheusMetricsExporter:
    """Exports metrics to Prometheus format."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.registry = CollectorRegistry()
        self.metrics = {}
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metrics."""
        # System metrics
        self.metrics['cpu_percent'] = Gauge('system_cpu_percent', 'CPU usage percentage', registry=self.registry)
        self.metrics['memory_percent'] = Gauge('system_memory_percent', 'Memory usage percentage', registry=self.registry)
        self.metrics['disk_percent'] = Gauge('system_disk_percent', 'Disk usage percentage', registry=self.registry)
        
        # Application metrics
        self.metrics['records_processed'] = Counter('app_records_processed_total', 'Total records processed', ['component'], registry=self.registry)
        self.metrics['processing_errors'] = Counter('app_processing_errors_total', 'Total processing errors', ['component'], registry=self.registry)
        self.metrics['processing_duration'] = Histogram('app_processing_duration_seconds', 'Processing duration', ['component'], registry=self.registry)
        self.metrics['anomalies_detected'] = Counter('app_anomalies_detected_total', 'Total anomalies detected', registry=self.registry)
        self.metrics['health_scores'] = Gauge('app_health_score', 'Current health score', ['equipment_id'], registry=self.registry)
    
    def update_system_metrics(self, system_metrics: Dict[str, Any]):
        """Update system metrics."""
        if 'cpu' in system_metrics:
            self.metrics['cpu_percent'].set(system_metrics['cpu']['percent'])
        
        if 'memory' in system_metrics:
            self.metrics['memory_percent'].set(system_metrics['memory']['percent'])
        
        if 'disk' in system_metrics:
            self.metrics['disk_percent'].set(system_metrics['disk']['percent'])
    
    def update_application_metrics(self, app_metrics: Dict[str, Any]):
        """Update application metrics."""
        metrics = app_metrics.get('metrics', {})
        
        for component, component_metrics in metrics.items():
            if 'records_processed' in component_metrics:
                self.metrics['records_processed'].labels(component=component).inc(component_metrics['records_processed'])
            
            if 'errors' in component_metrics:
                self.metrics['processing_errors'].labels(component=component).inc(component_metrics['errors'])
            
            if 'processing_time' in component_metrics:
                self.metrics['processing_duration'].labels(component=component).observe(component_metrics['processing_time'])
            
            if component == 'anomaly_detection' and 'anomalies_detected' in component_metrics:
                self.metrics['anomalies_detected'].inc(component_metrics['anomalies_detected'])
    
    def start_server(self):
        """Start Prometheus metrics server."""
        start_http_server(self.port, registry=self.registry)
        logger.info(f"Prometheus metrics server started on port {self.port}")


class CloudWatchMetricsPublisher:
    """Publishes metrics to AWS CloudWatch."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self.cloudwatch = boto3.client('cloudwatch', region_name=region_name)
        self.namespace = 'PredictiveMaintenance'
    
    def publish_metric(self, metric_name: str, value: float, unit: str = 'Count', 
                      dimensions: Dict[str, str] = None):
        """Publish a single metric to CloudWatch."""
        try:
            metric_data = {
                'MetricName': metric_name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.now(timezone.utc)
            }
            
            if dimensions:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in dimensions.items()
                ]
            
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
            
        except Exception as e:
            logger.error(f"Error publishing metric to CloudWatch: {e}")
    
    def publish_batch_metrics(self, metrics: List[Dict[str, Any]]):
        """Publish multiple metrics to CloudWatch."""
        try:
            metric_data = []
            
            for metric in metrics:
                metric_entry = {
                    'MetricName': metric['name'],
                    'Value': metric['value'],
                    'Unit': metric.get('unit', 'Count'),
                    'Timestamp': datetime.now(timezone.utc)
                }
                
                if 'dimensions' in metric:
                    metric_entry['Dimensions'] = [
                        {'Name': k, 'Value': v} for k, v in metric['dimensions'].items()
                    ]
                
                metric_data.append(metric_entry)
            
            # CloudWatch allows up to 20 metrics per batch
            for i in range(0, len(metric_data), 20):
                batch = metric_data[i:i+20]
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=batch
                )
            
        except Exception as e:
            logger.error(f"Error publishing batch metrics to CloudWatch: {e}")
    
    def publish_system_metrics(self, system_metrics: Dict[str, Any]):
        """Publish system metrics to CloudWatch."""
        try:
            metrics = []
            
            if 'cpu' in system_metrics:
                metrics.append({
                    'name': 'CPUUtilization',
                    'value': system_metrics['cpu']['percent'],
                    'unit': 'Percent'
                })
            
            if 'memory' in system_metrics:
                metrics.append({
                    'name': 'MemoryUtilization',
                    'value': system_metrics['memory']['percent'],
                    'unit': 'Percent'
                })
            
            if 'disk' in system_metrics:
                metrics.append({
                    'name': 'DiskUtilization',
                    'value': system_metrics['disk']['percent'],
                    'unit': 'Percent'
                })
            
            if metrics:
                self.publish_batch_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error publishing system metrics: {e}")
    
    def publish_application_metrics(self, app_metrics: Dict[str, Any]):
        """Publish application metrics to CloudWatch."""
        try:
            metrics = []
            component_metrics = app_metrics.get('metrics', {})
            
            for component, component_data in component_metrics.items():
                if 'records_processed' in component_data:
                    metrics.append({
                        'name': 'RecordsProcessed',
                        'value': component_data['records_processed'],
                        'unit': 'Count',
                        'dimensions': {'Component': component}
                    })
                
                if 'errors' in component_data:
                    metrics.append({
                        'name': 'ProcessingErrors',
                        'value': component_data['errors'],
                        'unit': 'Count',
                        'dimensions': {'Component': component}
                    })
                
                if 'processing_time' in component_data:
                    metrics.append({
                        'name': 'ProcessingTime',
                        'value': component_data['processing_time'],
                        'unit': 'Seconds',
                        'dimensions': {'Component': component}
                    })
            
            if metrics:
                self.publish_batch_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error publishing application metrics: {e}")


class MetricsAggregator:
    """Aggregates metrics from multiple sources."""
    
    def __init__(self, cloudwatch_publisher: CloudWatchMetricsPublisher = None):
        self.system_collector = SystemMetricsCollector()
        self.app_collector = ApplicationMetricsCollector()
        self.prometheus_exporter = PrometheusMetricsExporter()
        self.cloudwatch_publisher = cloudwatch_publisher
        self.aggregation_interval = 300  # 5 minutes
        self.running = False
        self.aggregation_thread = None
    
    def start_aggregation(self):
        """Start metrics aggregation."""
        self.running = True
        self.system_collector.start_collection()
        self.prometheus_exporter.start_server()
        
        self.aggregation_thread = threading.Thread(target=self._aggregation_loop)
        self.aggregation_thread.start()
        
        logger.info("Metrics aggregation started")
    
    def stop_aggregation(self):
        """Stop metrics aggregation."""
        self.running = False
        self.system_collector.stop_collection()
        
        if self.aggregation_thread:
            self.aggregation_thread.join()
        
        logger.info("Metrics aggregation stopped")
    
    def _aggregation_loop(self):
        """Main aggregation loop."""
        while self.running:
            try:
                # Collect metrics
                system_metrics = self.system_collector.get_metrics()
                app_metrics = self.app_collector.get_metrics()
                
                # Update Prometheus exporter
                self.prometheus_exporter.update_system_metrics(system_metrics)
                self.prometheus_exporter.update_application_metrics(app_metrics)
                
                # Publish to CloudWatch if available
                if self.cloudwatch_publisher:
                    self.cloudwatch_publisher.publish_system_metrics(system_metrics)
                    self.cloudwatch_publisher.publish_application_metrics(app_metrics)
                
                time.sleep(self.aggregation_interval)
                
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                time.sleep(self.aggregation_interval)
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all sources."""
        return {
            'system': self.system_collector.get_metrics(),
            'application': self.app_collector.get_metrics(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Example usage
if __name__ == "__main__":
    # Initialize CloudWatch publisher
    cloudwatch_publisher = CloudWatchMetricsPublisher()
    
    # Initialize metrics aggregator
    aggregator = MetricsAggregator(cloudwatch_publisher)
    
    # Start aggregation
    aggregator.start_aggregation()
    
    # Simulate some application activity
    app_collector = aggregator.app_collector
    
    # Simulate data processing
    for i in range(10):
        app_collector.increment_counter('data_ingestion', 'records_processed', 100)
        app_collector.record_timing('data_ingestion', 'processing_time', 1.5)
        
        app_collector.increment_counter('model_inference', 'predictions_made', 50)
        app_collector.record_timing('model_inference', 'inference_time', 0.8)
        
        app_collector.increment_counter('anomaly_detection', 'anomalies_detected', 5)
        app_collector.record_timing('anomaly_detection', 'detection_time', 0.3)
        
        time.sleep(1)
    
    # Get aggregated metrics
    metrics = aggregator.get_aggregated_metrics()
    print(f"Aggregated metrics: {json.dumps(metrics, indent=2)}")
    
    # Stop aggregation
    aggregator.stop_aggregation()
