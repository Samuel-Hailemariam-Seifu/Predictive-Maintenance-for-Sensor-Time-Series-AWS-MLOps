"""
Alerting System

Comprehensive alerting system for the predictive maintenance platform with
multiple notification channels and intelligent alert management.
"""

import json
import smtplib
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass
from enum import Enum
import threading
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = None
    acknowledged_by: str = None
    acknowledged_at: datetime = None
    resolved_at: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'metadata': self.metadata or {},
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class NotificationChannel:
    """Base class for notification channels."""
    
    def send(self, alert: Alert) -> bool:
        """Send alert notification."""
        raise NotImplementedError


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str,
                 from_email: str, to_emails: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
    
    def send(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = f"""
Alert Details:
- Title: {alert.title}
- Description: {alert.description}
- Severity: {alert.severity.value.upper()}
- Source: {alert.source}
- Timestamp: {alert.timestamp.isoformat()}
- Status: {alert.status.value}

Metadata:
{json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}

This is an automated alert from the Predictive Maintenance System.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str, channel: str = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffaa00",
                AlertSeverity.ERROR: "#ff6600",
                AlertSeverity.CRITICAL: "#ff0000"
            }
            
            # Create Slack message
            message = {
                "text": f"🚨 *{alert.title}*",
                "attachments": [
                    {
                        "color": color_map[alert.severity],
                        "fields": [
                            {
                                "title": "Description",
                                "value": alert.description,
                                "short": False
                            },
                            {
                                "title": "Severity",
                                "value": alert.severity.value.upper(),
                                "short": True
                            },
                            {
                                "title": "Source",
                                "value": alert.source,
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": alert.timestamp.isoformat(),
                                "short": True
                            },
                            {
                                "title": "Status",
                                "value": alert.status.value,
                                "short": True
                            }
                        ],
                        "footer": "Predictive Maintenance System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            if self.channel:
                message["channel"] = self.channel
            
            # Send to Slack
            response = requests.post(self.webhook_url, json=message)
            response.raise_for_status()
            
            logger.info(f"Slack alert sent: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False


class SNSNotificationChannel(NotificationChannel):
    """AWS SNS notification channel."""
    
    def __init__(self, topic_arn: str, region_name: str = 'us-east-1'):
        self.topic_arn = topic_arn
        self.sns = boto3.client('sns', region_name=region_name)
    
    def send(self, alert: Alert) -> bool:
        """Send SNS notification."""
        try:
            # Create SNS message
            message = {
                "alert": alert.to_dict()
            }
            
            # Send to SNS
            self.sns.publish(
                TopicArn=self.topic_arn,
                Message=json.dumps(message),
                Subject=f"[{alert.severity.value.upper()}] {alert.title}"
            )
            
            logger.info(f"SNS alert sent: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SNS alert: {e}")
            return False


class AlertRule:
    """Alert rule definition."""
    
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                 severity: AlertSeverity, description: str, source: str):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.description = description
        self.source = source
        self.enabled = True
        self.cooldown_seconds = 300  # 5 minutes
        self.last_triggered = None
    
    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if rule should trigger."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_triggered:
            time_since_last = (datetime.now(timezone.utc) - self.last_triggered).total_seconds()
            if time_since_last < self.cooldown_seconds:
                return False
        
        # Check condition
        try:
            return self.condition(data)
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")
            return False
    
    def trigger(self):
        """Mark rule as triggered."""
        self.last_triggered = datetime.now(timezone.utc)


class AlertManager:
    """Main alert management system."""
    
    def __init__(self):
        self.alerts = {}
        self.rules = {}
        self.channels = []
        self.running = False
        self.monitoring_thread = None
        self.alert_queue = queue.Queue()
        self.processing_thread = None
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a notification channel."""
        self.channels.append(channel)
        logger.info(f"Added notification channel: {type(channel).__name__}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def start_monitoring(self):
        """Start alert monitoring."""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.processing_thread = threading.Thread(target=self._processing_loop)
        
        self.monitoring_thread.start()
        self.processing_thread.start()
        
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        if self.processing_thread:
            self.processing_thread.join()
        
        logger.info("Alert monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # This would typically monitor various data sources
                # For now, we'll simulate by checking system metrics
                self._check_system_health()
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _processing_loop(self):
        """Process alerts from the queue."""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._process_alert(alert)
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def _check_system_health(self):
        """Check system health and trigger alerts."""
        # This is a simplified example - in practice, you'd check actual metrics
        import psutil
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            self._create_alert(
                title="High CPU Usage",
                description=f"CPU usage is {cpu_percent}%",
                severity=AlertSeverity.WARNING,
                source="system_monitor",
                metadata={"cpu_percent": cpu_percent}
            )
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            self._create_alert(
                title="High Memory Usage",
                description=f"Memory usage is {memory.percent}%",
                severity=AlertSeverity.CRITICAL,
                source="system_monitor",
                metadata={"memory_percent": memory.percent}
            )
    
    def _create_alert(self, title: str, description: str, severity: AlertSeverity,
                     source: str, metadata: Dict[str, Any] = None) -> Alert:
        """Create a new alert."""
        alert_id = f"{source}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            source=source,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata
        )
        
        # Add to queue for processing
        self.alert_queue.put(alert)
        
        return alert
    
    def _process_alert(self, alert: Alert):
        """Process an alert."""
        try:
            # Store alert
            self.alerts[alert.id] = alert
            
            # Send notifications
            for channel in self.channels:
                try:
                    channel.send(alert)
                except Exception as e:
                    logger.error(f"Error sending alert via {type(channel).__name__}: {e}")
            
            logger.info(f"Processed alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)
                
                logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        try:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now(timezone.utc)
                
                logger.info(f"Alert {alert_id} resolved")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [alert for alert in self.alerts.values() if alert.status == AlertStatus.ACTIVE]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity."""
        return [alert for alert in self.alerts.values() if alert.severity == severity]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len(self.get_alerts_by_severity(severity))
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'severity_counts': severity_counts,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class PredictiveMaintenanceAlerts:
    """Specialized alerts for predictive maintenance."""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self._setup_alert_rules()
    
    def _setup_alert_rules(self):
        """Set up predictive maintenance alert rules."""
        
        # Anomaly detection alerts
        def anomaly_detected(data):
            return data.get('is_anomaly', False) and data.get('anomaly_confidence', 0) > 0.8
        
        anomaly_rule = AlertRule(
            name="anomaly_detected",
            condition=anomaly_detected,
            severity=AlertSeverity.WARNING,
            description="Anomaly detected in sensor data",
            source="anomaly_detector"
        )
        self.alert_manager.add_alert_rule(anomaly_rule)
        
        # Critical health alerts
        def critical_health(data):
            return data.get('health_score', 1) < 0.2
        
        health_rule = AlertRule(
            name="critical_health",
            condition=critical_health,
            severity=AlertSeverity.CRITICAL,
            description="Equipment health is critical",
            source="health_scorer"
        )
        self.alert_manager.add_alert_rule(health_rule)
        
        # Failure prediction alerts
        def failure_predicted(data):
            return data.get('failure_probability', 0) > 0.8
        
        failure_rule = AlertRule(
            name="failure_predicted",
            condition=failure_predicted,
            severity=AlertSeverity.ERROR,
            description="Equipment failure predicted",
            source="failure_predictor"
        )
        self.alert_manager.add_alert_rule(failure_rule)
        
        # Model performance alerts
        def model_performance_degraded(data):
            return data.get('model_accuracy', 1) < 0.7
        
        model_rule = AlertRule(
            name="model_performance_degraded",
            condition=model_performance_degraded,
            severity=AlertSeverity.WARNING,
            description="Model performance has degraded",
            source="model_monitor"
        )
        self.alert_manager.add_alert_rule(model_rule)
    
    def check_sensor_data(self, sensor_data: Dict[str, Any]):
        """Check sensor data for alerts."""
        for rule_name, rule in self.alert_manager.rules.items():
            if rule.should_trigger(sensor_data):
                self.alert_manager._create_alert(
                    title=f"Rule Triggered: {rule.name}",
                    description=rule.description,
                    severity=rule.severity,
                    source=rule.source,
                    metadata=sensor_data
                )
                rule.trigger()


# Example usage
if __name__ == "__main__":
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Add notification channels
    email_channel = EmailNotificationChannel(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="your_email@gmail.com",
        password="your_app_password",
        from_email="alerts@predictivemaintenance.com",
        to_emails=["admin@company.com", "maintenance@company.com"]
    )
    alert_manager.add_notification_channel(email_channel)
    
    slack_channel = SlackNotificationChannel(
        webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
        channel="#alerts"
    )
    alert_manager.add_notification_channel(slack_channel)
    
    # Initialize predictive maintenance alerts
    pm_alerts = PredictiveMaintenanceAlerts(alert_manager)
    
    # Start monitoring
    alert_manager.start_monitoring()
    
    # Simulate some sensor data
    sensor_data = {
        'equipment_id': 'MOTOR_001',
        'is_anomaly': True,
        'anomaly_confidence': 0.9,
        'health_score': 0.15,
        'failure_probability': 0.85,
        'model_accuracy': 0.65
    }
    
    # Check for alerts
    pm_alerts.check_sensor_data(sensor_data)
    
    # Wait for alerts to be processed
    time.sleep(5)
    
    # Get alert statistics
    stats = alert_manager.get_alert_statistics()
    print(f"Alert statistics: {json.dumps(stats, indent=2)}")
    
    # Get active alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"Active alerts: {len(active_alerts)}")
    
    # Stop monitoring
    alert_manager.stop_monitoring()
