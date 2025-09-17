"""
Monitoring Dashboard

Real-time monitoring dashboard for the predictive maintenance system
with Grafana integration and custom visualizations.
"""

import json
import time
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import boto3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GrafanaDashboardManager:
    """Manages Grafana dashboards for monitoring."""
    
    def __init__(self, grafana_url: str, api_key: str):
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a new Grafana dashboard."""
        try:
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                headers=self.headers,
                json=dashboard_config
            )
            response.raise_for_status()
            
            result = response.json()
            dashboard_id = result['id']
            
            logger.info(f"Created Grafana dashboard: {dashboard_id}")
            return dashboard_id
            
        except Exception as e:
            logger.error(f"Error creating Grafana dashboard: {e}")
            raise
    
    def update_dashboard(self, dashboard_id: str, dashboard_config: Dict[str, Any]) -> bool:
        """Update an existing Grafana dashboard."""
        try:
            response = requests.put(
                f"{self.grafana_url}/api/dashboards/db/{dashboard_id}",
                headers=self.headers,
                json=dashboard_config
            )
            response.raise_for_status()
            
            logger.info(f"Updated Grafana dashboard: {dashboard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating Grafana dashboard: {e}")
            return False
    
    def create_predictive_maintenance_dashboard(self) -> str:
        """Create a comprehensive predictive maintenance dashboard."""
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Predictive Maintenance System",
                "tags": ["predictive-maintenance", "monitoring"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "System Overview",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "up{job=\"predictive-maintenance\"}",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "system_cpu_percent",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "system_memory_percent",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Records Processed",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "app_records_processed_total",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                    },
                    {
                        "id": 5,
                        "title": "Anomalies Detected",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "app_anomalies_detected_total",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
                    },
                    {
                        "id": 6,
                        "title": "Health Scores",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "app_health_score",
                                "refId": "A"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            }
        }
        
        return self.create_dashboard(dashboard_config)


class PlotlyDashboard:
    """Interactive Plotly dashboard for monitoring."""
    
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Predictive Maintenance System Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # System Overview
            html.Div([
                html.H2("System Overview"),
                html.Div(id='system-overview', className='row')
            ], className='section'),
            
            # Equipment Health
            html.Div([
                html.H2("Equipment Health"),
                dcc.Graph(id='health-scores-chart')
            ], className='section'),
            
            # Anomaly Detection
            html.Div([
                html.H2("Anomaly Detection"),
                dcc.Graph(id='anomaly-chart')
            ], className='section'),
            
            # Model Performance
            html.Div([
                html.H2("Model Performance"),
                dcc.Graph(id='model-performance-chart')
            ], className='section'),
            
            # Alerts
            html.Div([
                html.H2("Active Alerts"),
                html.Div(id='alerts-table')
            ], className='section'),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            [Output('system-overview', 'children'),
             Output('health-scores-chart', 'figure'),
             Output('anomaly-chart', 'figure'),
             Output('model-performance-chart', 'figure'),
             Output('alerts-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return (
                self.create_system_overview(),
                self.create_health_scores_chart(),
                self.create_anomaly_chart(),
                self.create_model_performance_chart(),
                self.create_alerts_table()
            )
    
    def create_system_overview(self):
        """Create system overview cards."""
        # This would typically fetch real data
        cpu_usage = 45.2
        memory_usage = 67.8
        disk_usage = 23.1
        active_alerts = 3
        
        return html.Div([
            html.Div([
                html.H3("CPU Usage"),
                html.H2(f"{cpu_usage}%", style={'color': 'green' if cpu_usage < 80 else 'red'})
            ], className='col-md-3 card'),
            
            html.Div([
                html.H3("Memory Usage"),
                html.H2(f"{memory_usage}%", style={'color': 'green' if memory_usage < 80 else 'red'})
            ], className='col-md-3 card'),
            
            html.Div([
                html.H3("Disk Usage"),
                html.H2(f"{disk_usage}%", style={'color': 'green' if disk_usage < 80 else 'red'})
            ], className='col-md-3 card'),
            
            html.Div([
                html.H3("Active Alerts"),
                html.H2(f"{active_alerts}", style={'color': 'red' if active_alerts > 0 else 'green'})
            ], className='col-md-3 card')
        ], className='row')
    
    def create_health_scores_chart(self):
        """Create equipment health scores chart."""
        # Sample data - in practice, this would come from your data source
        equipment_data = {
            'MOTOR_001': {'health_score': 0.85, 'status': 'Good'},
            'MOTOR_002': {'health_score': 0.72, 'status': 'Fair'},
            'PUMP_001': {'health_score': 0.45, 'status': 'Poor'},
            'PUMP_002': {'health_score': 0.91, 'status': 'Excellent'},
            'COMPRESSOR_001': {'health_score': 0.23, 'status': 'Critical'}
        }
        
        equipment_ids = list(equipment_data.keys())
        health_scores = [data['health_score'] for data in equipment_data.values()]
        statuses = [data['status'] for data in equipment_data.values()]
        
        # Color mapping
        color_map = {
            'Excellent': 'green',
            'Good': 'lightgreen',
            'Fair': 'yellow',
            'Poor': 'orange',
            'Critical': 'red'
        }
        colors = [color_map[status] for status in statuses]
        
        fig = go.Figure(data=[
            go.Bar(
                x=equipment_ids,
                y=health_scores,
                marker_color=colors,
                text=[f"{score:.2f}" for score in health_scores],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Equipment Health Scores",
            xaxis_title="Equipment ID",
            yaxis_title="Health Score",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def create_anomaly_chart(self):
        """Create anomaly detection chart."""
        # Sample time series data
        timestamps = pd.date_range(start='2024-01-01', periods=100, freq='1H')
        values = [0.5 + 0.3 * (i % 10) / 10 + 0.1 * (i % 7) for i in range(100)]
        anomalies = [i % 15 == 0 for i in range(100)]  # Simulate anomalies
        
        fig = go.Figure()
        
        # Normal data
        normal_data = [(t, v) for t, v, a in zip(timestamps, values, anomalies) if not a]
        if normal_data:
            normal_times, normal_values = zip(*normal_data)
            fig.add_trace(go.Scatter(
                x=normal_times,
                y=normal_values,
                mode='lines',
                name='Normal',
                line=dict(color='blue')
            ))
        
        # Anomaly data
        anomaly_data = [(t, v) for t, v, a in zip(timestamps, values, anomalies) if a]
        if anomaly_data:
            anomaly_times, anomaly_values = zip(*anomaly_data)
            fig.add_trace(go.Scatter(
                x=anomaly_times,
                y=anomaly_values,
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=10)
            ))
        
        fig.update_layout(
            title="Sensor Data with Anomalies",
            xaxis_title="Time",
            yaxis_title="Sensor Value"
        )
        
        return fig
    
    def create_model_performance_chart(self):
        """Create model performance chart."""
        # Sample model performance data
        models = ['Anomaly Detection', 'Failure Prediction', 'Health Scoring']
        accuracy = [0.92, 0.87, 0.89]
        precision = [0.88, 0.85, 0.91]
        recall = [0.90, 0.89, 0.87]
        f1_score = [0.89, 0.87, 0.89]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1 Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        metrics = [accuracy, precision, recall, f1_score]
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(x=models, y=metric, name=title, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Metrics",
            height=600
        )
        
        return fig
    
    def create_alerts_table(self):
        """Create alerts table."""
        # Sample alerts data
        alerts = [
            {'id': 'ALERT_001', 'title': 'High CPU Usage', 'severity': 'Warning', 'timestamp': '2024-01-01 10:30:00'},
            {'id': 'ALERT_002', 'title': 'Critical Health Score', 'severity': 'Critical', 'timestamp': '2024-01-01 10:25:00'},
            {'id': 'ALERT_003', 'title': 'Model Performance Degraded', 'severity': 'Error', 'timestamp': '2024-01-01 10:20:00'}
        ]
        
        table_rows = []
        for alert in alerts:
            severity_color = {
                'Critical': 'red',
                'Error': 'orange',
                'Warning': 'yellow',
                'Info': 'blue'
            }.get(alert['severity'], 'black')
            
            table_rows.append(
                html.Tr([
                    html.Td(alert['id']),
                    html.Td(alert['title']),
                    html.Td(alert['severity'], style={'color': severity_color}),
                    html.Td(alert['timestamp'])
                ])
            )
        
        return html.Table([
            html.Thead([
                html.Tr([
                    html.Th("ID"),
                    html.Th("Title"),
                    html.Th("Severity"),
                    html.Th("Timestamp")
                ])
            ]),
            html.Tbody(table_rows)
        ], className='table table-striped')
    
    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        self.app.run_server(debug=debug, port=port)


class CloudWatchDashboardManager:
    """Manages CloudWatch dashboards."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.cloudwatch = boto3.client('cloudwatch', region_name=region_name)
    
    def create_dashboard(self, dashboard_name: str, widgets: List[Dict[str, Any]]) -> bool:
        """Create a CloudWatch dashboard."""
        try:
            dashboard_body = {
                "widgets": widgets
            }
            
            self.cloudwatch.put_dashboard(
                DashboardName=dashboard_name,
                DashboardBody=json.dumps(dashboard_body)
            )
            
            logger.info(f"Created CloudWatch dashboard: {dashboard_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating CloudWatch dashboard: {e}")
            return False
    
    def create_predictive_maintenance_widgets(self) -> List[Dict[str, Any]]:
        """Create widgets for predictive maintenance dashboard."""
        widgets = [
            {
                "type": "metric",
                "x": 0,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["PredictiveMaintenance", "CPUUtilization"],
                        [".", "MemoryUtilization"],
                        [".", "DiskUtilization"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "System Metrics",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 0,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["PredictiveMaintenance", "RecordsProcessed", "Component", "data_ingestion"],
                        [".", ".", ".", "model_inference"],
                        [".", ".", ".", "anomaly_detection"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Application Metrics",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "x": 0,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["PredictiveMaintenance", "AnomaliesDetected"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Anomalies Detected",
                    "period": 300
                }
            },
            {
                "type": "metric",
                "x": 12,
                "y": 6,
                "width": 12,
                "height": 6,
                "properties": {
                    "metrics": [
                        ["PredictiveMaintenance", "ProcessingErrors", "Component", "data_ingestion"],
                        [".", ".", ".", "model_inference"],
                        [".", ".", ".", "anomaly_detection"]
                    ],
                    "view": "timeSeries",
                    "stacked": False,
                    "region": "us-east-1",
                    "title": "Processing Errors",
                    "period": 300
                }
            }
        ]
        
        return widgets


# Example usage
if __name__ == "__main__":
    # Initialize Grafana dashboard manager
    grafana_manager = GrafanaDashboardManager(
        grafana_url="http://localhost:3000",
        api_key="your_grafana_api_key"
    )
    
    # Create predictive maintenance dashboard
    dashboard_id = grafana_manager.create_predictive_maintenance_dashboard()
    print(f"Created Grafana dashboard: {dashboard_id}")
    
    # Initialize CloudWatch dashboard manager
    cloudwatch_manager = CloudWatchDashboardManager()
    
    # Create CloudWatch dashboard
    widgets = cloudwatch_manager.create_predictive_maintenance_widgets()
    cloudwatch_manager.create_dashboard("PredictiveMaintenance", widgets)
    
    # Initialize and run Plotly dashboard
    plotly_dashboard = PlotlyDashboard()
    plotly_dashboard.run(debug=True, port=8050)
