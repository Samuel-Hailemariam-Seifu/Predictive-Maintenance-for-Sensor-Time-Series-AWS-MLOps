# Predictive Maintenance for Sensor Time-Series (AWS + MLOps)

A comprehensive, production-ready predictive maintenance system for industrial equipment using sensor time-series data, built with AWS services and MLOps best practices.

[![CI/CD](https://github.com/Samuel-Hailemariam-Seifu/Predictive-Maintenance-for-Sensor-Time-Series--AWS---MLOps-/workflows/CI/CD/badge.svg)](https://github.com/Samuel-Hailemariam-Seifu/Predictive-Maintenance-for-Sensor-Time-Series--AWS---MLOps-/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AWS](https://img.shields.io/badge/AWS-Services-orange.svg)](https://aws.amazon.com/)
[![MLOps](https://img.shields.io/badge/MLOps-Ready-green.svg)](https://ml-ops.org/)

## 🚀 Features

### Core Capabilities
- **Real-time Data Ingestion**: Multi-source sensor data streaming with AWS Kinesis
- **Advanced Analytics**: Time-series analysis, feature engineering, and statistical modeling
- **Machine Learning Models**: Anomaly detection, failure prediction, and health scoring
- **AWS Integration**: S3, Lambda, SageMaker, CloudWatch, and API Gateway
- **MLOps Pipeline**: CI/CD, model versioning, automated retraining, and data drift detection
- **Monitoring & Alerting**: Real-time dashboards, multi-channel notifications, and comprehensive observability
- **Scalable Architecture**: Cloud-native design with auto-scaling and high availability

### Technical Highlights
- **Multi-Algorithm Support**: Isolation Forest, One-Class SVM, LSTM, Random Forest, Gradient Boosting
- **Real-time Processing**: Stream processing with configurable batching and windowing
- **Model Management**: MLflow integration for experiment tracking and model registry
- **Data Versioning**: DVC for data pipeline versioning and reproducibility
- **Infrastructure as Code**: Terraform and CloudFormation for automated deployment
- **Security**: IAM roles, VPC configuration, encryption at rest and in transit

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Components](#-components)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [API Documentation](#-api-documentation)
- [Monitoring & Alerting](#-monitoring--alerting)
- [Deployment](#-deployment)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- AWS CLI configured
- Docker (optional)
- Terraform 1.0+
- Git

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Samuel-Hailemariam-Seifu/Predictive-Maintenance-for-Sensor-Time-Series--AWS---MLOps-.git
cd "Predictive Maintenance for Sensor Time-Series (AWS + MLOps)"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit configuration
nano .env
```

### 3. Deploy Infrastructure

```bash
# Deploy AWS infrastructure
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

### 4. Run the Application

```bash
# Start the application
python src/main.py

# Or use Docker
docker-compose up -d
```

## 🏗️ Architecture

The system follows a microservices architecture with clear separation of concerns:

```mermaid
graph TB
    subgraph "Data Sources"
        A[Sensor Data] --> B[Kinesis Data Streams]
        C[Historical Data] --> D[S3 Data Lake]
    end
    
    subgraph "Processing Layer"
        B --> E[Lambda Functions]
        E --> F[Data Preprocessing]
        F --> G[Feature Engineering]
    end
    
    subgraph "ML Pipeline"
        G --> H[Model Training]
        H --> I[Model Registry]
        I --> J[SageMaker Endpoints]
    end
    
    subgraph "Monitoring & Alerting"
        E --> K[CloudWatch Metrics]
        J --> L[Health Scoring]
        L --> M[Alert System]
        M --> N[Notifications]
    end
    
    subgraph "Storage"
        D --> O[Raw Data]
        F --> P[Processed Data]
        I --> Q[Model Artifacts]
    end
```

### Key Components

- **Data Layer**: S3 for storage, Kinesis for streaming, RDS for metadata
- **Processing Layer**: Lambda functions for real-time processing
- **ML Layer**: SageMaker for model training and deployment
- **API Layer**: API Gateway for RESTful endpoints
- **Monitoring**: CloudWatch, Prometheus, and Grafana
- **MLOps**: GitHub Actions, MLflow, and DVC

## 🔧 Components

### Data Pipeline
- **Real-time Ingestion**: Multi-source sensor data collection
- **Data Preprocessing**: Cleaning, normalization, and outlier detection
- **Feature Engineering**: Time-series features, rolling statistics, health indicators
- **Stream Processing**: Real-time processing with configurable batching

### Machine Learning Models
- **Anomaly Detection**: Isolation Forest, One-Class SVM, LSTM Autoencoder
- **Failure Prediction**: Random Forest, Gradient Boosting, LSTM, Prophet
- **Health Scoring**: Multi-sensor integration and clustering-based scoring

### MLOps Pipeline
- **CI/CD**: Automated testing, validation, and deployment
- **Model Versioning**: MLflow for experiment tracking and model registry
- **Data Versioning**: DVC for data pipeline versioning
- **Automated Retraining**: Data drift detection and model retraining

### Monitoring & Alerting
- **Metrics Collection**: System and application metrics
- **Real-time Dashboards**: Grafana and custom Plotly dashboards
- **Multi-channel Alerting**: Email, Slack, SNS notifications
- **CloudWatch Integration**: AWS native monitoring and logging

## 📦 Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/Samuel-Hailemariam-Seifu/Predictive-Maintenance-for-Sensor-Time-Series--AWS---MLOps-.git
cd "Predictive Maintenance for Sensor Time-Series (AWS + MLOps)"

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual images
docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance
```

### AWS Deployment

```bash
# Deploy infrastructure
cd infrastructure/terraform
terraform init
terraform apply

# Deploy application
python scripts/deploy.py --environment production
```

## ⚙️ Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# S3 Buckets
S3_DATA_BUCKET=predictive-maintenance-data
S3_MODELS_BUCKET=predictive-maintenance-models

# Kinesis Stream
KINESIS_STREAM_NAME=sensor-data-stream

# SageMaker
SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/SageMakerExecutionRole

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_BUCKET=predictive-maintenance-mlflow

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/predictive_maintenance

# Monitoring
GRAFANA_URL=http://localhost:3000
GRAFANA_API_KEY=your_grafana_api_key
PROMETHEUS_URL=http://localhost:9090
```

### Model Configuration

```yaml
# config/model_config.yaml
models:
  anomaly_detection:
    algorithm: isolation_forest
    contamination: 0.1
    n_estimators: 100
    random_state: 42
  
  failure_prediction:
    algorithm: random_forest
    n_estimators: 200
    max_depth: 10
    random_state: 42
  
  health_scoring:
    algorithm: clustering
    n_clusters: 5
    method: kmeans
    random_state: 42

data_pipeline:
  batch_size: 1000
  window_size: 60
  overlap: 0.5

monitoring:
  metrics_interval: 300
  alert_threshold: 0.8
  health_threshold: 0.3
```

## 💡 Usage Examples

### Data Ingestion

```python
from src.data_pipeline.data_ingestion import DataIngestionPipeline

# Initialize pipeline
pipeline = DataIngestionPipeline()

# Ingest sensor data
data = {
    'timestamp': '2024-01-01T10:30:00Z',
    'equipment_id': 'MOTOR_001',
    'sensor_type': 'vibration',
    'value': 0.45,
    'unit': 'g',
    'location': 'bearing_1',
    'metadata': {
        'temperature': 75.2,
        'rpm': 1800,
        'load': 85.5
    }
}

# Ingest single record
pipeline.ingest_data(data)

# Ingest batch data
batch_data = [data1, data2, data3]
pipeline.ingest_batch_data(batch_data)
```

### Model Training

```python
from src.models.anomaly_detection import AnomalyDetectionPipeline

# Initialize pipeline
pipeline = AnomalyDetectionPipeline()

# Load training data
X_train, y_train = pipeline.load_training_data('data/training/')

# Train models
pipeline.train_models(X_train, y_train)

# Evaluate models
metrics = pipeline.evaluate_models(X_test, y_test)
print(f"Model accuracy: {metrics['accuracy']:.3f}")

# Save pipeline
pipeline.save_pipeline('models/anomaly_detection.pkl')
```

### Real-time Inference

```python
from src.aws.sagemaker_integration import SageMakerInferenceManager

# Initialize inference manager
inference_manager = SageMakerInferenceManager()

# Make predictions
features = [0.5, 75.0, 1800, 85.0]

# Anomaly detection
anomaly_prediction = inference_manager.predict_anomaly(features, 'anomaly-detection-endpoint')
print(f"Anomaly detected: {anomaly_prediction['is_anomaly']}")

# Failure prediction
failure_prediction = inference_manager.predict_failure(features, 'failure-prediction-endpoint')
print(f"Failure probability: {failure_prediction['failure_probability']:.3f}")

# Health scoring
health_prediction = inference_manager.predict_health_score(features, 'health-scoring-endpoint')
print(f"Health score: {health_prediction['health_score']:.3f}")
```

### Monitoring and Alerting

```python
from src.monitoring.alerting_system import AlertManager
from src.monitoring.metrics_collector import MetricsAggregator

# Initialize alert manager
alert_manager = AlertManager()

# Add notification channels
from src.monitoring.alerting_system import EmailNotificationChannel, SlackNotificationChannel

email_channel = EmailNotificationChannel(
    smtp_server="smtp.gmail.com",
    smtp_port=587,
    username="your_email@gmail.com",
    password="your_app_password",
    from_email="alerts@predictivemaintenance.com",
    to_emails=["admin@company.com"]
)

slack_channel = SlackNotificationChannel(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
    channel="#alerts"
)

alert_manager.add_notification_channel(email_channel)
alert_manager.add_notification_channel(slack_channel)

# Start monitoring
alert_manager.start_monitoring()

# Initialize metrics collection
metrics_aggregator = MetricsAggregator()
metrics_aggregator.start_aggregation()
```

## 📚 API Documentation

The system provides a comprehensive RESTful API for data ingestion, model inference, and system management.

### Base URL
```
https://api.predictivemaintenance.com/v1
```

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/data/sensor` | Ingest sensor data |
| `POST` | `/data/batch` | Ingest batch sensor data |
| `POST` | `/models/anomaly/predict` | Predict anomalies |
| `POST` | `/models/failure/predict` | Predict equipment failures |
| `POST` | `/models/health/score` | Calculate health scores |
| `GET` | `/equipment` | List all equipment |
| `GET` | `/equipment/{id}/health` | Get equipment health |
| `GET` | `/alerts` | Get active alerts |
| `POST` | `/alerts/{id}/acknowledge` | Acknowledge alert |
| `GET` | `/system/health` | Get system health |

### Example API Usage

```bash
# Ingest sensor data
curl -X POST "https://api.predictivemaintenance.com/v1/data/sensor" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "MOTOR_001",
    "sensor_type": "vibration",
    "value": 0.45,
    "unit": "g"
  }'

# Get equipment health
curl -X GET "https://api.predictivemaintenance.com/v1/equipment/MOTOR_001/health" \
  -H "Authorization: Bearer your_jwt_token"

# Predict anomalies
curl -X POST "https://api.predictivemaintenance.com/v1/models/anomaly/predict" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "equipment_id": "MOTOR_001",
      "sensor_readings": [0.45, 0.52, 0.48, 0.51, 0.49]
    }
  }'
```

For complete API documentation, see [API Documentation](docs/api.md).

## 📊 Monitoring & Alerting

### Dashboards

- **Grafana Dashboard**: Real-time system metrics and equipment health
- **CloudWatch Dashboard**: AWS service metrics and logs
- **Custom Plotly Dashboard**: Interactive business metrics and KPIs

### Alerting Channels

- **Email**: SMTP-based email notifications
- **Slack**: Webhook-based Slack notifications
- **SNS**: AWS SNS for scalable notifications
- **Webhooks**: Custom webhook endpoints

### Key Metrics

- **System Metrics**: CPU, memory, disk usage, network I/O
- **Application Metrics**: Request counts, error rates, response times
- **Business Metrics**: Equipment health, maintenance costs, downtime
- **ML Metrics**: Model accuracy, prediction latency, data drift

## 🚀 Deployment

### AWS Deployment

```bash
# Deploy infrastructure
cd infrastructure/terraform
terraform init
terraform plan
terraform apply

# Deploy application
python scripts/deploy.py --environment production

# Deploy models
python scripts/deploy_models.py --environment production
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual images
docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

For detailed deployment instructions, see [Deployment Guide](docs/deployment.md).

## 🛠️ Development

### Development Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/
mypy src/
```

### Code Quality

- **Linting**: flake8, black, mypy
- **Testing**: pytest with coverage reporting
- **Type Checking**: mypy for static type checking
- **Security**: bandit for security scanning
- **Pre-commit**: Automated code quality checks

### Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run end-to-end tests
pytest tests/e2e/

# Run with coverage
pytest --cov=src tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Run code quality checks (`pre-commit run --all-files`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 🔧 Troubleshooting

### Common Issues

1. **Lambda timeout**: Increase timeout in Lambda configuration
2. **SageMaker endpoint issues**: Check IAM permissions and model artifacts
3. **Database connection**: Verify security groups and VPC configuration
4. **API Gateway errors**: Check Lambda function logs and permissions
5. **Model performance**: Check data quality and model retraining

### Debugging

```bash
# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/

# Check SageMaker logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/

# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix /aws/apigateway/

# Monitor system health
python scripts/monitor_performance.py
```

### Performance Optimization

```bash
# Monitor performance
python scripts/monitor_performance.py --environment production

# Optimize resources
python scripts/optimize_resources.py --environment production

# Check model performance
python scripts/check_model_performance.py
```

## 📖 Documentation

- [Architecture Guide](docs/architecture.md) - System architecture and design
- [API Documentation](docs/api.md) - Complete API reference
- [Deployment Guide](docs/deployment.md) - Deployment instructions
- [Development Guide](docs/development.md) - Development setup and guidelines
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure code quality
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/Samuel-Hailemariam-Seifu/Predictive-Maintenance-for-Sensor-Time-Series--AWS---MLOps-/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Samuel-Hailemariam-Seifu/Predictive-Maintenance-for-Sensor-Time-Series--AWS---MLOps-/discussions)
- **Email**: support@predictivemaintenance.com

## 🗺️ Roadmap

- [ ] **Advanced Deep Learning**: Transformer models for time-series
- [ ] **Edge Computing**: Local processing for real-time response
- [ ] **Mobile Application**: Field technician mobile app
- [ ] **CMMS Integration**: Maintenance management system integration
- [ ] **Augmented Reality**: AR-based maintenance guidance
- [ ] **Digital Twins**: Virtual equipment representations
- [ ] **Federated Learning**: Distributed model training
- [ ] **AutoML**: Automated model selection and tuning

## 🙏 Acknowledgments

- **AWS** for cloud infrastructure services
- **MLflow** team for MLOps platform
- **Scikit-learn** team for machine learning algorithms
- **Plotly** team for visualization tools
- **All contributors** and maintainers

## 📊 Project Status

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2024--01--17-orange)

---

**Built with ❤️ for the future of predictive maintenance**