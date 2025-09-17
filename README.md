# Predictive Maintenance for Sensor Time-Series (AWS + MLOps)

A comprehensive predictive maintenance system for industrial sensor time-series data, built with AWS services and MLOps best practices.

## 🏗️ Architecture Overview

This project implements an end-to-end predictive maintenance solution that:

- **Ingests** sensor data from industrial equipment
- **Processes** time-series data for anomaly detection and failure prediction
- **Trains** machine learning models for predictive maintenance
- **Deploys** models using AWS SageMaker
- **Monitors** model performance and data drift
- **Alerts** maintenance teams when equipment needs attention

## 🚀 Key Features

- **Real-time Data Processing**: Stream processing with AWS Kinesis
- **ML Pipeline**: Automated model training and deployment
- **Anomaly Detection**: Multiple algorithms for detecting equipment anomalies
- **Failure Prediction**: Time-series forecasting for maintenance scheduling
- **Monitoring Dashboard**: Real-time monitoring of equipment health
- **Alert System**: Automated notifications for maintenance teams

## 📁 Project Structure

```
├── data/                          # Data storage and processing
│   ├── raw/                       # Raw sensor data
│   ├── processed/                 # Cleaned and feature-engineered data
│   └── models/                    # Trained model artifacts
├── src/                           # Source code
│   ├── data_pipeline/             # Data ingestion and preprocessing
│   ├── models/                    # ML model implementations
│   ├── monitoring/                # Monitoring and alerting
│   └── deployment/                # AWS deployment scripts
├── notebooks/                     # Jupyter notebooks for exploration
├── tests/                         # Unit and integration tests
├── infrastructure/                # Infrastructure as Code
│   ├── terraform/                 # Terraform configurations
│   └── cloudformation/            # CloudFormation templates
├── mlops/                         # MLOps pipeline configurations
│   ├── dvc/                       # Data Version Control
│   ├── mlflow/                    # MLflow tracking
│   └── github_actions/            # CI/CD workflows
└── docs/                          # Documentation
```

## 🛠️ Technology Stack

### Data Processing

- **Apache Kafka** / **AWS Kinesis** - Real-time data streaming
- **Apache Spark** - Large-scale data processing
- **Pandas** / **NumPy** - Data manipulation
- **Dask** - Parallel computing

### Machine Learning

- **Scikit-learn** - Traditional ML algorithms
- **TensorFlow** / **PyTorch** - Deep learning models
- **Prophet** - Time-series forecasting
- **Isolation Forest** - Anomaly detection
- **LSTM** - Recurrent neural networks

### AWS Services

- **S3** - Data lake storage
- **SageMaker** - ML model training and deployment
- **Lambda** - Serverless compute
- **Kinesis** - Real-time data streaming
- **CloudWatch** - Monitoring and logging
- **EventBridge** - Event-driven architecture

### MLOps

- **MLflow** - Experiment tracking and model registry
- **DVC** - Data version control
- **Docker** - Containerization
- **GitHub Actions** - CI/CD pipeline
- **Terraform** - Infrastructure as Code

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- AWS CLI configured
- Docker installed
- Terraform (optional)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd predictive-maintenance-aws-mlops
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and configuration
   ```

5. **Initialize DVC**
   ```bash
   dvc init
   dvc remote add -d storage s3://your-bucket/dvc-storage
   ```

### Running the System

1. **Start data pipeline**

   ```bash
   python src/data_pipeline/stream_processor.py
   ```

2. **Train models**

   ```bash
   python src/models/train_models.py
   ```

3. **Deploy to AWS**
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform apply
   ```

## 📊 Data Schema

The system expects sensor data in the following format:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "equipment_id": "MOTOR_001",
  "sensor_type": "vibration",
  "value": 0.45,
  "unit": "g",
  "location": "bearing_1",
  "metadata": {
    "temperature": 75.2,
    "rpm": 1800,
    "load": 85.5
  }
}
```

## 🤖 ML Models

### 1. Anomaly Detection

- **Isolation Forest**: Unsupervised anomaly detection
- **One-Class SVM**: Support vector machine for outliers
- **LSTM Autoencoder**: Deep learning approach

### 2. Failure Prediction

- **Prophet**: Time-series forecasting
- **ARIMA**: AutoRegressive Integrated Moving Average
- **LSTM**: Long Short-Term Memory networks
- **Random Forest**: Ensemble method for classification

### 3. Health Scoring

- **Multi-variate Analysis**: Combined sensor readings
- **Trend Analysis**: Historical pattern recognition
- **Threshold-based**: Rule-based health scoring

## 📈 Monitoring & Alerting

### Real-time Monitoring

- Equipment health scores
- Anomaly detection alerts
- Model performance metrics
- Data quality indicators

### Alert Types

- **Critical**: Immediate maintenance required
- **Warning**: Maintenance recommended within 24h
- **Info**: Equipment status update

### Dashboard

- Grafana dashboard for real-time visualization
- Historical trend analysis
- Maintenance scheduling interface

## 🔧 Configuration

### Environment Variables

```bash
AWS_REGION=us-east-1
S3_BUCKET=your-data-bucket
SAGEMAKER_ROLE=your-sagemaker-role
KINESIS_STREAM=your-kinesis-stream
MLFLOW_TRACKING_URI=your-mlflow-server
```

### Model Configuration

Models can be configured in `config/models.yaml`:

- Feature engineering parameters
- Model hyperparameters
- Training schedules
- Deployment settings

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v
```

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Model Development](docs/models.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the [troubleshooting guide](docs/troubleshooting.md)
- Contact the development team

## 🔄 Roadmap

- [ ] Real-time model retraining
- [ ] Multi-cloud deployment support
- [ ] Advanced visualization dashboard
- [ ] Mobile app for maintenance teams
- [ ] Integration with CMMS systems
