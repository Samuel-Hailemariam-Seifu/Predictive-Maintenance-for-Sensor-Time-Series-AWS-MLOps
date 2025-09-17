# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Predictive Maintenance for Sensor Time-Series system on AWS. The deployment includes infrastructure provisioning, application deployment, and configuration management.

## Prerequisites

### 1. AWS Account Setup

- AWS Account with appropriate permissions
- AWS CLI configured with credentials
- Terraform installed (version 1.0+)
- Docker installed
- Python 3.9+ installed
- Git installed

### 2. Required AWS Services

- S3 (Simple Storage Service)
- Lambda (Serverless Compute)
- SageMaker (Machine Learning Platform)
- Kinesis (Data Streaming)
- CloudWatch (Monitoring)
- IAM (Identity and Access Management)
- VPC (Virtual Private Cloud)
- API Gateway (API Management)

### 3. Third-Party Services

- GitHub (Code Repository)
- MLflow (Model Registry)
- Grafana (Monitoring Dashboards)
- Prometheus (Metrics Collection)

## Infrastructure Deployment

### 1. Clone Repository

```bash
git clone https://github.com/your-org/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

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

### 3. Deploy Infrastructure with Terraform

```bash
# Navigate to infrastructure directory
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="environment=production"

# Apply deployment
terraform apply -var="environment=production"
```

### 4. Deploy Infrastructure with CloudFormation

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name predictive-maintenance \
  --template-body file://infrastructure/cloudformation/template.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters ParameterKey=Environment,ParameterValue=production
```

## Application Deployment

### 1. Build Docker Images

```bash
# Build base image
docker build -t predictive-maintenance-base -f docker/Dockerfile.base .

# Build application image
docker build -t predictive-maintenance-app -f docker/Dockerfile.app .
```

### 2. Deploy Lambda Functions

```bash
# Package Lambda functions
cd src/aws
zip -r lambda-functions.zip lambda_functions.py

# Deploy to AWS
aws lambda create-function \
  --function-name data-processor \
  --runtime python3.9 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler lambda_handler \
  --zip-file fileb://lambda-functions.zip
```

### 3. Deploy SageMaker Models

```bash
# Create SageMaker model
python src/aws/sagemaker_integration.py \
  --action create-model \
  --model-name anomaly-detection \
  --model-artifact-path s3://predictive-maintenance-models/models/anomaly_detection/

# Deploy SageMaker endpoint
python src/aws/sagemaker_integration.py \
  --action deploy \
  --model-name anomaly-detection \
  --endpoint-name anomaly-detection-endpoint
```

### 4. Deploy API Gateway

```bash
# Create API Gateway
aws apigateway create-rest-api \
  --name predictive-maintenance-api \
  --description "Predictive Maintenance API"

# Deploy API Gateway
aws apigateway create-deployment \
  --rest-api-id your-api-id \
  --stage-name production
```

## Data Pipeline Deployment

### 1. Set up Kinesis Data Streams

```bash
# Create Kinesis stream
aws kinesis create-stream \
  --stream-name sensor-data-stream \
  --shard-count 2
```

### 2. Configure Data Ingestion

```bash
# Deploy data ingestion Lambda
python src/data_pipeline/data_ingestion.py \
  --deploy \
  --stream-name sensor-data-stream \
  --bucket-name predictive-maintenance-data
```

### 3. Set up Data Processing

```bash
# Deploy data processing pipeline
python src/data_pipeline/data_preprocessing.py \
  --deploy \
  --input-bucket predictive-maintenance-data \
  --output-bucket predictive-maintenance-processed
```

## MLOps Pipeline Deployment

### 1. Set up MLflow

```bash
# Start MLflow server
mlflow server \
  --backend-store-uri postgresql://user:password@localhost:5432/mlflow \
  --default-artifact-root s3://predictive-maintenance-mlflow \
  --host 0.0.0.0 \
  --port 5000
```

### 2. Configure DVC

```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d s3 s3://predictive-maintenance-data/dvc-storage

# Configure DVC
dvc config core.remote s3
```

### 3. Set up GitHub Actions

```bash
# Configure GitHub secrets
gh secret set AWS_ACCESS_KEY_ID --body "your_access_key"
gh secret set AWS_SECRET_ACCESS_KEY --body "your_secret_key"
gh secret set S3_DATA_BUCKET --body "predictive-maintenance-data"
gh secret set S3_MODELS_BUCKET --body "predictive-maintenance-models"
```

## Monitoring Deployment

### 1. Deploy Prometheus

```bash
# Create Prometheus configuration
kubectl apply -f monitoring/prometheus-config.yaml

# Deploy Prometheus
helm install prometheus prometheus-community/prometheus \
  --set server.persistentVolume.storageClass=gp2
```

### 2. Deploy Grafana

```bash
# Deploy Grafana
helm install grafana grafana/grafana \
  --set persistence.enabled=true \
  --set persistence.storageClassName=gp2

# Get Grafana admin password
kubectl get secret grafana -o jsonpath="{.data.admin-password}" | base64 --decode
```

### 3. Configure CloudWatch

```bash
# Create CloudWatch dashboard
python src/monitoring/dashboard.py \
  --action create-cloudwatch-dashboard \
  --dashboard-name predictive-maintenance
```

## Database Setup

### 1. PostgreSQL Database

```bash
# Create RDS instance
aws rds create-db-instance \
  --db-instance-identifier predictive-maintenance-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password your_password \
  --allocated-storage 20
```

### 2. Database Migrations

```bash
# Run database migrations
python -m alembic upgrade head
```

### 3. Seed Data

```bash
# Seed initial data
python scripts/seed_database.py
```

## Configuration Management

### 1. Environment Configuration

Create environment-specific configuration files:

```bash
# Production configuration
cp config/production.yaml.example config/production.yaml

# Staging configuration
cp config/staging.yaml.example config/staging.yaml

# Development configuration
cp config/development.yaml.example config/development.yaml
```

### 2. Secrets Management

```bash
# Store secrets in AWS Secrets Manager
aws secretsmanager create-secret \
  --name predictive-maintenance/database \
  --secret-string '{"username":"admin","password":"your_password"}'

aws secretsmanager create-secret \
  --name predictive-maintenance/api-keys \
  --secret-string '{"jwt_secret":"your_jwt_secret"}'
```

### 3. Configuration Validation

```bash
# Validate configuration
python scripts/validate_config.py --environment production
```

## Testing Deployment

### 1. Health Checks

```bash
# Check system health
curl -X GET "https://api.predictivemaintenance.com/v1/system/health"

# Check individual components
curl -X GET "https://api.predictivemaintenance.com/v1/system/health/database"
curl -X GET "https://api.predictivemaintenance.com/v1/system/health/ml-models"
```

### 2. Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v

# Run end-to-end tests
pytest tests/e2e/ -v
```

### 3. Load Testing

```bash
# Run load tests
locust -f tests/load/locustfile.py \
  --host=https://api.predictivemaintenance.com \
  --users=100 \
  --spawn-rate=10
```

## Monitoring and Alerting

### 1. Set up Alerts

```bash
# Create CloudWatch alarms
python scripts/create_alarms.py --environment production

# Configure SNS topics
aws sns create-topic --name predictive-maintenance-alerts
```

### 2. Dashboard Configuration

```bash
# Import Grafana dashboards
python scripts/import_dashboards.py \
  --grafana-url http://localhost:3000 \
  --api-key your_api_key
```

### 3. Log Aggregation

```bash
# Configure log aggregation
python scripts/setup_logging.py \
  --log-group predictive-maintenance \
  --retention-days 30
```

## Security Configuration

### 1. IAM Roles and Policies

```bash
# Create IAM roles
python scripts/create_iam_roles.py --environment production

# Attach policies
python scripts/attach_policies.py --environment production
```

### 2. VPC Configuration

```bash
# Create VPC
aws ec2 create-vpc --cidr-block 10.0.0.0/16

# Create subnets
aws ec2 create-subnet --vpc-id vpc-12345678 --cidr-block 10.0.1.0/24
```

### 3. Security Groups

```bash
# Create security groups
aws ec2 create-security-group \
  --group-name predictive-maintenance-sg \
  --description "Security group for predictive maintenance"
```

## Backup and Recovery

### 1. Database Backup

```bash
# Create automated backup
aws rds create-db-snapshot \
  --db-instance-identifier predictive-maintenance-db \
  --db-snapshot-identifier predictive-maintenance-backup-$(date +%Y%m%d)
```

### 2. Data Backup

```bash
# Backup S3 data
aws s3 sync s3://predictive-maintenance-data s3://predictive-maintenance-backup/data/

# Backup models
aws s3 sync s3://predictive-maintenance-models s3://predictive-maintenance-backup/models/
```

### 3. Disaster Recovery

```bash
# Create disaster recovery plan
python scripts/create_dr_plan.py --environment production
```

## Maintenance and Updates

### 1. Application Updates

```bash
# Deploy application updates
python scripts/deploy_update.py \
  --version 1.2.0 \
  --environment production
```

### 2. Model Updates

```bash
# Deploy model updates
python scripts/deploy_model.py \
  --model-name anomaly-detection \
  --version 1.2.0 \
  --environment production
```

### 3. Infrastructure Updates

```bash
# Update infrastructure
terraform plan -var="environment=production"
terraform apply -var="environment=production"
```

## Troubleshooting

### 1. Common Issues

- **Lambda timeout**: Increase timeout in Lambda configuration
- **SageMaker endpoint issues**: Check IAM permissions and model artifacts
- **Database connection**: Verify security groups and VPC configuration
- **API Gateway errors**: Check Lambda function logs and permissions

### 2. Debugging

```bash
# Check Lambda logs
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/

# Check SageMaker logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/

# Check API Gateway logs
aws logs describe-log-groups --log-group-name-prefix /aws/apigateway/
```

### 3. Performance Optimization

```bash
# Monitor performance
python scripts/monitor_performance.py --environment production

# Optimize resources
python scripts/optimize_resources.py --environment production
```

## Rollback Procedures

### 1. Application Rollback

```bash
# Rollback application
python scripts/rollback.py \
  --component application \
  --version 1.1.0 \
  --environment production
```

### 2. Model Rollback

```bash
# Rollback model
python scripts/rollback.py \
  --component model \
  --model-name anomaly-detection \
  --version 1.1.0 \
  --environment production
```

### 3. Infrastructure Rollback

```bash
# Rollback infrastructure
terraform apply -var="environment=production" -var="version=1.1.0"
```

## Support and Maintenance

### 1. Monitoring

- Set up comprehensive monitoring with CloudWatch, Prometheus, and Grafana
- Configure alerts for critical system components
- Monitor performance metrics and user experience

### 2. Maintenance

- Regular security updates and patches
- Database maintenance and optimization
- Model retraining and validation
- Infrastructure scaling and optimization

### 3. Support

- 24/7 monitoring and alerting
- Incident response procedures
- Regular health checks and maintenance windows
- Documentation updates and training

This deployment guide provides a comprehensive approach to deploying and maintaining the Predictive Maintenance system in a production environment.
