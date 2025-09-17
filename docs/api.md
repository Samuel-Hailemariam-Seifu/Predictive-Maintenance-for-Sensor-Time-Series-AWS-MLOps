# API Documentation

## Overview

The Predictive Maintenance API provides RESTful endpoints for managing sensor data, model inference, and system monitoring. The API is built using FastAPI and provides comprehensive documentation with interactive testing capabilities.

## Base URL

```
https://api.predictivemaintenance.com/v1
```

## Authentication

All API endpoints require authentication using AWS Cognito JWT tokens.

### Headers

```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

## Endpoints

### 1. Data Ingestion

#### POST /data/sensor

Ingest sensor data for real-time processing.

**Request Body:**
```json
{
  "timestamp": "2024-01-01T10:30:00Z",
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

**Response:**
```json
{
  "status": "success",
  "message": "Data ingested successfully",
  "data_id": "data_123456789",
  "timestamp": "2024-01-01T10:30:00Z"
}
```

#### POST /data/batch

Ingest multiple sensor data points in batch.

**Request Body:**
```json
{
  "data": [
    {
      "timestamp": "2024-01-01T10:30:00Z",
      "equipment_id": "MOTOR_001",
      "sensor_type": "vibration",
      "value": 0.45,
      "unit": "g",
      "location": "bearing_1"
    },
    {
      "timestamp": "2024-01-01T10:30:01Z",
      "equipment_id": "MOTOR_001",
      "sensor_type": "temperature",
      "value": 75.2,
      "unit": "C",
      "location": "housing"
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Batch data ingested successfully",
  "processed_count": 2,
  "failed_count": 0,
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### 2. Model Inference

#### POST /models/anomaly/predict

Predict anomalies in sensor data.

**Request Body:**
```json
{
  "data": {
    "equipment_id": "MOTOR_001",
    "sensor_readings": [0.45, 0.52, 0.48, 0.51, 0.49],
    "metadata": {
      "temperature": 75.2,
      "rpm": 1800,
      "load": 85.5
    }
  }
}
```

**Response:**
```json
{
  "prediction": {
    "is_anomaly": false,
    "confidence": 0.85,
    "anomaly_score": 0.15,
    "model_version": "v1.2.0"
  },
  "timestamp": "2024-01-01T10:30:00Z"
}
```

#### POST /models/failure/predict

Predict equipment failure probability.

**Request Body:**
```json
{
  "data": {
    "equipment_id": "MOTOR_001",
    "features": {
      "vibration_mean": 0.45,
      "temperature": 75.2,
      "rpm": 1800,
      "load": 85.5,
      "operating_hours": 8760
    }
  }
}
```

**Response:**
```json
{
  "prediction": {
    "failure_probability": 0.15,
    "time_to_failure": "30 days",
    "maintenance_urgency": "medium",
    "confidence": 0.82,
    "model_version": "v1.1.0"
  },
  "timestamp": "2024-01-01T10:30:00Z"
}
```

#### POST /models/health/score

Calculate equipment health score.

**Request Body:**
```json
{
  "data": {
    "equipment_id": "MOTOR_001",
    "sensor_data": {
      "vibration": 0.45,
      "temperature": 75.2,
      "pressure": 50.0,
      "rpm": 1800
    }
  }
}
```

**Response:**
```json
{
  "health_score": {
    "overall_score": 0.85,
    "status": "good",
    "maintenance_urgency": "low",
    "component_scores": {
      "vibration": 0.90,
      "temperature": 0.80,
      "pressure": 0.85,
      "rpm": 0.85
    },
    "model_version": "v1.0.0"
  },
  "timestamp": "2024-01-01T10:30:00Z"
}
```

### 3. Equipment Management

#### GET /equipment

Get list of all equipment.

**Query Parameters:**
- `limit` (optional): Number of results per page (default: 100)
- `offset` (optional): Number of results to skip (default: 0)
- `status` (optional): Filter by equipment status
- `location` (optional): Filter by equipment location

**Response:**
```json
{
  "equipment": [
    {
      "equipment_id": "MOTOR_001",
      "name": "Main Motor",
      "type": "electric_motor",
      "location": "Building A - Floor 1",
      "status": "operational",
      "last_maintenance": "2024-01-01T00:00:00Z",
      "next_maintenance": "2024-02-01T00:00:00Z",
      "health_score": 0.85
    }
  ],
  "total_count": 1,
  "limit": 100,
  "offset": 0
}
```

#### GET /equipment/{equipment_id}

Get detailed information about specific equipment.

**Response:**
```json
{
  "equipment_id": "MOTOR_001",
  "name": "Main Motor",
  "type": "electric_motor",
  "location": "Building A - Floor 1",
  "status": "operational",
  "specifications": {
    "power": "50 kW",
    "voltage": "400V",
    "rpm": 1800,
    "manufacturer": "Siemens"
  },
  "maintenance_history": [
    {
      "date": "2024-01-01T00:00:00Z",
      "type": "preventive",
      "description": "Routine maintenance",
      "technician": "John Doe"
    }
  ],
  "health_score": 0.85,
  "last_updated": "2024-01-01T10:30:00Z"
}
```

#### GET /equipment/{equipment_id}/health

Get health history for specific equipment.

**Query Parameters:**
- `start_date` (optional): Start date for health data
- `end_date` (optional): End date for health data
- `granularity` (optional): Data granularity (hourly, daily, weekly)

**Response:**
```json
{
  "equipment_id": "MOTOR_001",
  "health_history": [
    {
      "timestamp": "2024-01-01T10:00:00Z",
      "health_score": 0.85,
      "status": "good",
      "anomalies": 0,
      "maintenance_urgency": "low"
    }
  ],
  "summary": {
    "average_health": 0.85,
    "trend": "stable",
    "anomaly_count": 0,
    "critical_alerts": 0
  }
}
```

### 4. Alerts and Notifications

#### GET /alerts

Get list of active alerts.

**Query Parameters:**
- `severity` (optional): Filter by alert severity
- `equipment_id` (optional): Filter by equipment
- `status` (optional): Filter by alert status
- `limit` (optional): Number of results per page
- `offset` (optional): Number of results to skip

**Response:**
```json
{
  "alerts": [
    {
      "alert_id": "ALERT_001",
      "title": "High Vibration Detected",
      "description": "Vibration levels exceed normal operating range",
      "severity": "warning",
      "equipment_id": "MOTOR_001",
      "status": "active",
      "created_at": "2024-01-01T10:30:00Z",
      "acknowledged_by": null,
      "acknowledged_at": null
    }
  ],
  "total_count": 1,
  "limit": 100,
  "offset": 0
}
```

#### POST /alerts/{alert_id}/acknowledge

Acknowledge an alert.

**Request Body:**
```json
{
  "acknowledged_by": "user@company.com",
  "notes": "Investigating the issue"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Alert acknowledged successfully",
  "alert_id": "ALERT_001",
  "acknowledged_by": "user@company.com",
  "acknowledged_at": "2024-01-01T10:35:00Z"
}
```

#### POST /alerts/{alert_id}/resolve

Resolve an alert.

**Request Body:**
```json
{
  "resolved_by": "user@company.com",
  "resolution_notes": "Issue resolved by adjusting motor alignment"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Alert resolved successfully",
  "alert_id": "ALERT_001",
  "resolved_by": "user@company.com",
  "resolved_at": "2024-01-01T11:00:00Z"
}
```

### 5. Model Management

#### GET /models

Get list of available models.

**Response:**
```json
{
  "models": [
    {
      "model_id": "anomaly_detection_v1",
      "name": "Anomaly Detection",
      "version": "1.2.0",
      "status": "active",
      "accuracy": 0.92,
      "created_at": "2024-01-01T00:00:00Z",
      "last_updated": "2024-01-01T10:00:00Z"
    }
  ]
}
```

#### GET /models/{model_id}/performance

Get model performance metrics.

**Query Parameters:**
- `start_date` (optional): Start date for performance data
- `end_date` (optional): End date for performance data

**Response:**
```json
{
  "model_id": "anomaly_detection_v1",
  "performance_metrics": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90
  },
  "prediction_count": 10000,
  "error_rate": 0.08,
  "last_updated": "2024-01-01T10:00:00Z"
}
```

#### POST /models/{model_id}/retrain

Trigger model retraining.

**Request Body:**
```json
{
  "training_data_start": "2024-01-01T00:00:00Z",
  "training_data_end": "2024-01-31T23:59:59Z",
  "parameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Model retraining initiated",
  "training_job_id": "training_123456789",
  "estimated_completion": "2024-01-01T12:00:00Z"
}
```

### 6. System Monitoring

#### GET /system/health

Get system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T10:30:00Z",
  "components": {
    "database": "healthy",
    "message_queue": "healthy",
    "ml_models": "healthy",
    "data_pipeline": "healthy"
  },
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.1,
    "active_connections": 150
  }
}
```

#### GET /system/metrics

Get system metrics.

**Query Parameters:**
- `metric` (optional): Specific metric to retrieve
- `start_date` (optional): Start date for metrics
- `end_date` (optional): End date for metrics
- `granularity` (optional): Data granularity

**Response:**
```json
{
  "metrics": [
    {
      "timestamp": "2024-01-01T10:00:00Z",
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "disk_usage": 23.1,
      "requests_per_second": 150.5
    }
  ],
  "summary": {
    "average_cpu": 45.2,
    "average_memory": 67.8,
    "peak_cpu": 78.5,
    "peak_memory": 85.2
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "equipment_id",
      "issue": "Required field is missing"
    },
    "timestamp": "2024-01-01T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid input data |
| `AUTHENTICATION_ERROR` | 401 | Authentication failed |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limiting

API requests are rate-limited to ensure fair usage and system stability.

### Rate Limits

- **Authenticated Users**: 1000 requests per hour
- **API Keys**: 5000 requests per hour
- **Bulk Operations**: 100 requests per hour

### Rate Limit Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Pagination

List endpoints support pagination using `limit` and `offset` parameters.

### Pagination Parameters

- `limit`: Number of results per page (default: 100, max: 1000)
- `offset`: Number of results to skip (default: 0)

### Pagination Response

```json
{
  "data": [...],
  "pagination": {
    "total_count": 1000,
    "limit": 100,
    "offset": 0,
    "has_next": true,
    "has_previous": false
  }
}
```

## Webhooks

The API supports webhooks for real-time notifications.

### Webhook Events

- `alert.created`: New alert created
- `alert.acknowledged`: Alert acknowledged
- `alert.resolved`: Alert resolved
- `equipment.health_changed`: Equipment health status changed
- `model.retrained`: Model retraining completed

### Webhook Payload

```json
{
  "event": "alert.created",
  "timestamp": "2024-01-01T10:30:00Z",
  "data": {
    "alert_id": "ALERT_001",
    "title": "High Vibration Detected",
    "severity": "warning",
    "equipment_id": "MOTOR_001"
  }
}
```

## SDKs and Libraries

### Python SDK

```python
from predictive_maintenance import Client

client = Client(api_key="your_api_key")

# Ingest sensor data
response = client.data.ingest_sensor({
    "equipment_id": "MOTOR_001",
    "sensor_type": "vibration",
    "value": 0.45
})

# Get equipment health
health = client.equipment.get_health("MOTOR_001")
```

### JavaScript SDK

```javascript
const client = new PredictiveMaintenanceClient({
  apiKey: 'your_api_key'
});

// Ingest sensor data
const response = await client.data.ingestSensor({
  equipment_id: 'MOTOR_001',
  sensor_type: 'vibration',
  value: 0.45
});

// Get equipment health
const health = await client.equipment.getHealth('MOTOR_001');
```

## Testing

### Interactive API Documentation

Visit `/docs` for interactive API documentation with Swagger UI.

### Postman Collection

Download the Postman collection from `/postman/collection.json` for easy API testing.

### cURL Examples

```bash
# Ingest sensor data
curl -X POST "https://api.predictivemaintenance.com/v1/data/sensor" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "MOTOR_001",
    "sensor_type": "vibration",
    "value": 0.45
  }'

# Get equipment health
curl -X GET "https://api.predictivemaintenance.com/v1/equipment/MOTOR_001/health" \
  -H "Authorization: Bearer your_jwt_token"
```

## Support

For API support and questions:

- **Documentation**: https://docs.predictivemaintenance.com
- **Support Email**: support@predictivemaintenance.com
- **Status Page**: https://status.predictivemaintenance.com
- **GitHub Issues**: https://github.com/company/predictive-maintenance/issues
