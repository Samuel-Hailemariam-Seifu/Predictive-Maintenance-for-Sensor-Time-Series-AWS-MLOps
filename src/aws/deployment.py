"""
AWS Deployment Module

Handles deployment of the predictive maintenance system to AWS using
CloudFormation, Terraform, and other AWS services.
"""

import boto3
import json
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging
import os
import zipfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AWSDeploymentManager:
    """Manages AWS deployment for the predictive maintenance system."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize AWS deployment manager.
        
        Args:
            region_name: AWS region name
        """
        self.region_name = region_name
        self.session = boto3.Session(region_name=region_name)
        self.cf_client = self.session.client('cloudformation')
        self.s3_client = self.session.client('s3')
        self.lambda_client = self.session.client('lambda')
        self.iam_client = self.session.client('iam')
        self.sagemaker_client = self.session.client('sagemaker')
        self.kinesis_client = self.session.client('kinesis')
    
    def create_cloudformation_stack(self, stack_name: str, template_path: str,
                                  parameters: Dict[str, str] = None) -> bool:
        """
        Create a CloudFormation stack.
        
        Args:
            stack_name: Name of the CloudFormation stack
            template_path: Path to the CloudFormation template
            parameters: Stack parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read template
            with open(template_path, 'r') as f:
                template_body = f.read()
            
            # Prepare parameters
            stack_parameters = []
            if parameters:
                for key, value in parameters.items():
                    stack_parameters.append({
                        'ParameterKey': key,
                        'ParameterValue': value
                    })
            
            # Create stack
            response = self.cf_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=stack_parameters,
                Capabilities=['CAPABILITY_NAMED_IAM']
            )
            
            logger.info(f"CloudFormation stack {stack_name} creation initiated")
            
            # Wait for stack creation to complete
            self._wait_for_stack_completion(stack_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating CloudFormation stack: {e}")
            return False
    
    def _wait_for_stack_completion(self, stack_name: str, timeout: int = 1800):
        """Wait for CloudFormation stack to complete."""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                response = self.cf_client.describe_stacks(StackName=stack_name)
                stack_status = response['Stacks'][0]['StackStatus']
                
                if stack_status in ['CREATE_COMPLETE', 'UPDATE_COMPLETE']:
                    logger.info(f"Stack {stack_name} completed successfully")
                    return
                elif stack_status in ['CREATE_FAILED', 'UPDATE_FAILED', 'ROLLBACK_COMPLETE']:
                    raise Exception(f"Stack {stack_name} failed with status: {stack_status}")
                
                logger.info(f"Stack {stack_name} status: {stack_status}")
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error checking stack status: {e}")
                raise
        
        raise Exception(f"Stack {stack_name} did not complete within {timeout} seconds")
    
    def deploy_lambda_function(self, function_name: str, code_path: str,
                             handler: str, runtime: str = 'python3.9',
                             role_arn: str = None) -> bool:
        """
        Deploy a Lambda function.
        
        Args:
            function_name: Name of the Lambda function
            code_path: Path to the function code
            handler: Function handler
            runtime: Python runtime version
            role_arn: IAM role ARN for the function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create deployment package
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add Python files
                for root, dirs, files in os.walk(code_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            arc_name = os.path.relpath(file_path, code_path)
                            zip_file.write(file_path, arc_name)
                
                # Add requirements.txt if it exists
                requirements_path = os.path.join(code_path, 'requirements.txt')
                if os.path.exists(requirements_path):
                    zip_file.write(requirements_path, 'requirements.txt')
            
            zip_buffer.seek(0)
            
            # Create or update function
            try:
                # Try to update existing function
                self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_buffer.getvalue()
                )
                logger.info(f"Lambda function {function_name} updated successfully")
                
            except self.lambda_client.exceptions.ResourceNotFoundException:
                # Create new function
                if not role_arn:
                    role_arn = self._create_lambda_role(function_name)
                
                self.lambda_client.create_function(
                    FunctionName=function_name,
                    Runtime=runtime,
                    Role=role_arn,
                    Handler=handler,
                    Code={'ZipFile': zip_buffer.getvalue()},
                    Timeout=300,
                    MemorySize=512
                )
                logger.info(f"Lambda function {function_name} created successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying Lambda function: {e}")
            return False
    
    def _create_lambda_role(self, function_name: str) -> str:
        """Create IAM role for Lambda function."""
        try:
            role_name = f"{function_name}-role"
            
            # Create role
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps({
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "lambda.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                })
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach policies
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            # Attach custom policy for S3, Kinesis, and SageMaker access
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "s3:GetObject",
                            "s3:PutObject",
                            "s3:DeleteObject"
                        ],
                        "Resource": "arn:aws:s3:::*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "kinesis:PutRecord",
                            "kinesis:PutRecords",
                            "kinesis:GetRecords",
                            "kinesis:GetShardIterator"
                        ],
                        "Resource": "arn:aws:kinesis:*:*:stream/*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:InvokeEndpoint",
                            "sagemaker:DescribeEndpoint"
                        ],
                        "Resource": "arn:aws:sagemaker:*:*:endpoint/*"
                    }
                ]
            }
            
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=f"{function_name}-policy",
                PolicyDocument=json.dumps(policy_document)
            )
            
            return role_arn
            
        except Exception as e:
            logger.error(f"Error creating Lambda role: {e}")
            raise
    
    def create_kinesis_stream(self, stream_name: str, shard_count: int = 1) -> bool:
        """
        Create a Kinesis stream.
        
        Args:
            stream_name: Name of the stream
            shard_count: Number of shards
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.kinesis_client.create_stream(
                StreamName=stream_name,
                ShardCount=shard_count
            )
            
            logger.info(f"Kinesis stream {stream_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Kinesis stream: {e}")
            return False
    
    def create_s3_bucket(self, bucket_name: str) -> bool:
        """
        Create an S3 bucket.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': self.region_name
                }
            )
            
            logger.info(f"S3 bucket {bucket_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating S3 bucket: {e}")
            return False
    
    def deploy_sagemaker_model(self, model_name: str, model_artifact_path: str,
                              role_arn: str, instance_type: str = 'ml.m5.large') -> str:
        """
        Deploy a SageMaker model.
        
        Args:
            model_name: Name of the model
            model_artifact_path: S3 path to model artifacts
            role_arn: SageMaker execution role ARN
            instance_type: EC2 instance type
            
        Returns:
            Endpoint name
        """
        try:
            # Create model
            model_response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': '763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:0.23-1-cpu-py3',
                    'ModelDataUrl': model_artifact_path
                },
                ExecutionRoleArn=role_arn
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config"
            self.sagemaker_client.create_endpoint_configuration(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': 1,
                        'InstanceType': instance_type
                    }
                ]
            )
            
            # Create endpoint
            endpoint_name = f"{model_name}-endpoint"
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            logger.info(f"SageMaker model {model_name} deployed successfully")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Error deploying SageMaker model: {e}")
            raise
    
    def create_eventbridge_rule(self, rule_name: str, schedule_expression: str,
                               target_arn: str) -> bool:
        """
        Create an EventBridge rule.
        
        Args:
            rule_name: Name of the rule
            schedule_expression: Schedule expression (e.g., 'rate(5 minutes)')
            target_arn: ARN of the target (e.g., Lambda function)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            eventbridge_client = self.session.client('events')
            
            # Create rule
            eventbridge_client.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule_expression,
                State='ENABLED'
            )
            
            # Add target
            eventbridge_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': target_arn
                    }
                ]
            )
            
            logger.info(f"EventBridge rule {rule_name} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating EventBridge rule: {e}")
            return False
    
    def deploy_complete_system(self, config: Dict[str, Any]) -> bool:
        """
        Deploy the complete predictive maintenance system.
        
        Args:
            config: Deployment configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting complete system deployment")
            
            # Create S3 buckets
            if not self.create_s3_bucket(config['data_bucket']):
                return False
            
            if not self.create_s3_bucket(config['models_bucket']):
                return False
            
            # Create Kinesis stream
            if not self.create_kinesis_stream(config['stream_name'], config.get('shard_count', 1)):
                return False
            
            # Deploy Lambda functions
            lambda_functions = [
                {
                    'name': 'data-processor',
                    'code_path': 'src/aws/lambda_functions.py',
                    'handler': 'lambda_handler'
                },
                {
                    'name': 'anomaly-detector',
                    'code_path': 'src/aws/lambda_functions.py',
                    'handler': 'anomaly_detection_lambda'
                },
                {
                    'name': 'health-scorer',
                    'code_path': 'src/aws/lambda_functions.py',
                    'handler': 'health_scoring_lambda'
                }
            ]
            
            for func in lambda_functions:
                if not self.deploy_lambda_function(
                    func['name'],
                    func['code_path'],
                    func['handler']
                ):
                    return False
            
            # Create EventBridge rules for scheduled tasks
            if not self.create_eventbridge_rule(
                'model-retraining',
                'rate(24 hours)',
                f"arn:aws:lambda:{self.region_name}:123456789012:function:model-retrainer"
            ):
                return False
            
            logger.info("Complete system deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Error in complete system deployment: {e}")
            return False


class InfrastructureAsCode:
    """Manages Infrastructure as Code for the predictive maintenance system."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize Infrastructure as Code manager.
        
        Args:
            region_name: AWS region name
        """
        self.region_name = region_name
        self.session = boto3.Session(region_name=region_name)
    
    def generate_cloudformation_template(self, output_path: str = 'infrastructure/cloudformation/template.yaml'):
        """
        Generate CloudFormation template for the predictive maintenance system.
        
        Args:
            output_path: Path to save the template
        """
        template = {
            'AWSTemplateFormatVersion': '2010-09-09',
            'Description': 'Predictive Maintenance System Infrastructure',
            'Parameters': {
                'Environment': {
                    'Type': 'String',
                    'Default': 'dev',
                    'AllowedValues': ['dev', 'staging', 'prod']
                },
                'DataBucketName': {
                    'Type': 'String',
                    'Default': 'predictive-maintenance-data'
                },
                'ModelsBucketName': {
                    'Type': 'String',
                    'Default': 'predictive-maintenance-models'
                },
                'StreamName': {
                    'Type': 'String',
                    'Default': 'sensor-data-stream'
                }
            },
            'Resources': {
                'DataBucket': {
                    'Type': 'AWS::S3::Bucket',
                    'Properties': {
                        'BucketName': {'Ref': 'DataBucketName'},
                        'VersioningConfiguration': {
                            'Status': 'Enabled'
                        },
                        'LifecycleConfiguration': {
                            'Rules': [
                                {
                                    'Id': 'DeleteOldVersions',
                                    'Status': 'Enabled',
                                    'NoncurrentVersionExpirationInDays': 30
                                }
                            ]
                        }
                    }
                },
                'ModelsBucket': {
                    'Type': 'AWS::S3::Bucket',
                    'Properties': {
                        'BucketName': {'Ref': 'ModelsBucketName'},
                        'VersioningConfiguration': {
                            'Status': 'Enabled'
                        }
                    }
                },
                'KinesisStream': {
                    'Type': 'AWS::Kinesis::Stream',
                    'Properties': {
                        'Name': {'Ref': 'StreamName'},
                        'ShardCount': 1,
                        'RetentionPeriodHours': 24
                    }
                },
                'LambdaExecutionRole': {
                    'Type': 'AWS::IAM::Role',
                    'Properties': {
                        'AssumeRolePolicyDocument': {
                            'Version': '2012-10-17',
                            'Statement': [
                                {
                                    'Effect': 'Allow',
                                    'Principal': {
                                        'Service': 'lambda.amazonaws.com'
                                    },
                                    'Action': 'sts:AssumeRole'
                                }
                            ]
                        },
                        'ManagedPolicyArns': [
                            'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
                        ],
                        'Policies': [
                            {
                                'PolicyName': 'PredictiveMaintenancePolicy',
                                'PolicyDocument': {
                                    'Version': '2012-10-17',
                                    'Statement': [
                                        {
                                            'Effect': 'Allow',
                                            'Action': [
                                                's3:GetObject',
                                                's3:PutObject',
                                                's3:DeleteObject'
                                            ],
                                            'Resource': [
                                                {'Fn::Sub': '${DataBucket}/*'},
                                                {'Fn::Sub': '${ModelsBucket}/*'}
                                            ]
                                        },
                                        {
                                            'Effect': 'Allow',
                                            'Action': [
                                                'kinesis:PutRecord',
                                                'kinesis:PutRecords',
                                                'kinesis:GetRecords',
                                                'kinesis:GetShardIterator'
                                            ],
                                            'Resource': {'Fn::GetAtt': ['KinesisStream', 'Arn']}
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                },
                'DataProcessorFunction': {
                    'Type': 'AWS::Lambda::Function',
                    'Properties': {
                        'FunctionName': 'predictive-maintenance-data-processor',
                        'Runtime': 'python3.9',
                        'Handler': 'lambda_handler',
                        'Role': {'Fn::GetAtt': ['LambdaExecutionRole', 'Arn']},
                        'Code': {
                            'ZipFile': 'def lambda_handler(event, context): return {"statusCode": 200}'
                        },
                        'Timeout': 300,
                        'MemorySize': 512,
                        'Environment': {
                            'Variables': {
                                'DATA_BUCKET': {'Ref': 'DataBucket'},
                                'MODELS_BUCKET': {'Ref': 'ModelsBucket'},
                                'STREAM_NAME': {'Ref': 'StreamName'}
                            }
                        }
                    }
                }
            },
            'Outputs': {
                'DataBucketName': {
                    'Value': {'Ref': 'DataBucket'},
                    'Description': 'Name of the data bucket'
                },
                'ModelsBucketName': {
                    'Value': {'Ref': 'ModelsBucket'},
                    'Description': 'Name of the models bucket'
                },
                'StreamName': {
                    'Value': {'Ref': 'KinesisStream'},
                    'Description': 'Name of the Kinesis stream'
                },
                'LambdaFunctionArn': {
                    'Value': {'Fn::GetAtt': ['DataProcessorFunction', 'Arn']},
                    'Description': 'ARN of the Lambda function'
                }
            }
        }
        
        # Save template
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False)
        
        logger.info(f"CloudFormation template generated: {output_path}")
    
    def generate_terraform_config(self, output_path: str = 'infrastructure/terraform/main.tf'):
        """
        Generate Terraform configuration for the predictive maintenance system.
        
        Args:
            output_path: Path to save the configuration
        """
        terraform_config = """
# Predictive Maintenance System Infrastructure
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "data_bucket_name" {
  description = "Name of the data bucket"
  type        = string
  default     = "predictive-maintenance-data"
}

variable "models_bucket_name" {
  description = "Name of the models bucket"
  type        = string
  default     = "predictive-maintenance-models"
}

variable "stream_name" {
  description = "Name of the Kinesis stream"
  type        = string
  default     = "sensor-data-stream"
}

# S3 Buckets
resource "aws_s3_bucket" "data_bucket" {
  bucket = var.data_bucket_name

  versioning {
    enabled = true
  }

  lifecycle_rule {
    id      = "delete_old_versions"
    enabled = true

    noncurrent_version_expiration {
      days = 30
    }
  }

  tags = {
    Name        = "Predictive Maintenance Data"
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "models_bucket" {
  bucket = var.models_bucket_name

  versioning {
    enabled = true
  }

  tags = {
    Name        = "Predictive Maintenance Models"
    Environment = var.environment
  }
}

# Kinesis Stream
resource "aws_kinesis_stream" "sensor_stream" {
  name             = var.stream_name
  shard_count      = 1
  retention_period = 24

  tags = {
    Name        = "Sensor Data Stream"
    Environment = var.environment
  }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_execution_role" {
  name = "predictive-maintenance-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda_execution_role.name
}

resource "aws_iam_role_policy" "lambda_predictive_maintenance" {
  name = "predictive-maintenance-policy"
  role = aws_iam_role.lambda_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.data_bucket.arn}/*",
          "${aws_s3_bucket.models_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kinesis:PutRecord",
          "kinesis:PutRecords",
          "kinesis:GetRecords",
          "kinesis:GetShardIterator"
        ]
        Resource = aws_kinesis_stream.sensor_stream.arn
      }
    ]
  })
}

# Lambda Function
resource "aws_lambda_function" "data_processor" {
  filename         = "lambda_function.zip"
  function_name    = "predictive-maintenance-data-processor"
  role            = aws_iam_role.lambda_execution_role.arn
  handler         = "lambda_handler"
  source_code_hash = filebase64sha256("lambda_function.zip")
  runtime         = "python3.9"
  timeout         = 300
  memory_size     = 512

  environment {
    variables = {
      DATA_BUCKET   = aws_s3_bucket.data_bucket.bucket
      MODELS_BUCKET = aws_s3_bucket.models_bucket.bucket
      STREAM_NAME   = aws_kinesis_stream.sensor_stream.name
    }
  }

  tags = {
    Name        = "Data Processor"
    Environment = var.environment
  }
}

# Outputs
output "data_bucket_name" {
  description = "Name of the data bucket"
  value       = aws_s3_bucket.data_bucket.bucket
}

output "models_bucket_name" {
  description = "Name of the models bucket"
  value       = aws_s3_bucket.models_bucket.bucket
}

output "stream_name" {
  description = "Name of the Kinesis stream"
  value       = aws_kinesis_stream.sensor_stream.name
}

output "lambda_function_arn" {
  description = "ARN of the Lambda function"
  value       = aws_lambda_function.data_processor.arn
}
"""
        
        # Save configuration
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(terraform_config)
        
        logger.info(f"Terraform configuration generated: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize deployment manager
    deployment_manager = AWSDeploymentManager()
    
    # Deploy complete system
    config = {
        'data_bucket': 'predictive-maintenance-data',
        'models_bucket': 'predictive-maintenance-models',
        'stream_name': 'sensor-data-stream',
        'shard_count': 1
    }
    
    success = deployment_manager.deploy_complete_system(config)
    print(f"System deployment: {'Success' if success else 'Failed'}")
    
    # Generate Infrastructure as Code
    iac = InfrastructureAsCode()
    iac.generate_cloudformation_template()
    iac.generate_terraform_config()
    
    print("Infrastructure as Code generated successfully")
