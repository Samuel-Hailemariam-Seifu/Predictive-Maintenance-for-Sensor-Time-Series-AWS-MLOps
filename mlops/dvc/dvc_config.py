"""
DVC Configuration and Setup

Configures DVC (Data Version Control) for data pipeline versioning
and reproducibility in the predictive maintenance system.
"""

import os
import yaml
import subprocess
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DVCManager:
    """Manages DVC operations for data versioning."""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize DVC manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.dvc_config_file = os.path.join(project_root, ".dvc", "config")
        self.dvcignore_file = os.path.join(project_root, ".dvcignore")
    
    def init_dvc(self, remote_url: str = None, remote_name: str = "origin") -> bool:
        """
        Initialize DVC in the project.
        
        Args:
            remote_url: URL of the remote storage
            remote_name: Name of the remote
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize DVC
            result = subprocess.run(
                ["dvc", "init"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error initializing DVC: {result.stderr}")
                return False
            
            logger.info("DVC initialized successfully")
            
            # Add remote if provided
            if remote_url:
                self.add_remote(remote_url, remote_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing DVC: {e}")
            return False
    
    def add_remote(self, remote_url: str, remote_name: str = "origin") -> bool:
        """
        Add a remote storage to DVC.
        
        Args:
            remote_url: URL of the remote storage
            remote_name: Name of the remote
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["dvc", "remote", "add", remote_name, remote_url],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error adding remote: {result.stderr}")
                return False
            
            logger.info(f"Added remote '{remote_name}': {remote_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding remote: {e}")
            return False
    
    def add_data(self, data_path: str, dvc_path: str = None) -> bool:
        """
        Add data to DVC tracking.
        
        Args:
            data_path: Path to the data file or directory
            dvc_path: DVC path (defaults to data_path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if dvc_path is None:
                dvc_path = data_path
            
            result = subprocess.run(
                ["dvc", "add", data_path],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error adding data: {result.stderr}")
                return False
            
            logger.info(f"Added data to DVC: {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding data: {e}")
            return False
    
    def push_data(self, remote_name: str = "origin") -> bool:
        """
        Push data to remote storage.
        
        Args:
            remote_name: Name of the remote
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["dvc", "push", "-r", remote_name],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error pushing data: {result.stderr}")
                return False
            
            logger.info(f"Pushed data to remote '{remote_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error pushing data: {e}")
            return False
    
    def pull_data(self, remote_name: str = "origin") -> bool:
        """
        Pull data from remote storage.
        
        Args:
            remote_name: Name of the remote
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["dvc", "pull", "-r", remote_name],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error pulling data: {result.stderr}")
                return False
            
            logger.info(f"Pulled data from remote '{remote_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error pulling data: {e}")
            return False
    
    def create_pipeline(self, pipeline_file: str = "dvc.yaml") -> bool:
        """
        Create a DVC pipeline configuration.
        
        Args:
            pipeline_file: Name of the pipeline file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pipeline_config = {
                "stages": {
                    "data_ingestion": {
                        "cmd": "python src/data_pipeline/data_ingestion.py",
                        "deps": ["src/data_pipeline/data_ingestion.py"],
                        "outs": ["data/raw/sensor_data.csv"],
                        "metrics": ["metrics/data_ingestion.json"]
                    },
                    "data_preprocessing": {
                        "cmd": "python src/data_pipeline/data_preprocessing.py",
                        "deps": ["data/raw/sensor_data.csv", "src/data_pipeline/data_preprocessing.py"],
                        "outs": ["data/processed/cleaned_data.csv", "data/processed/features.csv"],
                        "metrics": ["metrics/preprocessing.json"]
                    },
                    "model_training": {
                        "cmd": "python src/models/train_models.py",
                        "deps": ["data/processed/features.csv", "src/models/"],
                        "outs": ["models/anomaly_detection.pkl", "models/failure_prediction.pkl", "models/health_scoring.pkl"],
                        "metrics": ["metrics/model_performance.json"]
                    },
                    "model_evaluation": {
                        "cmd": "python src/models/evaluate_models.py",
                        "deps": ["models/", "data/processed/features.csv"],
                        "outs": ["metrics/evaluation.json"],
                        "metrics": ["metrics/evaluation.json"]
                    }
                }
            }
            
            # Write pipeline configuration
            pipeline_path = os.path.join(self.project_root, pipeline_file)
            with open(pipeline_path, 'w') as f:
                yaml.dump(pipeline_config, f, default_flow_style=False)
            
            logger.info(f"Created DVC pipeline: {pipeline_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating pipeline: {e}")
            return False
    
    def run_pipeline(self, stage: str = None) -> bool:
        """
        Run DVC pipeline.
        
        Args:
            stage: Specific stage to run (None for all stages)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ["dvc", "repro"]
            if stage:
                cmd.append(stage)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error running pipeline: {result.stderr}")
                return False
            
            logger.info(f"Pipeline {'stage' if stage else 'all stages'} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running pipeline: {e}")
            return False
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about tracked data.
        
        Returns:
            Dictionary with data information
        """
        try:
            result = subprocess.run(
                ["dvc", "status"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error getting data status: {result.stderr}")
                return {}
            
            # Parse status output
            status_info = {
                "status": result.stdout,
                "has_changes": "modified" in result.stdout or "new" in result.stdout
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"Error getting data info: {e}")
            return {}
    
    def create_dvcignore(self) -> bool:
        """
        Create .dvcignore file to exclude unnecessary files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            dvcignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp

# Model artifacts (tracked by DVC)
models/*.pkl
models/*.joblib
models/*.h5
models/*.pth

# Data files (tracked by DVC)
data/raw/
data/processed/
data/models/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# pytest
.pytest_cache/

# Coverage reports
htmlcov/
.coverage
coverage.xml

# mypy
.mypy_cache/
.dmypy.json
dmypy.json
"""
            
            with open(self.dvcignore_file, 'w') as f:
                f.write(dvcignore_content)
            
            logger.info(f"Created .dvcignore file: {self.dvcignore_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating .dvcignore: {e}")
            return False
    
    def setup_data_pipeline(self) -> bool:
        """
        Set up the complete data pipeline with DVC.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create .dvcignore
            self.create_dvcignore()
            
            # Create pipeline
            self.create_pipeline()
            
            # Add data directories
            data_dirs = [
                "data/raw",
                "data/processed",
                "data/models",
                "metrics"
            ]
            
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    self.add_data(data_dir)
            
            logger.info("Data pipeline setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up data pipeline: {e}")
            return False


class DataVersioning:
    """Handles data versioning operations."""
    
    def __init__(self, dvc_manager: DVCManager):
        """
        Initialize data versioning.
        
        Args:
            dvc_manager: DVCManager instance
        """
        self.dvc_manager = dvc_manager
    
    def version_data(self, data_path: str, version_message: str) -> bool:
        """
        Version data with a commit message.
        
        Args:
            data_path: Path to the data
            version_message: Version message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add data to DVC
            if not self.dvc_manager.add_data(data_path):
                return False
            
            # Commit changes
            result = subprocess.run(
                ["git", "add", f"{data_path}.dvc"],
                cwd=self.dvc_manager.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error adding .dvc file: {result.stderr}")
                return False
            
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", version_message],
                cwd=self.dvc_manager.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error committing: {result.stderr}")
                return False
            
            logger.info(f"Versioned data: {data_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error versioning data: {e}")
            return False
    
    def get_data_versions(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Get version history for data.
        
        Args:
            data_path: Path to the data
            
        Returns:
            List of version information
        """
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", f"{data_path}.dvc"],
                cwd=self.dvc_manager.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error getting data versions: {result.stderr}")
                return []
            
            versions = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        versions.append({
                            'commit': parts[0],
                            'message': parts[1]
                        })
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting data versions: {e}")
            return []
    
    def checkout_data_version(self, data_path: str, commit_hash: str) -> bool:
        """
        Checkout a specific version of data.
        
        Args:
            data_path: Path to the data
            commit_hash: Commit hash to checkout
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Checkout specific commit
            result = subprocess.run(
                ["git", "checkout", commit_hash, f"{data_path}.dvc"],
                cwd=self.dvc_manager.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Error checking out version: {result.stderr}")
                return False
            
            # Pull data for this version
            if not self.dvc_manager.pull_data():
                return False
            
            logger.info(f"Checked out data version: {commit_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking out data version: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize DVC manager
    dvc_manager = DVCManager()
    
    # Initialize DVC
    if dvc_manager.init_dvc():
        print("DVC initialized successfully")
    
    # Add S3 remote
    s3_remote = "s3://predictive-maintenance-data/dvc-storage"
    if dvc_manager.add_remote(s3_remote, "s3"):
        print("S3 remote added successfully")
    
    # Set up data pipeline
    if dvc_manager.setup_data_pipeline():
        print("Data pipeline setup completed")
    
    # Add sample data
    if os.path.exists("data/raw"):
        dvc_manager.add_data("data/raw")
        print("Raw data added to DVC")
    
    # Push data to remote
    if dvc_manager.push_data("s3"):
        print("Data pushed to S3")
    
    # Run pipeline
    if dvc_manager.run_pipeline():
        print("Pipeline executed successfully")
    
    # Data versioning
    data_versioning = DataVersioning(dvc_manager)
    
    # Version data
    if data_versioning.version_data("data/raw", "Initial sensor data"):
        print("Data versioned successfully")
    
    # Get data versions
    versions = data_versioning.get_data_versions("data/raw")
    print(f"Data versions: {versions}")
    
    # Get data info
    info = dvc_manager.get_data_info()
    print(f"Data info: {info}")
