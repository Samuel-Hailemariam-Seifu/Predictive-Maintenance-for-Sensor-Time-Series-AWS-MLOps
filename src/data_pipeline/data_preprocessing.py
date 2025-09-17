"""
Data Preprocessing Module

Handles data cleaning, feature engineering, and preparation for machine learning models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main class for data preprocessing operations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary for preprocessing parameters
        """
        self.config = config or {}
        self.scalers = {}
        self.imputers = {}
        self.feature_columns = []
        self.target_columns = []
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw sensor data.
        
        Args:
            df: Raw sensor data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning process")
        original_shape = df.shape
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        # Validate data types
        df = self._validate_data_types(df)
        
        logger.info(f"Data cleaning completed. Shape: {original_shape} -> {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_threshold = self.config.get('missing_threshold', 0.5)
        
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
        df = df.drop(columns=columns_to_drop)
        
        if len(columns_to_drop) > 0:
            logger.info(f"Dropped columns with >{missing_threshold*100}% missing values: {list(columns_to_drop)}")
        
        # Impute remaining missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Use KNN imputation for numeric columns
        if len(numeric_columns) > 0:
            knn_imputer = KNNImputer(n_neighbors=5)
            df[numeric_columns] = knn_imputer.fit_transform(df[numeric_columns])
            self.imputers['numeric'] = knn_imputer
        
        # Use mode imputation for categorical columns
        if len(categorical_columns) > 0:
            mode_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = mode_imputer.fit_transform(df[categorical_columns])
            self.imputers['categorical'] = mode_imputer
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_threshold = self.config.get('outlier_threshold', 1.5)
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            df = df[~outliers]
            
            if outliers.sum() > 0:
                logger.info(f"Removed {outliers.sum()} outliers from column {column}")
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure numeric columns are numeric
        numeric_columns = ['value', 'temperature', 'rpm', 'load']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for predictive maintenance.
        
        Args:
            df: Cleaned sensor data DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Sort by timestamp and equipment_id
        df = df.sort_values(['equipment_id', 'timestamp'])
        
        # Time-based features
        df = self._add_time_features(df)
        
        # Rolling statistics
        df = self._add_rolling_features(df)
        
        # Equipment-specific features
        df = self._add_equipment_features(df)
        
        # Sensor correlation features
        df = self._add_correlation_features(df)
        
        # Health indicators
        df = self._add_health_indicators(df)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in df.columns:
            return df
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].isin(range(22, 24)) | df['hour'].isin(range(0, 6))
        df['is_night'] = df['is_night'].astype(int)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features."""
        numeric_columns = ['value', 'temperature', 'rpm', 'load']
        window_sizes = [5, 10, 30, 60]  # minutes
        
        for equipment_id in df['equipment_id'].unique():
            equipment_mask = df['equipment_id'] == equipment_id
            equipment_data = df[equipment_mask].copy()
            
            for col in numeric_columns:
                if col in equipment_data.columns:
                    for window in window_sizes:
                        # Rolling mean
                        df.loc[equipment_mask, f'{col}_mean_{window}'] = (
                            equipment_data[col].rolling(window=window, min_periods=1).mean()
                        )
                        
                        # Rolling std
                        df.loc[equipment_mask, f'{col}_std_{window}'] = (
                            equipment_data[col].rolling(window=window, min_periods=1).std()
                        )
                        
                        # Rolling max
                        df.loc[equipment_mask, f'{col}_max_{window}'] = (
                            equipment_data[col].rolling(window=window, min_periods=1).max()
                        )
                        
                        # Rolling min
                        df.loc[equipment_mask, f'{col}_min_{window}'] = (
                            equipment_data[col].rolling(window=window, min_periods=1).min()
                        )
        
        return df
    
    def _add_equipment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add equipment-specific features."""
        # Equipment age (if start_date is available)
        if 'start_date' in df.columns:
            df['equipment_age_days'] = (df['timestamp'] - df['start_date']).dt.days
        
        # Operating hours
        df['operating_hours'] = df.groupby('equipment_id')['timestamp'].rank(method='first')
        
        # Maintenance cycles (if maintenance data is available)
        if 'last_maintenance' in df.columns:
            df['days_since_maintenance'] = (df['timestamp'] - df['last_maintenance']).dt.days
        
        return df
    
    def _add_correlation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on sensor correlations."""
        numeric_columns = ['value', 'temperature', 'rpm', 'load']
        available_columns = [col for col in numeric_columns if col in df.columns]
        
        if len(available_columns) >= 2:
            # Calculate correlation between sensors
            for i, col1 in enumerate(available_columns):
                for col2 in available_columns[i+1:]:
                    df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                    df[f'{col1}_{col2}_diff'] = df[col1] - df[col2]
        
        return df
    
    def _add_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add health indicator features."""
        # Vibration health score
        if 'value' in df.columns:
            # Normalize vibration values (assuming 0-1 scale)
            df['vibration_health'] = 1 - np.clip(df['value'], 0, 1)
        
        # Temperature health score
        if 'temperature' in df.columns:
            # Assume normal operating temperature is 20-80°C
            normal_temp_min, normal_temp_max = 20, 80
            df['temperature_health'] = 1 - np.clip(
                (df['temperature'] - normal_temp_min) / (normal_temp_max - normal_temp_min), 0, 1
            )
        
        # Overall health score (weighted average)
        health_columns = [col for col in df.columns if col.endswith('_health')]
        if health_columns:
            df['overall_health'] = df[health_columns].mean(axis=1)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale features for machine learning.
        
        Args:
            df: DataFrame with features to scale
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {method} method")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['timestamp', 'equipment_id']]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df_scaled = df.copy()
        df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        self.scalers[method] = scaler
        self.feature_columns = feature_columns
        
        return df_scaled
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for predictive maintenance.
        
        Args:
            df: DataFrame with sensor data
            
        Returns:
            DataFrame with target variables
        """
        logger.info("Creating target variables")
        
        # Anomaly detection target (binary)
        df['is_anomaly'] = 0
        
        # Failure prediction target (binary)
        df['will_fail'] = 0
        
        # Health score target (continuous)
        if 'overall_health' in df.columns:
            df['health_score'] = df['overall_health']
        else:
            # Create a simple health score based on available metrics
            health_metrics = ['value', 'temperature', 'rpm', 'load']
            available_metrics = [col for col in health_metrics if col in df.columns]
            
            if available_metrics:
                # Normalize and combine metrics
                normalized_metrics = []
                for col in available_metrics:
                    normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    normalized_metrics.append(1 - normalized)  # Invert so higher is better
                
                df['health_score'] = np.mean(normalized_metrics, axis=0)
            else:
                df['health_score'] = 0.5  # Default neutral score
        
        # Maintenance urgency (categorical)
        df['maintenance_urgency'] = 'low'
        df.loc[df['health_score'] < 0.3, 'maintenance_urgency'] = 'high'
        df.loc[(df['health_score'] >= 0.3) & (df['health_score'] < 0.6), 'maintenance_urgency'] = 'medium'
        
        self.target_columns = ['is_anomaly', 'will_fail', 'health_score', 'maintenance_urgency']
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for machine learning.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        logger.info("Preparing data for machine learning")
        
        # Select feature columns
        feature_columns = [col for col in df.columns if col not in [
            'timestamp', 'equipment_id', 'is_anomaly', 'will_fail', 
            'health_score', 'maintenance_urgency'
        ]]
        
        features_df = df[feature_columns].copy()
        targets_df = df[self.target_columns].copy()
        
        # Handle any remaining missing values
        features_df = features_df.fillna(features_df.median())
        targets_df = targets_df.fillna(0)
        
        logger.info(f"ML data prepared. Features: {features_df.shape}, Targets: {targets_df.shape}")
        
        return features_df, targets_df
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate feature importance using correlation with target variables.
        
        Args:
            df: DataFrame with features and targets
            
        Returns:
            DataFrame with feature importance scores
        """
        feature_columns = [col for col in df.columns if col not in [
            'timestamp', 'equipment_id', 'is_anomaly', 'will_fail', 
            'health_score', 'maintenance_urgency'
        ]]
        
        target_columns = ['health_score']
        importance_scores = []
        
        for feature in feature_columns:
            if feature in df.columns:
                correlations = []
                for target in target_columns:
                    if target in df.columns:
                        corr = abs(df[feature].corr(df[target]))
                        correlations.append(corr)
                
                avg_correlation = np.mean(correlations) if correlations else 0
                importance_scores.append({
                    'feature': feature,
                    'importance': avg_correlation,
                    'type': 'numeric' if df[feature].dtype in ['int64', 'float64'] else 'categorical'
                })
        
        return pd.DataFrame(importance_scores).sort_values('importance', ascending=False)


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'equipment_id': np.random.choice(['MOTOR_001', 'MOTOR_002', 'PUMP_001'], n_samples),
        'sensor_type': np.random.choice(['vibration', 'temperature', 'pressure'], n_samples),
        'value': np.random.normal(0.5, 0.1, n_samples),
        'temperature': np.random.normal(50, 10, n_samples),
        'rpm': np.random.normal(1800, 100, n_samples),
        'load': np.random.normal(75, 15, n_samples),
        'unit': 'g',
        'location': 'bearing_1'
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize preprocessor
    config = {
        'missing_threshold': 0.3,
        'outlier_threshold': 1.5
    }
    preprocessor = DataPreprocessor(config)
    
    # Process the data
    df_cleaned = preprocessor.clean_data(df)
    df_features = preprocessor.engineer_features(df_cleaned)
    df_scaled = preprocessor.scale_features(df_features)
    df_with_targets = preprocessor.create_target_variables(df_scaled)
    
    # Prepare for ML
    features_df, targets_df = preprocessor.prepare_ml_data(df_with_targets)
    
    print("Data preprocessing completed!")
    print(f"Features shape: {features_df.shape}")
    print(f"Targets shape: {targets_df.shape}")
    print(f"Feature columns: {list(features_df.columns)}")
    print(f"Target columns: {list(targets_df.columns)}")
