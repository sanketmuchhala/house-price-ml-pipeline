"""
Data ingestion module for California housing dataset.
"""

import os
import logging
import pandas as pd
from sklearn.datasets import fetch_california_housing
from typing import Tuple, Optional
import joblib


class DataIngestion:
    """
    Handles data ingestion for the California housing dataset.
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataIngestion class.
        
        Args:
            data_dir (str): Directory to save raw data
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_data(self, save_to_disk: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Download California housing dataset from sklearn.
        
        Args:
            save_to_disk (bool): Whether to save data to disk
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target data
        """
        try:
            self.logger.info("Downloading California housing dataset...")
            
            # Fetch data from sklearn
            housing_data = fetch_california_housing(as_frame=True)
            
            # Extract features and target
            X = housing_data.data
            y = housing_data.target
            
            # Add target to features for easier handling
            data = X.copy()
            data['target'] = y
            
            self.logger.info(f"Dataset downloaded successfully. Shape: {data.shape}")
            self.logger.info(f"Features: {list(X.columns)}")
            
            if save_to_disk:
                self._save_data(data, X, y)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def _save_data(self, data: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Save data to disk in multiple formats.
        
        Args:
            data (pd.DataFrame): Combined dataset
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
        """
        try:
            # Save complete dataset as CSV
            data_path = os.path.join(self.data_dir, "california_housing.csv")
            data.to_csv(data_path, index=False)
            self.logger.info(f"Complete dataset saved to {data_path}")
            
            # Save features and target separately as pickle for faster loading
            features_path = os.path.join(self.data_dir, "features.pkl")
            target_path = os.path.join(self.data_dir, "target.pkl")
            
            joblib.dump(X, features_path)
            joblib.dump(y, target_path)
            
            self.logger.info(f"Features saved to {features_path}")
            self.logger.info(f"Target saved to {target_path}")
            
            # Save dataset info
            self._save_dataset_info(X, y)
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
    
    def _save_dataset_info(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Save dataset information and feature descriptions.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
        """
        info = {
            'dataset_name': 'California Housing Dataset',
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_names': list(X.columns),
            'target_name': 'median_house_value',
            'feature_descriptions': {
                'MedInc': 'Median income in block group',
                'HouseAge': 'Median house age in block group',
                'AveRooms': 'Average number of rooms per household',
                'AveBedrms': 'Average number of bedrooms per household',
                'Population': 'Block group population',
                'AveOccup': 'Average number of household members',
                'Latitude': 'Block group latitude',
                'Longitude': 'Block group longitude'
            },
            'target_description': 'Median house value for California districts (in hundreds of thousands of dollars)',
            'data_source': 'sklearn.datasets.fetch_california_housing'
        }
        
        info_path = os.path.join(self.data_dir, "dataset_info.pkl")
        joblib.dump(info, info_path)
        self.logger.info(f"Dataset info saved to {info_path}")
    
    def load_data(self, format_type: str = "pickle") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load data from disk.
        
        Args:
            format_type (str): Format to load ('pickle' or 'csv')
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target data
        """
        try:
            if format_type == "pickle":
                features_path = os.path.join(self.data_dir, "features.pkl")
                target_path = os.path.join(self.data_dir, "target.pkl")
                
                if not os.path.exists(features_path) or not os.path.exists(target_path):
                    self.logger.warning("Pickle files not found. Downloading data...")
                    return self.download_data()
                
                X = joblib.load(features_path)
                y = joblib.load(target_path)
                
            elif format_type == "csv":
                data_path = os.path.join(self.data_dir, "california_housing.csv")
                
                if not os.path.exists(data_path):
                    self.logger.warning("CSV file not found. Downloading data...")
                    return self.download_data()
                
                data = pd.read_csv(data_path)
                X = data.drop('target', axis=1)
                y = data['target']
            
            else:
                raise ValueError("format_type must be 'pickle' or 'csv'")
            
            self.logger.info(f"Data loaded successfully from {format_type} format")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_dataset_info(self) -> Optional[dict]:
        """
        Get dataset information.
        
        Returns:
            dict: Dataset information
        """
        try:
            info_path = os.path.join(self.data_dir, "dataset_info.pkl")
            
            if not os.path.exists(info_path):
                self.logger.warning("Dataset info not found. Downloading data first...")
                self.download_data()
            
            return joblib.load(info_path)
            
        except Exception as e:
            self.logger.error(f"Error loading dataset info: {str(e)}")
            return None


def main():
    """
    Main function to demonstrate data ingestion.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Download and save data
    X, y = data_ingestion.download_data()
    
    # Display basic info
    print(f"Dataset shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")


if __name__ == "__main__":
    main()