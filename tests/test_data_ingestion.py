"""
Unit tests for data ingestion module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_ingestion import DataIngestion


class TestDataIngestion(unittest.TestCase):
    """Test cases for DataIngestion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_ingestion = DataIngestion(data_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test DataIngestion initialization."""
        self.assertEqual(self.data_ingestion.data_dir, self.test_dir)
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_download_data(self):
        """Test data downloading functionality."""
        # Download data
        X, y = self.data_ingestion.download_data(save_to_disk=True)
        
        # Check data types and shapes
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)
        
        # Check expected columns
        expected_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                          'Population', 'AveOccup', 'Latitude', 'Longitude']
        for col in expected_columns:
            self.assertIn(col, X.columns)
        
        # Check data ranges (basic sanity checks)
        self.assertTrue(X['MedInc'].min() >= 0)
        self.assertTrue(X['HouseAge'].min() >= 0)
        self.assertTrue(X['AveRooms'].min() > 0)
        self.assertTrue(y.min() > 0)
    
    def test_save_data(self):
        """Test data saving functionality."""
        # Download and save data
        X, y = self.data_ingestion.download_data(save_to_disk=True)
        
        # Check if files are created
        expected_files = [
            'california_housing.csv',
            'features.pkl',
            'target.pkl',
            'dataset_info.pkl'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"File {filename} not created")
    
    def test_load_data_pickle(self):
        """Test loading data from pickle format."""
        # First download and save data
        original_X, original_y = self.data_ingestion.download_data(save_to_disk=True)
        
        # Load data using pickle format
        loaded_X, loaded_y = self.data_ingestion.load_data(format_type="pickle")
        
        # Compare data
        pd.testing.assert_frame_equal(original_X, loaded_X)
        pd.testing.assert_series_equal(original_y, loaded_y)
    
    def test_load_data_csv(self):
        """Test loading data from CSV format."""
        # First download and save data
        original_X, original_y = self.data_ingestion.download_data(save_to_disk=True)
        
        # Load data using CSV format
        loaded_X, loaded_y = self.data_ingestion.load_data(format_type="csv")
        
        # Compare shapes (CSV loading might have slight differences due to serialization)
        self.assertEqual(original_X.shape, loaded_X.shape)
        self.assertEqual(len(original_y), len(loaded_y))
        
        # Compare column names
        self.assertEqual(list(original_X.columns), list(loaded_X.columns))
    
    def test_get_dataset_info(self):
        """Test dataset info retrieval."""
        # Download data first
        self.data_ingestion.download_data(save_to_disk=True)
        
        # Get dataset info
        info = self.data_ingestion.get_dataset_info()
        
        # Check info structure
        self.assertIsInstance(info, dict)
        self.assertIn('dataset_name', info)
        self.assertIn('n_samples', info)
        self.assertIn('n_features', info)
        self.assertIn('feature_names', info)
        self.assertIn('target_name', info)
        self.assertIn('feature_descriptions', info)
        
        # Check info values
        self.assertGreater(info['n_samples'], 0)
        self.assertEqual(info['n_features'], 8)
        self.assertEqual(len(info['feature_names']), 8)
    
    def test_load_nonexistent_data(self):
        """Test loading data when files don't exist."""
        # This should trigger downloading
        X, y = self.data_ingestion.load_data()
        
        # Should successfully return data
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertGreater(len(X), 0)
    
    def test_invalid_format_type(self):
        """Test invalid format type for loading data."""
        with self.assertRaises(ValueError):
            self.data_ingestion.load_data(format_type="invalid_format")


class TestDataIngestionIntegration(unittest.TestCase):
    """Integration tests for DataIngestion class."""
    
    def test_full_workflow(self):
        """Test the complete data ingestion workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize ingestion
            ingestion = DataIngestion(data_dir=temp_dir)
            
            # Download data
            X, y = ingestion.download_data(save_to_disk=True)
            
            # Load data back
            loaded_X, loaded_y = ingestion.load_data()
            
            # Get info
            info = ingestion.get_dataset_info()
            
            # Verify workflow
            pd.testing.assert_frame_equal(X, loaded_X)
            pd.testing.assert_series_equal(y, loaded_y)
            self.assertIsInstance(info, dict)
            self.assertEqual(info['n_samples'], len(X))


if __name__ == '__main__':
    unittest.main()