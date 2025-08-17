"""
Unit tests for data preprocessing module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_preprocessing import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(10, 2, n_samples),
            'feature2': np.random.uniform(0, 100, n_samples),
            'feature3': np.random.exponential(5, n_samples),
            'feature4': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.normal(50, 10, n_samples)
        })
        
        # Introduce some missing values
        self.sample_data.loc[5:9, 'feature1'] = np.nan
        self.sample_data.loc[15:17, 'feature2'] = np.nan
        self.sample_data.loc[25:26, 'feature4'] = np.nan
        
        # Introduce some outliers
        self.sample_data.loc[0, 'feature1'] = 100  # Outlier
        self.sample_data.loc[1, 'feature2'] = 1000  # Outlier
        
        self.X = self.sample_data.drop('target', axis=1)
        self.y = self.sample_data['target']
        
        self.preprocessor = DataPreprocessor(scaling_method="standard")
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        # Test default initialization
        preprocessor = DataPreprocessor()
        self.assertEqual(preprocessor.scaling_method, "standard")
        
        # Test with different scaling methods
        for method in ["standard", "robust", "minmax"]:
            preprocessor = DataPreprocessor(scaling_method=method)
            self.assertEqual(preprocessor.scaling_method, method)
        
        # Test invalid scaling method
        with self.assertRaises(ValueError):
            DataPreprocessor(scaling_method="invalid")
    
    def test_analyze_data_quality(self):
        """Test data quality analysis."""
        analysis = self.preprocessor.analyze_data_quality(self.sample_data)
        
        # Check analysis structure
        self.assertIn('shape', analysis)
        self.assertIn('missing_values', analysis)
        self.assertIn('missing_percentage', analysis)
        self.assertIn('duplicated_rows', analysis)
        self.assertIn('data_types', analysis)
        self.assertIn('numeric_summary', analysis)
        self.assertIn('outliers_iqr', analysis)
        
        # Check analysis values
        self.assertEqual(analysis['shape'], self.sample_data.shape)
        self.assertGreater(analysis['missing_values']['feature1'], 0)
        self.assertGreater(analysis['missing_values']['feature2'], 0)
        self.assertGreater(analysis['missing_values']['feature4'], 0)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Test with numeric data
        cleaned_data = self.preprocessor.handle_missing_values(self.X, strategy="median")
        
        # Check that missing values are handled
        self.assertEqual(cleaned_data.isnull().sum().sum(), 0)
        self.assertEqual(cleaned_data.shape, self.X.shape)
        
        # Test different strategies
        for strategy in ["mean", "median", "most_frequent"]:
            cleaned = self.preprocessor.handle_missing_values(self.X, strategy=strategy)
            self.assertEqual(cleaned.isnull().sum().sum(), 0)
    
    def test_detect_and_handle_outliers(self):
        """Test outlier detection and handling."""
        # Test IQR method
        cleaned_data, outlier_info = self.preprocessor.detect_and_handle_outliers(
            self.sample_data, method="iqr"
        )
        
        # Check outlier info structure
        self.assertIsInstance(outlier_info, dict)
        for col in self.sample_data.select_dtypes(include=[np.number]).columns:
            self.assertIn(col, outlier_info)
            self.assertIn('count', outlier_info[col])
            self.assertIn('percentage', outlier_info[col])
        
        # Check that outliers are detected
        self.assertGreater(outlier_info['feature1']['count'], 0)
        self.assertGreater(outlier_info['feature2']['count'], 0)
        
        # Test Z-score method
        cleaned_data_z, outlier_info_z = self.preprocessor.detect_and_handle_outliers(
            self.sample_data, method="zscore", factor=2.0
        )
        
        self.assertIsInstance(outlier_info_z, dict)
        self.assertEqual(cleaned_data_z.shape, self.sample_data.shape)
    
    def test_scale_features(self):
        """Test feature scaling."""
        # Create clean data for scaling test
        clean_X = self.preprocessor.handle_missing_values(self.X)
        numeric_X = clean_X.select_dtypes(include=[np.number])
        
        # Split data
        split_idx = len(numeric_X) // 2
        X_train = numeric_X.iloc[:split_idx]
        X_test = numeric_X.iloc[split_idx:]
        
        # Test scaling
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Check shapes
        self.assertEqual(X_train_scaled.shape, X_train.shape)
        self.assertEqual(X_test_scaled.shape, X_test.shape)
        
        # Check that scaling was applied (mean should be close to 0 for standard scaling)
        for col in X_train_scaled.columns:
            self.assertAlmostEqual(X_train_scaled[col].mean(), 0, places=10)
        
        # Test scaling without test set
        X_train_scaled_only, X_test_none = self.preprocessor.scale_features(X_train)
        self.assertIsNone(X_test_none)
        self.assertEqual(X_train_scaled_only.shape, X_train.shape)
    
    def test_split_data(self):
        """Test data splitting."""
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        # Check shapes
        expected_train_size = int(len(self.X) * 0.7)
        expected_test_size = len(self.X) - expected_train_size
        
        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)
        
        # Check that indices align
        pd.testing.assert_index_equal(X_train.index, y_train.index)
        pd.testing.assert_index_equal(X_test.index, y_test.index)
        
        # Test stratified split
        X_train_strat, X_test_strat, y_train_strat, y_test_strat = self.preprocessor.split_data(
            self.X, self.y, test_size=0.3, stratify=True, random_state=42
        )
        
        self.assertEqual(len(X_train_strat), expected_train_size)
        self.assertEqual(len(X_test_strat), expected_test_size)
    
    def test_preprocess_pipeline(self):
        """Test complete preprocessing pipeline."""
        results = self.preprocessor.preprocess_pipeline(
            self.X, self.y, test_size=0.3, handle_outliers=True
        )
        
        # Check result structure
        expected_keys = [
            'X_train', 'X_test', 'y_train', 'y_test',
            'X_train_unscaled', 'X_test_unscaled',
            'quality_analysis', 'outlier_info', 'preprocessing_info'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check data shapes
        total_samples = len(self.X)
        expected_train_size = int(total_samples * 0.7)
        expected_test_size = total_samples - expected_train_size
        
        self.assertEqual(len(results['X_train']), expected_train_size)
        self.assertEqual(len(results['X_test']), expected_test_size)
        self.assertEqual(len(results['y_train']), expected_train_size)
        self.assertEqual(len(results['y_test']), expected_test_size)
        
        # Check that preprocessing was applied
        self.assertEqual(results['X_train'].isnull().sum().sum(), 0)
        self.assertEqual(results['X_test'].isnull().sum().sum(), 0)
        
        # Check preprocessing info
        self.assertIn('scaling_method', results['preprocessing_info'])
        self.assertIn('test_size', results['preprocessing_info'])
        self.assertEqual(results['preprocessing_info']['scaling_method'], "standard")
        self.assertEqual(results['preprocessing_info']['test_size'], 0.3)
    
    def test_save_and_load_preprocessor(self):
        """Test saving and loading preprocessor."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            try:
                # Fit preprocessor
                clean_X = self.preprocessor.handle_missing_values(self.X)
                numeric_X = clean_X.select_dtypes(include=[np.number])
                self.preprocessor.scale_features(numeric_X)
                
                # Save preprocessor
                self.preprocessor.save_preprocessor(tmp_file.name)
                
                # Create new preprocessor and load
                new_preprocessor = DataPreprocessor()
                new_preprocessor.load_preprocessor(tmp_file.name)
                
                # Check that scaling method is preserved
                self.assertEqual(new_preprocessor.scaling_method, "standard")
                self.assertIsNotNone(new_preprocessor.scaler)
                
            finally:
                # Clean up
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class TestDataPreprocessorEdgeCases(unittest.TestCase):
    """Test edge cases for DataPreprocessor."""
    
    def test_no_missing_values(self):
        """Test preprocessing with no missing values."""
        # Create data without missing values
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.uniform(0, 10, 50)
        })
        
        preprocessor = DataPreprocessor()
        cleaned_data = preprocessor.handle_missing_values(data)
        
        # Data should remain unchanged
        pd.testing.assert_frame_equal(data, cleaned_data)
    
    def test_all_missing_values(self):
        """Test preprocessing with all missing values in a column."""
        # Create data with one column all NaN
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': [np.nan] * 50
        })
        
        preprocessor = DataPreprocessor()
        
        # This should handle gracefully
        cleaned_data = preprocessor.handle_missing_values(data)
        self.assertEqual(cleaned_data.shape, data.shape)
    
    def test_single_value_column(self):
        """Test preprocessing with constant column."""
        # Create data with constant column
        data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': [5.0] * 50  # Constant column
        })
        
        preprocessor = DataPreprocessor()
        
        # Should handle gracefully
        _, outlier_info = preprocessor.detect_and_handle_outliers(data)
        self.assertIsInstance(outlier_info, dict)


if __name__ == '__main__':
    unittest.main()