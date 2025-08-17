"""
Unit tests for feature engineering module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features.feature_engineering import FeatureEngineer, FeatureSelector, FeaturePipeline


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample California housing-like data
        np.random.seed(42)
        n_samples = 100
        
        self.sample_data = pd.DataFrame({
            'MedInc': np.random.uniform(0.5, 15, n_samples),
            'HouseAge': np.random.uniform(1, 52, n_samples),
            'AveRooms': np.random.uniform(3, 10, n_samples),
            'AveBedrms': np.random.uniform(0.8, 2.5, n_samples),
            'Population': np.random.uniform(100, 5000, n_samples),
            'AveOccup': np.random.uniform(2, 6, n_samples),
            'Latitude': np.random.uniform(32, 42, n_samples),
            'Longitude': np.random.uniform(-124, -114, n_samples)
        })
        
        self.target = np.random.uniform(0.5, 5, n_samples)
        
        self.feature_engineer = FeatureEngineer(
            create_interaction_features=True,
            create_polynomial_features=False
        )
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        # Test default initialization
        engineer = FeatureEngineer()
        self.assertTrue(engineer.create_interaction_features)
        self.assertFalse(engineer.create_polynomial_features)
        self.assertEqual(engineer.polynomial_degree, 2)
        
        # Test custom initialization
        engineer = FeatureEngineer(
            create_interaction_features=False,
            create_polynomial_features=True,
            polynomial_degree=3
        )
        self.assertFalse(engineer.create_interaction_features)
        self.assertTrue(engineer.create_polynomial_features)
        self.assertEqual(engineer.polynomial_degree, 3)
    
    def test_fit_transform(self):
        """Test fit and transform methods."""
        # Fit the engineer
        self.feature_engineer.fit(self.sample_data)
        self.assertEqual(self.feature_engineer.feature_names_, list(self.sample_data.columns))
        
        # Transform the data
        transformed_data = self.feature_engineer.transform(self.sample_data)
        
        # Check that new features were created
        self.assertGreater(transformed_data.shape[1], self.sample_data.shape[1])
        self.assertEqual(transformed_data.shape[0], self.sample_data.shape[0])
        
        # Check that original features are preserved
        for col in self.sample_data.columns:
            self.assertIn(col, transformed_data.columns)
    
    def test_housing_features_creation(self):
        """Test creation of housing-specific features."""
        transformed_data = self.feature_engineer.fit_transform(self.sample_data)
        
        # Check for expected new features
        expected_features = [
            'RoomDensity', 'BedroomRatio', 'PopulationDensity',
            'IncomePerPerson', 'IsNewHouse', 'IsOldHouse',
            'DistanceToSF', 'DistanceToLA', 'CoastalProximity',
            'IsNorthernCA', 'HighIncome', 'LowIncome'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, transformed_data.columns)
        
        # Check feature value ranges
        self.assertTrue((transformed_data['BedroomRatio'] >= 0).all())
        self.assertTrue((transformed_data['BedroomRatio'] <= 1).all())
        self.assertTrue((transformed_data['IsNewHouse'].isin([0, 1])).all())
        self.assertTrue((transformed_data['IsOldHouse'].isin([0, 1])).all())
        self.assertTrue((transformed_data['IsNorthernCA'].isin([0, 1])).all())
    
    def test_interaction_features_creation(self):
        """Test creation of interaction features."""
        engineer = FeatureEngineer(create_interaction_features=True)
        transformed_data = engineer.fit_transform(self.sample_data)
        
        # Check for interaction features
        interaction_features = [col for col in transformed_data.columns if '_x_' in col]
        self.assertGreater(len(interaction_features), 0)
        
        # Test specific interactions
        if 'MedInc_x_AveRooms' in transformed_data.columns:
            # Verify interaction calculation
            expected = self.sample_data['MedInc'] * self.sample_data['AveRooms']
            actual = transformed_data['MedInc_x_AveRooms']
            np.testing.assert_array_almost_equal(expected.values, actual.values)
    
    def test_polynomial_features_creation(self):
        """Test creation of polynomial features."""
        engineer = FeatureEngineer(
            create_interaction_features=False,
            create_polynomial_features=True,
            polynomial_degree=3
        )
        transformed_data = engineer.fit_transform(self.sample_data)
        
        # Check for polynomial features
        poly_features = [col for col in transformed_data.columns if '_poly_' in col]
        self.assertGreater(len(poly_features), 0)
        
        # Test specific polynomial
        if 'MedInc_poly_2' in transformed_data.columns:
            expected = self.sample_data['MedInc'] ** 2
            actual = transformed_data['MedInc_poly_2']
            np.testing.assert_array_almost_equal(expected.values, actual.values)


class TestFeatureSelector(unittest.TestCase):
    """Test cases for FeatureSelector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        # Create sample data with some redundant features
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target that's correlated with some features
        self.y = (self.X['feature_0'] * 2 + 
                 self.X['feature_1'] * 1.5 + 
                 self.X['feature_2'] * 0.5 + 
                 np.random.randn(n_samples) * 0.1)
    
    def test_initialization(self):
        """Test FeatureSelector initialization."""
        # Test default initialization
        selector = FeatureSelector()
        self.assertEqual(selector.selection_method, "mutual_info")
        self.assertIsNone(selector.k_features)
        
        # Test custom initialization
        selector = FeatureSelector(selection_method="f_test", k_features=10)
        self.assertEqual(selector.selection_method, "f_test")
        self.assertEqual(selector.k_features, 10)
    
    def test_f_test_selection(self):
        """Test F-test feature selection."""
        selector = FeatureSelector(selection_method="f_test", k_features=5)
        
        # Fit and transform
        X_selected = selector.fit_transform(self.X, self.y)
        
        # Check results
        self.assertEqual(X_selected.shape[1], 5)
        self.assertEqual(X_selected.shape[0], self.X.shape[0])
        self.assertIsNotNone(selector.selected_features_)
        self.assertEqual(len(selector.selected_features_), 5)
        
        # Check that most important features are selected
        self.assertIn('feature_0', selector.selected_features_)  # Should be selected (highest correlation)
    
    def test_rfe_selection(self):
        """Test RFE feature selection."""
        selector = FeatureSelector(selection_method="rfe", k_features=8)
        
        # Fit and transform
        X_selected = selector.fit_transform(self.X, self.y)
        
        # Check results
        self.assertEqual(X_selected.shape[1], 8)
        self.assertIsNotNone(selector.selected_features_)
        self.assertEqual(len(selector.selected_features_), 8)
    
    def test_lasso_selection(self):
        """Test Lasso-based feature selection."""
        selector = FeatureSelector(selection_method="lasso", k_features=6)
        
        # Fit and transform
        X_selected = selector.fit_transform(self.X, self.y)
        
        # Check results
        self.assertLessEqual(X_selected.shape[1], 6)  # Lasso might select fewer features
        self.assertIsNotNone(selector.selected_features_)
    
    def test_get_feature_importance(self):
        """Test feature importance retrieval."""
        selector = FeatureSelector(selection_method="f_test", k_features=10)
        selector.fit(self.X, self.y)
        
        importance = selector.get_feature_importance()
        self.assertIsInstance(importance, dict)
        self.assertGreater(len(importance), 0)
    
    def test_invalid_selection_method(self):
        """Test invalid selection method."""
        with self.assertRaises(ValueError):
            selector = FeatureSelector(selection_method="invalid_method")
            selector.fit(self.X, self.y)


class TestFeaturePipeline(unittest.TestCase):
    """Test cases for FeaturePipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 100
        
        # Create California housing-like data
        self.X = pd.DataFrame({
            'MedInc': np.random.uniform(0.5, 15, n_samples),
            'HouseAge': np.random.uniform(1, 52, n_samples),
            'AveRooms': np.random.uniform(3, 10, n_samples),
            'AveBedrms': np.random.uniform(0.8, 2.5, n_samples),
            'Population': np.random.uniform(100, 5000, n_samples),
            'AveOccup': np.random.uniform(2, 6, n_samples),
            'Latitude': np.random.uniform(32, 42, n_samples),
            'Longitude': np.random.uniform(-124, -114, n_samples)
        })
        
        self.y = pd.Series(np.random.uniform(0.5, 5, n_samples))
    
    def test_initialization(self):
        """Test FeaturePipeline initialization."""
        # Test default initialization
        pipeline = FeaturePipeline()
        self.assertIsNotNone(pipeline.feature_engineer)
        self.assertIsNotNone(pipeline.feature_selector)
        self.assertFalse(pipeline.is_fitted)
        
        # Test custom initialization
        engineer_config = {'create_interaction_features': False}
        selector_config = {'selection_method': 'rfe', 'k_features': 10}
        
        pipeline = FeaturePipeline(
            engineer_config=engineer_config,
            selector_config=selector_config
        )
        self.assertFalse(pipeline.feature_engineer.create_interaction_features)
        self.assertEqual(pipeline.feature_selector.selection_method, 'rfe')
        self.assertEqual(pipeline.feature_selector.k_features, 10)
    
    def test_fit_transform(self):
        """Test pipeline fit and transform."""
        pipeline = FeaturePipeline()
        
        # Fit and transform
        X_transformed = pipeline.fit_transform(self.X, self.y)
        
        # Check results
        self.assertTrue(pipeline.is_fitted)
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        
        # Should have fewer features due to selection
        # But more than original due to engineering
        self.assertNotEqual(X_transformed.shape[1], self.X.shape[1])
    
    def test_transform_only(self):
        """Test transform on fitted pipeline."""
        pipeline = FeaturePipeline()
        
        # Fit first
        pipeline.fit(self.X, self.y)
        
        # Transform
        X_transformed = pipeline.transform(self.X)
        
        self.assertIsInstance(X_transformed, pd.DataFrame)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
    
    def test_transform_without_fit(self):
        """Test transform without fitting."""
        pipeline = FeaturePipeline()
        
        with self.assertRaises(ValueError):
            pipeline.transform(self.X)
    
    def test_get_feature_info(self):
        """Test feature info retrieval."""
        pipeline = FeaturePipeline()
        
        # Before fitting
        info = pipeline.get_feature_info()
        self.assertIn('error', info)
        
        # After fitting
        pipeline.fit(self.X, self.y)
        info = pipeline.get_feature_info()
        
        self.assertIn('original_features', info)
        self.assertIn('selected_features', info)
        self.assertIn('feature_scores', info)
        self.assertIn('selection_method', info)
        self.assertIn('n_original_features', info)
        self.assertIn('n_selected_features', info)
        
        self.assertEqual(info['original_features'], list(self.X.columns))
        self.assertGreater(info['n_selected_features'], 0)
    
    def test_save_and_load_pipeline(self):
        """Test saving and loading pipeline."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            try:
                # Fit pipeline
                pipeline = FeaturePipeline()
                pipeline.fit(self.X, self.y)
                
                # Save pipeline
                pipeline.save_pipeline(tmp_file.name)
                
                # Create new pipeline and load
                new_pipeline = FeaturePipeline()
                new_pipeline.load_pipeline(tmp_file.name)
                
                # Check that pipeline is loaded correctly
                self.assertTrue(new_pipeline.is_fitted)
                
                # Test that transform works
                X_transformed_original = pipeline.transform(self.X)
                X_transformed_loaded = new_pipeline.transform(self.X)
                
                pd.testing.assert_frame_equal(X_transformed_original, X_transformed_loaded)
                
            finally:
                # Clean up
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)


class TestFeatureEngineeringIntegration(unittest.TestCase):
    """Integration tests for feature engineering components."""
    
    def test_full_feature_pipeline(self):
        """Test complete feature engineering workflow."""
        np.random.seed(42)
        n_samples = 200
        
        # Create realistic California housing data
        X = pd.DataFrame({
            'MedInc': np.random.uniform(0.5, 15, n_samples),
            'HouseAge': np.random.uniform(1, 52, n_samples),
            'AveRooms': np.random.uniform(3, 10, n_samples),
            'AveBedrms': np.random.uniform(0.8, 2.5, n_samples),
            'Population': np.random.uniform(100, 5000, n_samples),
            'AveOccup': np.random.uniform(2, 6, n_samples),
            'Latitude': np.random.uniform(32, 42, n_samples),
            'Longitude': np.random.uniform(-124, -114, n_samples)
        })
        
        # Create target with some relationship to features
        y = (X['MedInc'] * 2 + 
             X['AveRooms'] * 1.5 + 
             np.random.randn(n_samples) * 0.5)
        
        # Full pipeline
        pipeline = FeaturePipeline(
            engineer_config={'create_interaction_features': True, 'create_polynomial_features': False},
            selector_config={'selection_method': 'f_test', 'k_features': 15}
        )
        
        # Fit and transform
        X_final = pipeline.fit_transform(X, y)
        
        # Verify pipeline worked
        self.assertTrue(pipeline.is_fitted)
        self.assertEqual(X_final.shape[0], X.shape[0])
        self.assertEqual(X_final.shape[1], 15)  # Selected features
        
        # Get feature info
        info = pipeline.get_feature_info()
        self.assertEqual(info['n_original_features'], 8)
        self.assertEqual(info['n_selected_features'], 15)
        self.assertIsInstance(info['selected_features'], list)
        self.assertEqual(len(info['selected_features']), 15)


if __name__ == '__main__':
    unittest.main()