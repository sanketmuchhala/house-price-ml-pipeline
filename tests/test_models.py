"""
Unit tests for machine learning models module.
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.ml_models import ModelTrainer
from src.models.model_evaluation import ModelEvaluator


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        # Create sample data
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        self.X_test = pd.DataFrame(
            np.random.randn(50, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target with some relationship to features
        self.y_train = pd.Series(
            self.X_train['feature_0'] * 2 + 
            self.X_train['feature_1'] * 1.5 + 
            np.random.randn(n_samples) * 0.5
        )
        
        self.y_test = pd.Series(
            self.X_test['feature_0'] * 2 + 
            self.X_test['feature_1'] * 1.5 + 
            np.random.randn(50) * 0.5
        )
        
        self.trainer = ModelTrainer(cv_folds=3, random_state=42)
    
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        # Test default initialization
        trainer = ModelTrainer()
        self.assertEqual(trainer.cv_folds, 5)
        self.assertEqual(trainer.random_state, 42)
        self.assertIsInstance(trainer.models, dict)
        self.assertGreater(len(trainer.models), 0)
        
        # Test custom initialization
        trainer = ModelTrainer(cv_folds=3, random_state=123)
        self.assertEqual(trainer.cv_folds, 3)
        self.assertEqual(trainer.random_state, 123)
    
    def test_model_initialization(self):
        """Test that models are properly initialized."""
        expected_models = [
            'linear_regression', 'ridge', 'lasso', 'elastic_net',
            'random_forest', 'gradient_boosting', 'xgboost',
            'decision_tree', 'svr', 'knn'
        ]
        
        for model_name in expected_models:
            self.assertIn(model_name, self.trainer.models)
            self.assertIsNotNone(self.trainer.models[model_name])
    
    def test_train_single_model(self):
        """Test training a single model."""
        # Train linear regression
        results = self.trainer.train_single_model(
            'linear_regression', 
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        # Check results structure
        self.assertIn('model_name', results)
        self.assertIn('training_time', results)
        self.assertIn('cv_scores', results)
        self.assertIn('train_metrics', results)
        self.assertIn('test_metrics', results)
        self.assertIn('model_params', results)
        
        # Check values
        self.assertEqual(results['model_name'], 'linear_regression')
        self.assertGreater(results['training_time'], 0)
        self.assertIn('mean_rmse', results['cv_scores'])
        self.assertIn('rmse', results['train_metrics'])
        self.assertIn('rmse', results['test_metrics'])
        
        # Check that model is trained and stored
        self.assertIn('linear_regression', self.trainer.trained_models)
        self.assertIn('linear_regression', self.trainer.model_results)
    
    def test_train_multiple_models(self):
        """Test training multiple models."""
        # Train a subset of models for faster testing
        models_to_test = ['linear_regression', 'ridge', 'random_forest']
        
        # Temporarily modify models dict
        original_models = self.trainer.models.copy()
        self.trainer.models = {k: v for k, v in self.trainer.models.items() if k in models_to_test}
        
        try:
            results = self.trainer.train_all_models(
                self.X_train, self.y_train,
                self.X_test, self.y_test
            )
            
            # Check that all models were trained
            self.assertEqual(len(results), len(models_to_test))
            for model_name in models_to_test:
                self.assertIn(model_name, results)
                self.assertIn(model_name, self.trainer.trained_models)
        
        finally:
            # Restore original models
            self.trainer.models = original_models
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Train a few models first
        self.trainer.train_single_model('linear_regression', self.X_train, self.y_train, self.X_test, self.y_test)
        self.trainer.train_single_model('ridge', self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Compare models
        comparison = self.trainer.compare_models()
        
        # Check comparison structure
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        
        expected_columns = ['Model', 'CV_RMSE_Mean', 'CV_RMSE_Std', 'Train_RMSE', 'Train_R2', 'Training_Time']
        for col in expected_columns:
            self.assertIn(col, comparison.columns)
        
        # Check that results are sorted by CV_RMSE_Mean
        cv_rmse_values = comparison['CV_RMSE_Mean'].values
        self.assertTrue(np.all(cv_rmse_values[:-1] <= cv_rmse_values[1:]))
    
    def test_get_best_model(self):
        """Test getting the best model."""
        # Train some models
        self.trainer.train_single_model('linear_regression', self.X_train, self.y_train, self.X_test, self.y_test)
        self.trainer.train_single_model('ridge', self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Get best model by CV RMSE
        best_name, best_model = self.trainer.get_best_model(metric='cv_rmse')
        
        self.assertIn(best_name, ['linear_regression', 'ridge'])
        self.assertIsNotNone(best_model)
        
        # Get best model by test R2
        best_name_r2, best_model_r2 = self.trainer.get_best_model(metric='test_r2')
        self.assertIn(best_name_r2, ['linear_regression', 'ridge'])
    
    def test_predict(self):
        """Test making predictions."""
        # Train a model
        self.trainer.train_single_model('linear_regression', self.X_train, self.y_train)
        
        # Make predictions
        predictions = self.trainer.predict('linear_regression', self.X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_save_and_load_model(self):
        """Test saving and loading models."""
        # Train a model
        self.trainer.train_single_model('linear_regression', self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            try:
                # Save model
                self.trainer.save_model('linear_regression', tmp_file.name)
                self.assertTrue(os.path.exists(tmp_file.name))
                
                # Load model
                loaded_model = self.trainer.load_model(tmp_file.name)
                self.assertIsNotNone(loaded_model)
                
                # Test predictions are the same
                original_pred = self.trainer.predict('linear_regression', self.X_test)
                loaded_pred = loaded_model.predict(self.X_test)
                
                np.testing.assert_array_almost_equal(original_pred, loaded_pred)
                
            finally:
                # Clean up
                if os.path.exists(tmp_file.name):
                    os.unlink(tmp_file.name)
                # Clean up metadata file if it exists
                metadata_file = tmp_file.name.replace('.pkl', '_metadata.json')
                if os.path.exists(metadata_file):
                    os.unlink(metadata_file)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Test with tree-based model (has feature_importances_)
        self.trainer.train_single_model('random_forest', self.X_train, self.y_train)
        importance = self.trainer.get_feature_importance('random_forest')
        
        if importance is not None:
            self.assertIsInstance(importance, pd.Series)
            self.assertEqual(len(importance), len(self.X_train.columns))
        
        # Test with linear model (has coef_)
        self.trainer.train_single_model('linear_regression', self.X_train, self.y_train)
        importance = self.trainer.get_feature_importance('linear_regression')
        
        if importance is not None:
            self.assertIsInstance(importance, pd.Series)
            self.assertEqual(len(importance), len(self.X_train.columns))
    
    def test_invalid_model_operations(self):
        """Test error handling for invalid operations."""
        # Test training non-existent model
        with self.assertRaises(ValueError):
            self.trainer.train_single_model('nonexistent_model', self.X_train, self.y_train)
        
        # Test predicting with untrained model
        with self.assertRaises(ValueError):
            self.trainer.predict('untrained_model', self.X_test)
        
        # Test saving non-existent model
        with self.assertRaises(ValueError):
            self.trainer.save_model('nonexistent_model', 'dummy_path.pkl')
        
        # Test getting feature importance for non-existent model
        with self.assertRaises(ValueError):
            self.trainer.get_feature_importance('nonexistent_model')


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        n_train, n_test = 150, 50
        n_features = 8
        
        # Create sample data
        self.X_train = pd.DataFrame(
            np.random.randn(n_train, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.X_test = pd.DataFrame(
            np.random.randn(n_test, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create targets
        self.y_train = pd.Series(
            self.X_train['feature_0'] * 2 + 
            self.X_train['feature_1'] * 1.5 + 
            np.random.randn(n_train) * 0.5
        )
        self.y_test = pd.Series(
            self.X_test['feature_0'] * 2 + 
            self.X_test['feature_1'] * 1.5 + 
            np.random.randn(n_test) * 0.5
        )
        
        # Train a simple model for testing
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        # Create evaluator with temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.evaluator = ModelEvaluator(save_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        self.assertEqual(self.evaluator.save_dir, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertIsInstance(self.evaluator.evaluation_results, dict)
    
    def test_evaluate_model(self):
        """Test comprehensive model evaluation."""
        results = self.evaluator.evaluate_model(
            self.model, 'test_model',
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        # Check results structure
        expected_keys = [
            'model_name', 'evaluation_timestamp', 'dataset_info',
            'metrics', 'residuals_analysis', 'feature_importance',
            'model_complexity', 'prediction_intervals', 'predictions'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check specific content
        self.assertEqual(results['model_name'], 'test_model')
        self.assertEqual(results['dataset_info']['train_size'], len(self.X_train))
        self.assertEqual(results['dataset_info']['test_size'], len(self.X_test))
        self.assertEqual(results['dataset_info']['n_features'], self.X_train.shape[1])
        
        # Check metrics
        self.assertIn('train', results['metrics'])
        self.assertIn('test', results['metrics'])
        
        for split in ['train', 'test']:
            metrics = results['metrics'][split]
            for metric in ['rmse', 'mae', 'r2', 'mse', 'mape']:
                self.assertIn(metric, metrics)
                self.assertIsInstance(metrics[metric], (int, float))
        
        # Check that results are stored
        self.assertIn('test_model', self.evaluator.evaluation_results)
    
    def test_calculate_detailed_metrics(self):
        """Test detailed metrics calculation."""
        y_true = pd.Series([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        
        metrics = self.evaluator._calculate_detailed_metrics(y_true, y_pred)
        
        # Check all expected metrics are present
        expected_metrics = [
            'rmse', 'mae', 'r2', 'mse', 'mape',
            'max_error', 'mean_residual', 'std_residual',
            'q95_error', 'q99_error', 'adjusted_r2'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Evaluate multiple models
        model1 = LinearRegression()
        model1.fit(self.X_train, self.y_train)
        
        model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        model2.fit(self.X_train, self.y_train)
        
        self.evaluator.evaluate_model(model1, 'linear_reg', self.X_train, self.y_train, self.X_test, self.y_test)
        self.evaluator.evaluate_model(model2, 'random_forest', self.X_train, self.y_train, self.X_test, self.y_test)
        
        # Compare models
        comparison = self.evaluator.compare_models()
        
        # Check comparison structure
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        
        expected_columns = ['Model', 'Test_RMSE', 'Test_MAE', 'Test_R2', 'Test_MAPE']
        for col in expected_columns:
            self.assertIn(col, comparison.columns)
        
        # Check that results are sorted by RMSE
        rmse_values = comparison['Test_RMSE'].values
        self.assertTrue(np.all(rmse_values[:-1] <= rmse_values[1:]))
    
    def test_save_evaluation_results(self):
        """Test saving evaluation results."""
        # Evaluate a model
        self.evaluator.evaluate_model(
            self.model, 'test_model',
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        # Save results
        save_path = os.path.join(self.temp_dir, 'test_results.json')
        self.evaluator.save_evaluation_results('test_model', save_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Check file content
        import json
        with open(save_path, 'r') as f:
            loaded_results = json.load(f)
        
        self.assertEqual(loaded_results['model_name'], 'test_model')
    
    def test_generate_model_report(self):
        """Test model report generation."""
        # Evaluate a model
        self.evaluator.evaluate_model(
            self.model, 'test_model',
            self.X_train, self.y_train,
            self.X_test, self.y_test
        )
        
        # Generate report
        report = self.evaluator.generate_model_report('test_model')
        
        # Check report content
        self.assertIsInstance(report, str)
        self.assertIn('MODEL EVALUATION REPORT', report)
        self.assertIn('test_model', report)
        self.assertIn('PERFORMANCE METRICS', report)
        self.assertIn('RESIDUALS ANALYSIS', report)
    
    def test_error_handling(self):
        """Test error handling for invalid operations."""
        # Test evaluating with non-existent model results
        with self.assertRaises(ValueError):
            self.evaluator.save_evaluation_results('nonexistent_model')
        
        with self.assertRaises(ValueError):
            self.evaluator.generate_model_report('nonexistent_model')
        
        # Test compare with no models
        empty_evaluator = ModelEvaluator(save_dir=self.temp_dir)
        comparison = empty_evaluator.compare_models()
        self.assertTrue(comparison.empty)


class TestModelsIntegration(unittest.TestCase):
    """Integration tests for models components."""
    
    def test_full_training_and_evaluation_workflow(self):
        """Test complete model training and evaluation workflow."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create sample data
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(X['feature_0'] * 2 + X['feature_1'] * 1.5 + np.random.randn(n_samples) * 0.5)
        
        # Split data
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize trainer and evaluator
            trainer = ModelTrainer(cv_folds=3)
            evaluator = ModelEvaluator(save_dir=temp_dir)
            
            # Train models (subset for faster testing)
            models_to_test = ['linear_regression', 'ridge']
            original_models = trainer.models.copy()
            trainer.models = {k: v for k, v in trainer.models.items() if k in models_to_test}
            
            try:
                # Train all models
                training_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
                
                # Evaluate all trained models
                for model_name in training_results.keys():
                    model = trainer.trained_models[model_name]
                    evaluator.evaluate_model(
                        model, model_name,
                        X_train, y_train, X_test, y_test
                    )
                
                # Compare models
                training_comparison = trainer.compare_models()
                evaluation_comparison = evaluator.compare_models()
                
                # Verify workflow
                self.assertEqual(len(training_results), len(models_to_test))
                self.assertEqual(len(training_comparison), len(models_to_test))
                self.assertEqual(len(evaluation_comparison), len(models_to_test))
                
                # Get best model
                best_name, best_model = trainer.get_best_model()
                self.assertIn(best_name, models_to_test)
                
                # Save best model
                save_path = os.path.join(temp_dir, f'{best_name}_best.pkl')
                trainer.save_model(best_name, save_path)
                self.assertTrue(os.path.exists(save_path))
                
            finally:
                # Restore original models
                trainer.models = original_models


if __name__ == '__main__':
    unittest.main()