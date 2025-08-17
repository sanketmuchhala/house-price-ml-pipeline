"""
Machine Learning models module with multiple algorithms and evaluation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Train and evaluate multiple machine learning models.
    """
    
    def __init__(self, models_config: Dict = None, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize ModelTrainer.
        
        Args:
            models_config (Dict): Configuration for models
            cv_folds (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize models with default configuration
        self._initialize_models(models_config)
    
    def _initialize_models(self, models_config: Dict = None) -> None:
        """
        Initialize machine learning models.
        
        Args:
            models_config (Dict): Custom model configurations
        """
        default_config = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {'alpha': 1.0}
            },
            'lasso': {
                'model': Lasso(random_state=self.random_state),
                'params': {'alpha': 1.0}
            },
            'elastic_net': {
                'model': ElasticNet(random_state=self.random_state),
                'params': {'alpha': 1.0, 'l1_ratio': 0.5}
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                }
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=self.random_state),
                'params': {
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1
                }
            },
            'svr': {
                'model': SVR(),
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale'
                }
            },
            'knn': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': 5,
                    'weights': 'uniform'
                }
            }
        }
        
        # Update with custom configuration if provided
        if models_config:
            default_config.update(models_config)
        
        # Set model parameters
        for name, config in default_config.items():
            model = config['model']
            params = config['params']
            
            # Set parameters
            model.set_params(**params)
            self.models[name] = model
        
        self.logger.info(f"Initialized {len(self.models)} models")
    
    def train_single_model(self, name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict:
        """
        Train a single model and evaluate its performance.
        
        Args:
            name (str): Model name
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features (optional)
            y_test (pd.Series): Test target (optional)
            
        Returns:
            Dict: Model training results
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in available models")
        
        try:
            self.logger.info(f"Training {name} model...")
            
            model = self.models[name]
            start_time = datetime.now()
            
            # Train the model
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='neg_mean_squared_error'
            )
            cv_rmse_scores = np.sqrt(-cv_scores)
            
            # Training predictions and metrics
            y_train_pred = model.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            
            # Test predictions and metrics (if test data provided)
            test_metrics = {}
            if X_test is not None and y_test is not None:
                y_test_pred = model.predict(X_test)
                test_metrics = self._calculate_metrics(y_test, y_test_pred)
            
            # Store trained model
            self.trained_models[name] = model
            
            # Prepare results
            results = {
                'model_name': name,
                'training_time': training_time,
                'cv_scores': {
                    'mean_rmse': cv_rmse_scores.mean(),
                    'std_rmse': cv_rmse_scores.std(),
                    'scores': cv_rmse_scores.tolist()
                },
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model_params': model.get_params()
            }
            
            self.model_results[name] = results
            self.logger.info(f"{name} training completed. CV RMSE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training {name}: {str(e)}")
            raise
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict:
        """
        Train all available models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features (optional)
            y_test (pd.Series): Test target (optional)
            
        Returns:
            Dict: Results for all models
        """
        self.logger.info(f"Training {len(self.models)} models...")
        
        results = {}
        for name in self.models.keys():
            try:
                model_result = self.train_single_model(name, X_train, y_train, X_test, y_test)
                results[name] = model_result
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {str(e)}")
                continue
        
        self.logger.info(f"Completed training {len(results)} models successfully")
        return results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate regression metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Calculated metrics
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all trained models.
        
        Returns:
            pd.DataFrame: Model comparison results
        """
        if not self.model_results:
            self.logger.warning("No trained models found for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, results in self.model_results.items():
            row = {
                'Model': name,
                'CV_RMSE_Mean': results['cv_scores']['mean_rmse'],
                'CV_RMSE_Std': results['cv_scores']['std_rmse'],
                'Train_RMSE': results['train_metrics']['rmse'],
                'Train_R2': results['train_metrics']['r2'],
                'Training_Time': results['training_time']
            }
            
            # Add test metrics if available
            if results['test_metrics']:
                row.update({
                    'Test_RMSE': results['test_metrics']['rmse'],
                    'Test_R2': results['test_metrics']['r2'],
                    'Test_MAE': results['test_metrics']['mae']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by CV RMSE (lower is better)
        comparison_df = comparison_df.sort_values('CV_RMSE_Mean')
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'cv_rmse') -> Tuple[str, Any]:
        """
        Get the best performing model.
        
        Args:
            metric (str): Metric to use for selection ('cv_rmse', 'test_rmse', 'test_r2')
            
        Returns:
            Tuple[str, Any]: Best model name and model object
        """
        if not self.model_results:
            raise ValueError("No trained models found")
        
        if metric == 'cv_rmse':
            best_name = min(self.model_results.keys(),
                          key=lambda x: self.model_results[x]['cv_scores']['mean_rmse'])
        elif metric == 'test_rmse':
            # Filter models with test metrics
            models_with_test = {k: v for k, v in self.model_results.items() 
                              if v['test_metrics']}
            if not models_with_test:
                raise ValueError("No models with test metrics found")
            best_name = min(models_with_test.keys(),
                          key=lambda x: models_with_test[x]['test_metrics']['rmse'])
        elif metric == 'test_r2':
            models_with_test = {k: v for k, v in self.model_results.items() 
                              if v['test_metrics']}
            if not models_with_test:
                raise ValueError("No models with test metrics found")
            best_name = max(models_with_test.keys(),
                          key=lambda x: models_with_test[x]['test_metrics']['r2'])
        else:
            raise ValueError("metric must be 'cv_rmse', 'test_rmse', or 'test_r2'")
        
        return best_name, self.trained_models[best_name]
    
    def save_model(self, model_name: str, filepath: str, include_metadata: bool = True) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
            include_metadata (bool): Whether to save metadata alongside model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self.trained_models[model_name], filepath)
            
            # Save metadata if requested
            if include_metadata and model_name in self.model_results:
                metadata_path = filepath.replace('.pkl', '_metadata.json')
                metadata = self.model_results[model_name].copy()
                
                # Convert numpy arrays to lists for JSON serialization
                if 'cv_scores' in metadata and 'scores' in metadata['cv_scores']:
                    metadata['cv_scores']['scores'] = [float(x) for x in metadata['cv_scores']['scores']]
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                self.logger.info(f"Model metadata saved to {metadata_path}")
            
            self.logger.info(f"Model '{model_name}' saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Any: Loaded model
        """
        try:
            model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the model to use
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        return self.trained_models[model_name].predict(X)
    
    def get_feature_importance(self, model_name: str) -> Optional[pd.Series]:
        """
        Get feature importance for models that support it.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Optional[pd.Series]: Feature importance scores
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found in trained models")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return pd.Series(model.feature_importances_, name='importance')
        elif hasattr(model, 'coef_'):
            return pd.Series(np.abs(model.coef_), name='importance')
        else:
            self.logger.warning(f"Model '{model_name}' does not support feature importance")
            return None


def main():
    """
    Main function to demonstrate model training.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import required modules
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from data.data_ingestion import DataIngestion
    from data.data_preprocessing import DataPreprocessor
    from features.feature_engineering import FeaturePipeline
    
    # Load and preprocess data
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    
    preprocessor = DataPreprocessor()
    results = preprocessor.preprocess_pipeline(X, y)
    
    # Feature engineering
    feature_pipeline = FeaturePipeline()
    X_train_featured = feature_pipeline.fit_transform(results['X_train_unscaled'], results['y_train'])
    X_test_featured = feature_pipeline.transform(results['X_test_unscaled'])
    
    # Initialize and train models
    trainer = ModelTrainer()
    
    # Train all models
    training_results = trainer.train_all_models(
        X_train_featured, results['y_train'],
        X_test_featured, results['y_test']
    )
    
    # Compare models
    comparison = trainer.compare_models()
    print("Model Comparison:")
    print(comparison.round(4))
    
    # Get best model
    best_name, best_model = trainer.get_best_model()
    print(f"\nBest model: {best_name}")
    
    # Save best model
    os.makedirs('models/trained', exist_ok=True)
    trainer.save_model(best_name, f'models/trained/{best_name}_best.pkl')


if __name__ == "__main__":
    main()