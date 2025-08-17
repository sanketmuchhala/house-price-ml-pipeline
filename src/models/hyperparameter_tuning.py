"""
Hyperparameter tuning module using multiple optimization strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning with multiple search strategies.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize HyperparameterTuner.
        
        Args:
            cv_folds (int): Number of cross-validation folds
            random_state (int): Random state for reproducibility
            n_jobs (int): Number of jobs for parallel processing
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        self.tuning_results = {}
        
        # Define parameter grids for different models
        self.param_grids = self._get_parameter_grids()
    
    def _get_parameter_grids(self) -> Dict:
        """
        Get parameter grids for different models.
        
        Returns:
            Dict: Parameter grids for each model type
        """
        return {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'max_iter': [1000, 5000, 10000]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'max_iter': [1000, 5000]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 6, 9, 12],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            },
            'svr': {
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'p': [1, 2]
            }
        }
    
    def grid_search_tuning(self, model: Any, model_name: str, X: pd.DataFrame, y: pd.Series,
                          param_grid: Dict = None, scoring: str = 'neg_mean_squared_error') -> Dict:
        """
        Perform grid search hyperparameter tuning.
        
        Args:
            model: Machine learning model
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            param_grid (Dict): Parameter grid (uses default if None)
            scoring (str): Scoring metric
            
        Returns:
            Dict: Tuning results
        """
        try:
            self.logger.info(f"Starting grid search tuning for {model_name}...")
            
            if param_grid is None:
                param_grid = self.param_grids.get(model_name.lower(), {})
            
            if not param_grid:
                self.logger.warning(f"No parameter grid found for {model_name}")
                return {}
            
            start_time = datetime.now()
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=0
            )
            
            grid_search.fit(X, y)
            
            tuning_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare results
            results = {
                'method': 'grid_search',
                'model_name': model_name,
                'best_params': grid_search.best_params_,
                'best_score': -grid_search.best_score_,  # Convert back to positive RMSE
                'cv_results': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                },
                'tuning_time': tuning_time,
                'total_fits': len(grid_search.cv_results_['params']) * self.cv_folds
            }
            
            self.tuning_results[f"{model_name}_grid_search"] = results
            
            self.logger.info(f"Grid search completed for {model_name}. "
                           f"Best RMSE: {results['best_score']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in grid search tuning: {str(e)}")
            raise
    
    def random_search_tuning(self, model: Any, model_name: str, X: pd.DataFrame, y: pd.Series,
                           param_distributions: Dict = None, n_iter: int = 100,
                           scoring: str = 'neg_mean_squared_error') -> Dict:
        """
        Perform randomized search hyperparameter tuning.
        
        Args:
            model: Machine learning model
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            param_distributions (Dict): Parameter distributions
            n_iter (int): Number of parameter settings sampled
            scoring (str): Scoring metric
            
        Returns:
            Dict: Tuning results
        """
        try:
            self.logger.info(f"Starting random search tuning for {model_name}...")
            
            if param_distributions is None:
                param_distributions = self.param_grids.get(model_name.lower(), {})
            
            if not param_distributions:
                self.logger.warning(f"No parameter distributions found for {model_name}")
                return {}
            
            start_time = datetime.now()
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=self.cv_folds,
                scoring=scoring,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=0
            )
            
            random_search.fit(X, y)
            
            tuning_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare results
            results = {
                'method': 'random_search',
                'model_name': model_name,
                'best_params': random_search.best_params_,
                'best_score': -random_search.best_score_,
                'cv_results': {
                    'mean_test_scores': random_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': random_search.cv_results_['std_test_score'].tolist(),
                    'params': random_search.cv_results_['params']
                },
                'tuning_time': tuning_time,
                'total_fits': n_iter * self.cv_folds,
                'n_iter': n_iter
            }
            
            self.tuning_results[f"{model_name}_random_search"] = results
            
            self.logger.info(f"Random search completed for {model_name}. "
                           f"Best RMSE: {results['best_score']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in random search tuning: {str(e)}")
            raise
    
    def optuna_tuning(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                     n_trials: int = 100, timeout: int = None) -> Dict:
        """
        Perform Optuna-based hyperparameter tuning.
        
        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            n_trials (int): Number of trials
            timeout (int): Timeout in seconds
            
        Returns:
            Dict: Tuning results
        """
        try:
            self.logger.info(f"Starting Optuna tuning for {model_name}...")
            
            def objective(trial):
                # Define hyperparameter search space based on model
                if model_name.lower() == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                        'random_state': self.random_state
                    }
                    model = RandomForestRegressor(**params)
                    
                elif model_name.lower() == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1, 3),
                        'random_state': self.random_state
                    }
                    model = xgb.XGBRegressor(**params)
                    
                elif model_name.lower() == 'gradient_boosting':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'random_state': self.random_state
                    }
                    model = GradientBoostingRegressor(**params)
                    
                elif model_name.lower() == 'ridge':
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.001, 1000, log=True),
                        'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
                        'random_state': self.random_state
                    }
                    model = Ridge(**params)
                    
                elif model_name.lower() == 'lasso':
                    params = {
                        'alpha': trial.suggest_float('alpha', 0.001, 100, log=True),
                        'max_iter': trial.suggest_int('max_iter', 1000, 10000),
                        'random_state': self.random_state
                    }
                    model = Lasso(**params)
                    
                else:
                    raise ValueError(f"Optuna tuning not implemented for {model_name}")
                
                # Cross-validation
                scores = cross_val_score(model, X, y, cv=self.cv_folds, 
                                       scoring='neg_mean_squared_error', n_jobs=self.n_jobs)
                return np.mean(scores)
            
            start_time = datetime.now()
            
            # Create study and optimize
            study = optuna.create_study(
                direction='maximize',  # Maximize negative MSE
                sampler=TPESampler(seed=self.random_state)
            )
            
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
            
            tuning_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare results
            results = {
                'method': 'optuna',
                'model_name': model_name,
                'best_params': study.best_params,
                'best_score': np.sqrt(-study.best_value),  # Convert to RMSE
                'n_trials': len(study.trials),
                'tuning_time': tuning_time,
                'study_summary': {
                    'best_trial_number': study.best_trial.number,
                    'best_value': study.best_value,
                    'trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                }
            }
            
            self.tuning_results[f"{model_name}_optuna"] = results
            
            self.logger.info(f"Optuna tuning completed for {model_name}. "
                           f"Best RMSE: {results['best_score']:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Optuna tuning: {str(e)}")
            raise
    
    def compare_tuning_results(self) -> pd.DataFrame:
        """
        Compare results from different tuning methods.
        
        Returns:
            pd.DataFrame: Comparison of tuning results
        """
        if not self.tuning_results:
            self.logger.warning("No tuning results found for comparison")
            return pd.DataFrame()
        
        comparison_data = []
        
        for key, results in self.tuning_results.items():
            row = {
                'Experiment': key,
                'Model': results['model_name'],
                'Method': results['method'],
                'Best_RMSE': results['best_score'],
                'Tuning_Time': results['tuning_time'],
                'Total_Fits': results.get('total_fits', results.get('n_trials', 'N/A'))
            }
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Best_RMSE')
        
        return comparison_df
    
    def get_best_params(self, model_name: str, method: str = None) -> Dict:
        """
        Get best parameters for a specific model.
        
        Args:
            model_name (str): Name of the model
            method (str): Tuning method (None for best across all methods)
            
        Returns:
            Dict: Best parameters
        """
        if method:
            key = f"{model_name}_{method}"
            if key in self.tuning_results:
                return self.tuning_results[key]['best_params']
            else:
                raise ValueError(f"No results found for {key}")
        else:
            # Find best across all methods for this model
            model_results = {k: v for k, v in self.tuning_results.items() 
                           if v['model_name'] == model_name}
            
            if not model_results:
                raise ValueError(f"No results found for model {model_name}")
            
            best_key = min(model_results.keys(), 
                          key=lambda x: model_results[x]['best_score'])
            
            return model_results[best_key]['best_params']
    
    def save_tuning_results(self, filepath: str) -> None:
        """
        Save tuning results to file.
        
        Args:
            filepath (str): Path to save results
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.tuning_results, f, indent=2, default=str)
            
            self.logger.info(f"Tuning results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving tuning results: {str(e)}")
            raise
    
    def load_tuning_results(self, filepath: str) -> None:
        """
        Load tuning results from file.
        
        Args:
            filepath (str): Path to load results from
        """
        try:
            with open(filepath, 'r') as f:
                self.tuning_results = json.load(f)
            
            self.logger.info(f"Tuning results loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading tuning results: {str(e)}")
            raise


def main():
    """
    Main function to demonstrate hyperparameter tuning.
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
    results = preprocessor.preprocess_pipeline(X, y, test_size=0.3)
    
    # Feature engineering
    feature_pipeline = FeaturePipeline()
    X_train_featured = feature_pipeline.fit_transform(results['X_train_unscaled'], results['y_train'])
    
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Example: Tune Random Forest with different methods
    rf_model = RandomForestRegressor(random_state=42)
    
    # Grid search (fast example with limited grid)
    limited_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    
    grid_results = tuner.grid_search_tuning(
        rf_model, 'random_forest', X_train_featured, results['y_train'], 
        param_grid=limited_grid
    )
    
    # Optuna tuning
    optuna_results = tuner.optuna_tuning(
        'random_forest', X_train_featured, results['y_train'], n_trials=20
    )
    
    # Compare results
    comparison = tuner.compare_tuning_results()
    print("Tuning Results Comparison:")
    print(comparison)
    
    # Save results
    os.makedirs('models/artifacts', exist_ok=True)
    tuner.save_tuning_results('models/artifacts/tuning_results.json')


if __name__ == "__main__":
    main()