"""
Model evaluation and artifact management module.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import json
import os
from datetime import datetime
import pickle
import shutil
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation and performance analysis.
    """
    
    def __init__(self, save_dir: str = "models/artifacts"):
        """
        Initialize ModelEvaluator.
        
        Args:
            save_dir (str): Directory to save evaluation artifacts
        """
        self.save_dir = save_dir
        self.logger = logging.getLogger(__name__)
        self.evaluation_results = {}
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
    
    def evaluate_model(self, model: Any, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series, X_val: pd.DataFrame = None,
                      y_val: pd.Series = None) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            model_name (str): Name of the model
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            X_val (pd.DataFrame): Validation features (optional)
            y_val (pd.Series): Validation target (optional)
            
        Returns:
            Dict: Comprehensive evaluation results
        """
        try:
            self.logger.info(f"Evaluating model: {model_name}")
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            y_val_pred = None
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_metrics = self._calculate_detailed_metrics(y_train, y_train_pred)
            test_metrics = self._calculate_detailed_metrics(y_test, y_test_pred)
            
            val_metrics = {}
            if y_val_pred is not None:
                val_metrics = self._calculate_detailed_metrics(y_val, y_val_pred)
            
            # Calculate residuals analysis
            residuals_analysis = self._analyze_residuals(y_test, y_test_pred)
            
            # Feature importance (if available)
            feature_importance = self._get_feature_importance(model, X_train.columns)
            
            # Model complexity metrics
            complexity_metrics = self._calculate_model_complexity(model)
            
            # Prediction intervals (for regression)
            prediction_intervals = self._calculate_prediction_intervals(model, X_test, y_test_pred)
            
            # Prepare results
            results = {
                'model_name': model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'dataset_info': {
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'val_size': len(X_val) if X_val is not None else 0,
                    'n_features': X_train.shape[1],
                    'feature_names': list(X_train.columns)
                },
                'metrics': {
                    'train': train_metrics,
                    'test': test_metrics,
                    'validation': val_metrics
                },
                'residuals_analysis': residuals_analysis,
                'feature_importance': feature_importance,
                'model_complexity': complexity_metrics,
                'prediction_intervals': prediction_intervals,
                'predictions': {
                    'train': {
                        'y_true': y_train.tolist(),
                        'y_pred': y_train_pred.tolist()
                    },
                    'test': {
                        'y_true': y_test.tolist(),
                        'y_pred': y_test_pred.tolist()
                    }
                }
            }
            
            # Add validation predictions if available
            if y_val_pred is not None:
                results['predictions']['validation'] = {
                    'y_true': y_val.tolist(),
                    'y_pred': y_val_pred.tolist()
                }
            
            self.evaluation_results[model_name] = results
            
            self.logger.info(f"Model evaluation completed for {model_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
    
    def _calculate_detailed_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Calculate detailed regression metrics.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Detailed metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_residual': np.mean(y_true - y_pred),
            'std_residual': np.std(y_true - y_pred),
            'q95_error': np.percentile(np.abs(y_true - y_pred), 95),
            'q99_error': np.percentile(np.abs(y_true - y_pred), 99)
        }
        
        # Adjusted R-squared (requires number of features)
        n = len(y_true)
        p = 1  # Will be updated if feature count is available
        metrics['adjusted_r2'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
        
        return metrics
    
    def _analyze_residuals(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """
        Analyze model residuals.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict: Residuals analysis
        """
        residuals = y_true - y_pred
        
        analysis = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(residuals.skew()) if hasattr(residuals, 'skew') else 0,
            'kurtosis': float(residuals.kurtosis()) if hasattr(residuals, 'kurtosis') else 0,
            'normality_test': self._test_normality(residuals),
            'autocorrelation': self._test_autocorrelation(residuals),
            'heteroscedasticity': self._test_heteroscedasticity(y_pred, residuals)
        }
        
        return analysis
    
    def _test_normality(self, residuals: np.ndarray) -> Dict:
        """Test normality of residuals."""
        try:
            from scipy import stats
            statistic, p_value = stats.jarque_bera(residuals)
            return {
                'test': 'Jarque-Bera',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        except:
            return {'test': 'Jarque-Bera', 'error': 'Test failed'}
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict:
        """Test autocorrelation in residuals."""
        try:
            from scipy import stats
            # Durbin-Watson test approximation
            dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
            return {
                'test': 'Durbin-Watson',
                'statistic': float(dw_stat),
                'interpretation': 'no_autocorr' if 1.5 < dw_stat < 2.5 else 'autocorr_present'
            }
        except:
            return {'test': 'Durbin-Watson', 'error': 'Test failed'}
    
    def _test_heteroscedasticity(self, y_pred: np.ndarray, residuals: np.ndarray) -> Dict:
        """Test heteroscedasticity."""
        try:
            from scipy import stats
            # Breusch-Pagan test approximation
            corr_coef, p_value = stats.pearsonr(y_pred, np.abs(residuals))
            return {
                'test': 'Breusch-Pagan (approximation)',
                'correlation': float(corr_coef),
                'p_value': float(p_value),
                'is_homoscedastic': p_value > 0.05
            }
        except:
            return {'test': 'Breusch-Pagan', 'error': 'Test failed'}
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[Dict]:
        """Get feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                importance_dict = dict(zip(feature_names, importance_scores))
                # Sort by importance
                sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                return {
                    'type': 'gini_importance',
                    'scores': sorted_importance,
                    'top_features': list(sorted_importance.keys())[:10]
                }
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coef_abs = np.abs(model.coef_)
                importance_dict = dict(zip(feature_names, coef_abs))
                sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
                return {
                    'type': 'coefficient_magnitude',
                    'scores': sorted_importance,
                    'top_features': list(sorted_importance.keys())[:10]
                }
            else:
                return None
        except:
            return None
    
    def _calculate_model_complexity(self, model: Any) -> Dict:
        """Calculate model complexity metrics."""
        complexity = {
            'model_type': type(model).__name__,
            'n_parameters': self._count_parameters(model)
        }
        
        # Model-specific complexity metrics
        if hasattr(model, 'n_estimators'):
            complexity['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            complexity['max_depth'] = model.max_depth
        if hasattr(model, 'n_features_in_'):
            complexity['n_features'] = model.n_features_in_
        
        return complexity
    
    def _count_parameters(self, model: Any) -> Optional[int]:
        """Count model parameters (approximation)."""
        try:
            if hasattr(model, 'coef_'):
                return len(model.coef_) + (1 if hasattr(model, 'intercept_') else 0)
            elif hasattr(model, 'n_estimators') and hasattr(model, 'max_depth'):
                # Rough estimate for tree-based models
                return getattr(model, 'n_estimators', 1) * (2 ** getattr(model, 'max_depth', 5))
            else:
                return None
        except:
            return None
    
    def _calculate_prediction_intervals(self, model: Any, X_test: pd.DataFrame, 
                                      y_pred: np.ndarray, confidence: float = 0.95) -> Dict:
        """Calculate prediction intervals (simplified approach)."""
        try:
            # For ensemble models, use prediction variance
            if hasattr(model, 'estimators_'):
                # Get predictions from individual estimators
                individual_preds = np.array([est.predict(X_test) for est in model.estimators_])
                pred_std = np.std(individual_preds, axis=0)
                
                # Calculate confidence intervals
                from scipy import stats
                alpha = 1 - confidence
                z_score = stats.norm.ppf(1 - alpha/2)
                
                lower_bound = y_pred - z_score * pred_std
                upper_bound = y_pred + z_score * pred_std
                
                return {
                    'confidence_level': confidence,
                    'method': 'ensemble_variance',
                    'lower_bound': lower_bound.tolist(),
                    'upper_bound': upper_bound.tolist(),
                    'prediction_std': pred_std.tolist(),
                    'interval_width': (upper_bound - lower_bound).tolist()
                }
            else:
                return {'error': 'Prediction intervals not available for this model type'}
        except:
            return {'error': 'Failed to calculate prediction intervals'}
    
    def create_evaluation_plots(self, model_name: str, save_plots: bool = True) -> None:
        """
        Create comprehensive evaluation plots.
        
        Args:
            model_name (str): Name of the model to plot
            save_plots (bool): Whether to save plots to disk
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for model {model_name}")
        
        results = self.evaluation_results[model_name]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)
        
        # 1. Predicted vs Actual (Test set)
        test_pred = results['predictions']['test']
        axes[0, 0].scatter(test_pred['y_true'], test_pred['y_pred'], alpha=0.6)
        axes[0, 0].plot([min(test_pred['y_true']), max(test_pred['y_true'])], 
                       [min(test_pred['y_true']), max(test_pred['y_true'])], 'r--')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Predicted vs Actual (Test Set)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = np.array(test_pred['y_true']) - np.array(test_pred['y_pred'])
        axes[0, 1].scatter(test_pred['y_pred'], residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        axes[0, 2].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residuals Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature importance (if available)
        if results['feature_importance']:
            importance_data = results['feature_importance']['scores']
            top_10 = dict(list(importance_data.items())[:10])
            
            axes[1, 0].barh(list(top_10.keys())[::-1], list(top_10.values())[::-1])
            axes[1, 0].set_xlabel('Importance Score')
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')
        
        # 5. Prediction errors distribution
        errors = np.abs(residuals)
        axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Errors Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model performance metrics
        test_metrics = results['metrics']['test']
        metrics_text = f"""Test Set Metrics:
        
RMSE: {test_metrics['rmse']:.4f}
MAE: {test_metrics['mae']:.4f}
R²: {test_metrics['r2']:.4f}
MAPE: {test_metrics['mape']:.2f}%

Residuals Analysis:
Mean: {test_metrics['mean_residual']:.4f}
Std: {test_metrics['std_residual']:.4f}"""
        
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.save_dir, f'{model_name}_evaluation_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Evaluation plots saved to {plot_path}")
        
        plt.show()
    
    def save_evaluation_results(self, model_name: str, filepath: str = None) -> None:
        """
        Save evaluation results to file.
        
        Args:
            model_name (str): Name of the model
            filepath (str): Custom filepath (optional)
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for model {model_name}")
        
        if filepath is None:
            filepath = os.path.join(self.save_dir, f'{model_name}_evaluation_results.json')
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.evaluation_results[model_name], f, indent=2, default=str)
            
            self.logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {str(e)}")
            raise
    
    def compare_models(self, model_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_names (List[str]): List of model names to compare (all if None)
            
        Returns:
            pd.DataFrame: Model comparison results
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        comparison_data = []
        
        for name in model_names:
            if name in self.evaluation_results:
                results = self.evaluation_results[name]
                test_metrics = results['metrics']['test']
                
                row = {
                    'Model': name,
                    'Test_RMSE': test_metrics['rmse'],
                    'Test_MAE': test_metrics['mae'],
                    'Test_R2': test_metrics['r2'],
                    'Test_MAPE': test_metrics['mape'],
                    'Max_Error': test_metrics['max_error'],
                    'Q95_Error': test_metrics['q95_error'],
                    'N_Features': results['dataset_info']['n_features'],
                    'Train_Size': results['dataset_info']['train_size'],
                    'Test_Size': results['dataset_info']['test_size']
                }
                
                # Add complexity metrics if available
                if results['model_complexity']['n_parameters']:
                    row['N_Parameters'] = results['model_complexity']['n_parameters']
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('Test_RMSE')
        
        return comparison_df
    
    def generate_model_report(self, model_name: str) -> str:
        """
        Generate a comprehensive text report for a model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Formatted model report
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"No evaluation results found for model {model_name}")
        
        results = self.evaluation_results[model_name]
        
        report = f"""
MODEL EVALUATION REPORT
========================

Model: {model_name}
Evaluation Date: {results['evaluation_timestamp']}
Model Type: {results['model_complexity']['model_type']}

DATASET INFORMATION
-------------------
Training Set Size: {results['dataset_info']['train_size']:,}
Test Set Size: {results['dataset_info']['test_size']:,}
Number of Features: {results['dataset_info']['n_features']}

PERFORMANCE METRICS
-------------------
Test Set Performance:
  RMSE: {results['metrics']['test']['rmse']:.6f}
  MAE: {results['metrics']['test']['mae']:.6f}
  R² Score: {results['metrics']['test']['r2']:.6f}
  MAPE: {results['metrics']['test']['mape']:.2f}%
  Max Error: {results['metrics']['test']['max_error']:.6f}

Training Set Performance:
  RMSE: {results['metrics']['train']['rmse']:.6f}
  MAE: {results['metrics']['train']['mae']:.6f}
  R² Score: {results['metrics']['train']['r2']:.6f}

RESIDUALS ANALYSIS
------------------
Mean Residual: {results['residuals_analysis']['mean']:.6f}
Residual Std Dev: {results['residuals_analysis']['std']:.6f}
Skewness: {results['residuals_analysis']['skewness']:.6f}
Kurtosis: {results['residuals_analysis']['kurtosis']:.6f}

Normality Test: {results['residuals_analysis']['normality_test'].get('test', 'N/A')}
  P-value: {results['residuals_analysis']['normality_test'].get('p_value', 'N/A')}
  Normal Distribution: {results['residuals_analysis']['normality_test'].get('is_normal', 'N/A')}

MODEL COMPLEXITY
----------------
Parameters: {results['model_complexity'].get('n_parameters', 'N/A')}
"""

        # Add feature importance if available
        if results['feature_importance']:
            report += f"""
FEATURE IMPORTANCE
------------------
Top 10 Most Important Features:
"""
            for i, (feature, importance) in enumerate(list(results['feature_importance']['scores'].items())[:10], 1):
                report += f"  {i:2d}. {feature}: {importance:.6f}\n"
        
        return report


def main():
    """
    Main function to demonstrate model evaluation.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # This would typically be called after training models
    print("Model evaluation module ready for use.")
    print("Import and use ModelEvaluator class to evaluate trained models.")


if __name__ == "__main__":
    main()