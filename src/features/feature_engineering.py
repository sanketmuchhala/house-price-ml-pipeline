"""
Feature engineering module for creating new features and feature selection.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering class for California housing dataset.
    """
    
    def __init__(self, create_interaction_features: bool = True,
                 create_polynomial_features: bool = False,
                 polynomial_degree: int = 2):
        """
        Initialize FeatureEngineer.
        
        Args:
            create_interaction_features (bool): Whether to create interaction features
            create_polynomial_features (bool): Whether to create polynomial features
            polynomial_degree (int): Degree for polynomial features
        """
        self.create_interaction_features = create_interaction_features
        self.create_polynomial_features = create_polynomial_features
        self.polynomial_degree = polynomial_degree
        self.feature_names_ = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Fit the feature engineer to the data.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable (unused)
        """
        self.feature_names_ = list(X.columns)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input features by creating new engineered features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Transformed features with new engineered features
        """
        try:
            X_transformed = X.copy()
            
            # Create domain-specific features for California housing dataset
            X_transformed = self._create_housing_features(X_transformed)
            
            # Create interaction features
            if self.create_interaction_features:
                X_transformed = self._create_interaction_features(X_transformed)
            
            # Create polynomial features
            if self.create_polynomial_features:
                X_transformed = self._create_polynomial_features(X_transformed)
            
            self.logger.info(f"Feature engineering completed. Original features: {X.shape[1]}, "
                           f"New features: {X_transformed.shape[1]}")
            
            return X_transformed
            
        except Exception as e:
            self.logger.error(f"Error in feature transformation: {str(e)}")
            raise
    
    def _create_housing_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for housing data.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with new housing-specific features
        """
        X_new = X.copy()
        
        # Price per square foot approximation (assuming rooms correlate with area)
        if 'AveRooms' in X_new.columns:
            X_new['RoomDensity'] = X_new['AveRooms'] / X_new['AveOccup'] if 'AveOccup' in X_new.columns else X_new['AveRooms']
        
        # Bedroom ratio
        if 'AveBedrms' in X_new.columns and 'AveRooms' in X_new.columns:
            X_new['BedroomRatio'] = X_new['AveBedrms'] / X_new['AveRooms']
            X_new['BedroomRatio'] = X_new['BedroomRatio'].fillna(0)
        
        # Population density
        if 'Population' in X_new.columns and 'AveOccup' in X_new.columns:
            X_new['PopulationDensity'] = X_new['Population'] / (X_new['AveOccup'] + 1e-8)
        
        # Income per person
        if 'MedInc' in X_new.columns and 'AveOccup' in X_new.columns:
            X_new['IncomePerPerson'] = X_new['MedInc'] / (X_new['AveOccup'] + 1e-8)
        
        # Age categories
        if 'HouseAge' in X_new.columns:
            X_new['IsNewHouse'] = (X_new['HouseAge'] < 10).astype(int)
            X_new['IsOldHouse'] = (X_new['HouseAge'] > 40).astype(int)
        
        # Geographic features
        if 'Latitude' in X_new.columns and 'Longitude' in X_new.columns:
            # Distance from major city centers (approximated)
            # San Francisco: 37.7749° N, 122.4194° W
            # Los Angeles: 34.0522° N, 118.2437° W
            
            X_new['DistanceToSF'] = np.sqrt(
                (X_new['Latitude'] - 37.7749)**2 + (X_new['Longitude'] + 122.4194)**2
            )
            
            X_new['DistanceToLA'] = np.sqrt(
                (X_new['Latitude'] - 34.0522)**2 + (X_new['Longitude'] + 118.2437)**2
            )
            
            # Coastal proximity (longitude-based approximation)
            X_new['CoastalProximity'] = np.abs(X_new['Longitude'] + 120)  # Closer to coast = lower value
            
            # Northern vs Southern California
            X_new['IsNorthernCA'] = (X_new['Latitude'] > 36.0).astype(int)
        
        # Income categories
        if 'MedInc' in X_new.columns:
            X_new['HighIncome'] = (X_new['MedInc'] > X_new['MedInc'].quantile(0.75)).astype(int)
            X_new['LowIncome'] = (X_new['MedInc'] < X_new['MedInc'].quantile(0.25)).astype(int)
        
        self.logger.info(f"Created {X_new.shape[1] - X.shape[1]} housing-specific features")
        return X_new
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with interaction terms
        """
        X_new = X.copy()
        
        # Define important feature pairs for interactions
        interaction_pairs = [
            ('MedInc', 'AveRooms'),
            ('MedInc', 'HouseAge'),
            ('AveRooms', 'AveBedrms'),
            ('Population', 'AveOccup'),
            ('Latitude', 'Longitude')
        ]
        
        created_interactions = 0
        for feature1, feature2 in interaction_pairs:
            if feature1 in X_new.columns and feature2 in X_new.columns:
                # Multiplicative interaction
                interaction_name = f'{feature1}_x_{feature2}'
                X_new[interaction_name] = X_new[feature1] * X_new[feature2]
                created_interactions += 1
        
        self.logger.info(f"Created {created_interactions} interaction features")
        return X_new
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create polynomial features for important variables.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with polynomial terms
        """
        X_new = X.copy()
        
        # Create polynomial features for specific important features
        polynomial_features = ['MedInc', 'AveRooms', 'HouseAge']
        
        for feature in polynomial_features:
            if feature in X_new.columns:
                for degree in range(2, self.polynomial_degree + 1):
                    poly_name = f'{feature}_poly_{degree}'
                    X_new[poly_name] = X_new[feature] ** degree
        
        self.logger.info(f"Created polynomial features up to degree {self.polynomial_degree}")
        return X_new


class FeatureSelector:
    """
    Feature selection class with multiple selection methods.
    """
    
    def __init__(self, selection_method: str = "mutual_info", k_features: int = None):
        """
        Initialize FeatureSelector.
        
        Args:
            selection_method (str): Selection method ('mutual_info', 'f_test', 'rfe', 'lasso')
            k_features (int): Number of features to select (None for automatic)
        """
        self.selection_method = selection_method
        self.k_features = k_features
        self.selector = None
        self.selected_features_ = None
        self.feature_scores_ = None
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            
        Returns:
            FeatureSelector: Fitted selector
        """
        try:
            n_features = X.shape[1]
            k = self.k_features if self.k_features else min(n_features, max(5, n_features // 2))
            
            if self.selection_method == "f_test":
                self.selector = SelectKBest(score_func=f_regression, k=k)
                
            elif self.selection_method == "rfe":
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                self.selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
                
            elif self.selection_method == "lasso":
                lasso = LassoCV(cv=5, random_state=42)
                self.selector = SelectFromModel(lasso, max_features=k)
                
            else:
                raise ValueError(f"Unknown selection method: {self.selection_method}")
            
            # Fit the selector
            self.selector.fit(X, y)
            
            # Get selected features
            if hasattr(self.selector, 'get_support'):
                selected_mask = self.selector.get_support()
                self.selected_features_ = X.columns[selected_mask].tolist()
            
            # Get feature scores if available
            if hasattr(self.selector, 'scores_'):
                self.feature_scores_ = dict(zip(X.columns, self.selector.scores_))
            elif hasattr(self.selector, 'ranking_'):
                self.feature_scores_ = dict(zip(X.columns, self.selector.ranking_))
            
            self.logger.info(f"Feature selection completed using {self.selection_method}. "
                           f"Selected {len(self.selected_features_)} features from {n_features}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted selector.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Selected features
        """
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        if self.selected_features_:
            return X[self.selected_features_]
        else:
            X_selected = self.selector.transform(X)
            return pd.DataFrame(X_selected, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            
        Returns:
            pd.DataFrame: Selected features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.feature_scores_ is None:
            return {}
        
        # Sort by importance (lower ranking/higher score is better)
        if self.selection_method == "rfe":
            # For RFE, lower ranking is better
            return dict(sorted(self.feature_scores_.items(), key=lambda x: x[1]))
        else:
            # For others, higher score is better
            return dict(sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True))


class FeaturePipeline:
    """
    Complete feature engineering and selection pipeline.
    """
    
    def __init__(self, engineer_config: Dict = None, selector_config: Dict = None):
        """
        Initialize FeaturePipeline.
        
        Args:
            engineer_config (Dict): Configuration for feature engineering
            selector_config (Dict): Configuration for feature selection
        """
        # Default configurations
        engineer_config = engineer_config or {}
        selector_config = selector_config or {'selection_method': 'f_test', 'k_features': None}
        
        self.feature_engineer = FeatureEngineer(**engineer_config)
        self.feature_selector = FeatureSelector(**selector_config)
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeaturePipeline':
        """
        Fit the complete pipeline.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            
        Returns:
            FeaturePipeline: Fitted pipeline
        """
        try:
            self.logger.info("Starting feature pipeline fitting...")
            
            # 1. Feature engineering
            self.feature_engineer.fit(X, y)
            X_engineered = self.feature_engineer.transform(X)
            
            # 2. Feature selection
            self.feature_selector.fit(X_engineered, y)
            
            self.is_fitted = True
            self.logger.info("Feature pipeline fitted successfully")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting feature pipeline: {str(e)}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted pipeline.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Transformed and selected features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        try:
            # 1. Feature engineering
            X_engineered = self.feature_engineer.transform(X)
            
            # 2. Feature selection
            X_selected = self.feature_selector.transform(X_engineered)
            
            return X_selected
            
        except Exception as e:
            self.logger.error(f"Error transforming features: {str(e)}")
            raise
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            
        Returns:
            pd.DataFrame: Transformed and selected features
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the feature pipeline.
        
        Returns:
            Dict: Feature pipeline information
        """
        if not self.is_fitted:
            return {"error": "Pipeline not fitted"}
        
        return {
            "original_features": self.feature_engineer.feature_names_,
            "selected_features": self.feature_selector.selected_features_,
            "feature_scores": self.feature_selector.get_feature_importance(),
            "selection_method": self.feature_selector.selection_method,
            "n_original_features": len(self.feature_engineer.feature_names_) if self.feature_engineer.feature_names_ else 0,
            "n_selected_features": len(self.feature_selector.selected_features_) if self.feature_selector.selected_features_ else 0
        }
    
    def save_pipeline(self, filepath: str) -> None:
        """
        Save the fitted pipeline.
        
        Args:
            filepath (str): Path to save the pipeline
        """
        try:
            pipeline_data = {
                'feature_engineer': self.feature_engineer,
                'feature_selector': self.feature_selector,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(pipeline_data, filepath)
            self.logger.info(f"Feature pipeline saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline: {str(e)}")
            raise
    
    def load_pipeline(self, filepath: str) -> None:
        """
        Load a fitted pipeline.
        
        Args:
            filepath (str): Path to the saved pipeline
        """
        try:
            pipeline_data = joblib.load(filepath)
            self.feature_engineer = pipeline_data['feature_engineer']
            self.feature_selector = pipeline_data['feature_selector']
            self.is_fitted = pipeline_data['is_fitted']
            
            self.logger.info(f"Feature pipeline loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline: {str(e)}")
            raise


def main():
    """
    Main function to demonstrate feature engineering pipeline.
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
    
    # Load and preprocess data
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    
    preprocessor = DataPreprocessor()
    results = preprocessor.preprocess_pipeline(X, y)
    
    X_train = results['X_train_unscaled']
    y_train = results['y_train']
    
    # Initialize and run feature pipeline
    feature_pipeline = FeaturePipeline(
        engineer_config={'create_interaction_features': True, 'create_polynomial_features': False},
        selector_config={'selection_method': 'f_test', 'k_features': 15}
    )
    
    # Fit and transform
    X_train_featured = feature_pipeline.fit_transform(X_train, y_train)
    
    # Display results
    info = feature_pipeline.get_feature_info()
    print("Feature Engineering Results:")
    print(f"Original features: {info['n_original_features']}")
    print(f"Selected features: {info['n_selected_features']}")
    print(f"Selected feature names: {info['selected_features']}")


if __name__ == "__main__":
    main()