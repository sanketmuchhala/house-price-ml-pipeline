"""
Data preprocessing and cleaning module.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    """
    Handles data cleaning and preprocessing operations.
    """
    
    def __init__(self, scaling_method: str = "standard"):
        """
        Initialize DataPreprocessor.
        
        Args:
            scaling_method (str): Scaling method ('standard', 'robust', 'minmax')
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.imputer = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize scaler based on method
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard', 'robust', or 'minmax'")
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Analyze data quality and provide summary statistics.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Data quality analysis results
        """
        analysis = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicated_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'outliers_iqr': {}
        }
        
        # Detect outliers using IQR method for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            analysis['outliers_iqr'][col] = outliers
        
        self.logger.info(f"Data quality analysis completed for dataset with shape {analysis['shape']}")
        return analysis
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        try:
            df_cleaned = df.copy()
            
            # Check for missing values
            missing_count = df_cleaned.isnull().sum().sum()
            if missing_count == 0:
                self.logger.info("No missing values found in the dataset")
                return df_cleaned
            
            self.logger.info(f"Found {missing_count} missing values. Applying {strategy} imputation...")
            
            # Separate numeric and categorical columns
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            categorical_cols = df_cleaned.select_dtypes(exclude=[np.number]).columns
            
            # Handle numeric columns
            if len(numeric_cols) > 0 and df_cleaned[numeric_cols].isnull().any().any():
                numeric_strategy = strategy if strategy in ['mean', 'median'] else 'median'
                self.imputer = SimpleImputer(strategy=numeric_strategy)
                df_cleaned[numeric_cols] = self.imputer.fit_transform(df_cleaned[numeric_cols])
                self.logger.info(f"Numeric columns imputed using {numeric_strategy} strategy")
            
            # Handle categorical columns
            if len(categorical_cols) > 0 and df_cleaned[categorical_cols].isnull().any().any():
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_cleaned[categorical_cols] = cat_imputer.fit_transform(df_cleaned[categorical_cols])
                self.logger.info("Categorical columns imputed using most_frequent strategy")
            
            return df_cleaned
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = "iqr", 
                                 factor: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and handle outliers in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Outlier detection method ('iqr', 'zscore')
            factor (float): Factor for outlier detection
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned dataframe and outlier info
        """
        try:
            df_cleaned = df.copy()
            outlier_info = {}
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if method == "iqr":
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                    
                elif method == "zscore":
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(df_cleaned[col]))
                    outliers_mask = z_scores > factor
                    
                else:
                    raise ValueError("method must be 'iqr' or 'zscore'")
                
                outlier_count = outliers_mask.sum()
                outlier_info[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(df_cleaned)) * 100
                }
                
                # Cap outliers instead of removing them to preserve data
                if outlier_count > 0:
                    if method == "iqr":
                        df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                        df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                    elif method == "zscore":
                        # Cap at 3 standard deviations
                        mean_val = df_cleaned[col].mean()
                        std_val = df_cleaned[col].std()
                        df_cleaned.loc[df_cleaned[col] < mean_val - 3*std_val, col] = mean_val - 3*std_val
                        df_cleaned.loc[df_cleaned[col] > mean_val + 3*std_val, col] = mean_val + 3*std_val
            
            total_outliers = sum([info['count'] for info in outlier_info.values()])
            self.logger.info(f"Detected and capped {total_outliers} outliers using {method} method")
            
            return df_cleaned, outlier_info
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {str(e)}")
            raise
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale features using the specified scaling method.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features (optional)
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Scaled training and test features
        """
        try:
            # Fit scaler on training data
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            self.logger.info(f"Features scaled using {self.scaling_method} scaler")
            
            X_test_scaled = None
            if X_test is not None:
                X_test_scaled = pd.DataFrame(
                    self.scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                self.logger.info("Test features scaled using fitted scaler")
            
            return X_train_scaled, X_test_scaled
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                  random_state: int = 42, stratify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            random_state (int): Random state for reproducibility
            stratify (bool): Whether to stratify split for regression (creates bins)
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        try:
            stratify_param = None
            if stratify:
                # Create bins for stratification in regression
                y_binned = pd.cut(y, bins=10, labels=False)
                stratify_param = y_binned
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=stratify_param
            )
            
            self.logger.info(f"Data split into train: {X_train.shape}, test: {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def preprocess_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2, handle_outliers: bool = True,
                          outlier_method: str = "iqr") -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Test set proportion
            handle_outliers (bool): Whether to handle outliers
            outlier_method (str): Outlier detection method
            
        Returns:
            Dict: Preprocessed data and metadata
        """
        try:
            self.logger.info("Starting preprocessing pipeline...")
            
            # Combine features and target for preprocessing
            df = X.copy()
            df['target'] = y
            
            # 1. Data quality analysis
            quality_analysis = self.analyze_data_quality(df)
            
            # 2. Handle missing values
            df_cleaned = self.handle_missing_values(df)
            
            # 3. Handle outliers if requested
            outlier_info = {}
            if handle_outliers:
                df_cleaned, outlier_info = self.detect_and_handle_outliers(
                    df_cleaned, method=outlier_method
                )
            
            # 4. Separate features and target
            X_cleaned = df_cleaned.drop('target', axis=1)
            y_cleaned = df_cleaned['target']
            
            # 5. Split data
            X_train, X_test, y_train, y_test = self.split_data(
                X_cleaned, y_cleaned, test_size=test_size
            )
            
            # 6. Scale features
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
            
            # Prepare results
            results = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_unscaled': X_train,
                'X_test_unscaled': X_test,
                'quality_analysis': quality_analysis,
                'outlier_info': outlier_info,
                'preprocessing_info': {
                    'scaling_method': self.scaling_method,
                    'outlier_method': outlier_method if handle_outliers else None,
                    'test_size': test_size,
                    'original_shape': df.shape,
                    'final_shape': X_cleaned.shape
                }
            }
            
            self.logger.info("Preprocessing pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the fitted preprocessor objects.
        
        Args:
            filepath (str): Path to save the preprocessor
        """
        try:
            preprocessor_data = {
                'scaler': self.scaler,
                'imputer': self.imputer,
                'scaling_method': self.scaling_method
            }
            
            joblib.dump(preprocessor_data, filepath)
            self.logger.info(f"Preprocessor saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def load_preprocessor(self, filepath: str) -> None:
        """
        Load a fitted preprocessor.
        
        Args:
            filepath (str): Path to the saved preprocessor
        """
        try:
            preprocessor_data = joblib.load(filepath)
            self.scaler = preprocessor_data['scaler']
            self.imputer = preprocessor_data['imputer']
            self.scaling_method = preprocessor_data['scaling_method']
            
            self.logger.info(f"Preprocessor loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {str(e)}")
            raise


def main():
    """
    Main function to demonstrate preprocessing pipeline.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import data ingestion
    from .data_ingestion import DataIngestion
    
    # Load data
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(scaling_method="standard")
    
    # Run preprocessing pipeline
    results = preprocessor.preprocess_pipeline(X, y)
    
    # Display results
    print("Preprocessing Results:")
    print(f"Training set shape: {results['X_train'].shape}")
    print(f"Test set shape: {results['X_test'].shape}")
    print(f"Quality analysis: {results['quality_analysis']['missing_values']}")


if __name__ == "__main__":
    main()