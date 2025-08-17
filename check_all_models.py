#!/usr/bin/env python3
"""
Check all XGBoost model files to see which one has the feature mismatch.
"""

import sys
import os
sys.path.append('.')
import joblib
import pandas as pd
from src.features.feature_engineering import FeaturePipeline
from src.data.data_ingestion import DataIngestion

def check_all_models():
    """Check all saved models and their expected features."""
    
    print("🔍 Checking all saved XGBoost models...")
    
    # Get sample data and feature pipeline
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    sample_data = X.head(1)
    
    # Load saved feature pipeline
    feature_pipeline = FeaturePipeline()
    feature_pipeline.load_pipeline('models/trained/feature_pipeline.pkl')
    engineered_features = feature_pipeline.transform(sample_data)
    
    print(f"Feature pipeline produces: {list(engineered_features.columns)}")
    print(f"Number of features: {engineered_features.shape[1]}")
    
    # Check all XGBoost model files
    xgb_files = ['xgboost.pkl', 'xgboost_best.pkl']
    
    for model_file in xgb_files:
        model_path = f'models/trained/{model_file}'
        if os.path.exists(model_path):
            print(f"\n📊 Checking {model_file}:")
            try:
                model = joblib.load(model_path)
                
                if hasattr(model, 'feature_names_in_'):
                    expected_features = list(model.feature_names_in_)
                    print(f"   Expected features: {expected_features}")
                    print(f"   Number of expected features: {len(expected_features)}")
                    
                    # Check if features match
                    pipeline_features = list(engineered_features.columns)
                    if set(expected_features) == set(pipeline_features):
                        print("   ✅ Features match perfectly!")
                        
                        # Test prediction
                        try:
                            # Reorder features to match model expectation
                            ordered_features = engineered_features[expected_features]
                            pred = model.predict(ordered_features)
                            print(f"   ✅ Prediction successful: {pred[0]:.3f}")
                        except Exception as e:
                            print(f"   ❌ Prediction failed: {e}")
                    else:
                        print("   ❌ Features don't match!")
                        missing = set(expected_features) - set(pipeline_features)
                        extra = set(pipeline_features) - set(expected_features)
                        if missing:
                            print(f"   Missing: {missing}")
                        if extra:
                            print(f"   Extra: {extra}")
                else:
                    print("   ⚠️ Model doesn't have feature_names_in_ attribute")
                    
            except Exception as e:
                print(f"   ❌ Error loading model: {e}")
        else:
            print(f"\n❌ {model_file} not found")

if __name__ == "__main__":
    check_all_models()