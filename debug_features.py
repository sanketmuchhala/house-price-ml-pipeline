#!/usr/bin/env python3
"""
Debug script to check what features the saved XGBoost model expects vs what the pipeline produces.
"""

import sys
import os
sys.path.append('.')
import joblib
import pandas as pd
from src.features.feature_engineering import FeaturePipeline
from src.data.data_ingestion import DataIngestion

def debug_feature_mismatch():
    """Debug the feature mismatch between saved model and pipeline."""
    
    print("🔍 Debugging feature mismatch...")
    
    # 1. Load the saved XGBoost model
    print("\n1. Loading saved XGBoost model...")
    xgb_model = joblib.load('models/trained/xgboost.pkl')
    if hasattr(xgb_model, 'feature_names_in_'):
        saved_model_features = list(xgb_model.feature_names_in_)
        print(f"   XGBoost model expects {len(saved_model_features)} features:")
        for i, feat in enumerate(saved_model_features):
            print(f"   [{i+1:2d}] {feat}")
    else:
        print("   XGBoost model doesn't have feature_names_in_ attribute")
    
    # 2. Load the saved feature pipeline 
    print("\n2. Loading saved feature pipeline...")
    try:
        saved_pipeline = FeaturePipeline()
        saved_pipeline.load_pipeline('models/trained/feature_pipeline.pkl')
        print("   ✅ Saved feature pipeline loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load saved pipeline: {e}")
        return
    
    # 3. Create a fresh feature pipeline with same config
    print("\n3. Creating fresh feature pipeline...")
    fresh_pipeline = FeaturePipeline(
        engineer_config={'create_interaction_features': True, 'create_polynomial_features': False},
        selector_config={'selection_method': 'f_test', 'k_features': 15}
    )
    
    # 4. Load sample data
    print("\n4. Loading sample data...")
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    sample_data = X.head(5)
    print(f"   Sample data shape: {sample_data.shape}")
    print(f"   Sample data columns: {list(sample_data.columns)}")
    
    # 5. Transform with saved pipeline
    print("\n5. Testing saved pipeline transformation...")
    try:
        saved_features = saved_pipeline.transform(sample_data)
        print(f"   Saved pipeline output: {saved_features.shape[1]} features")
        saved_feature_names = list(saved_features.columns)
        print("   Saved pipeline features:")
        for i, feat in enumerate(saved_feature_names):
            print(f"   [{i+1:2d}] {feat}")
    except Exception as e:
        print(f"   ❌ Saved pipeline failed: {e}")
        return
    
    # 6. Transform with fresh pipeline (fit first)
    print("\n6. Testing fresh pipeline transformation...")
    try:
        fresh_features = fresh_pipeline.fit_transform(sample_data, y.head(5))
        print(f"   Fresh pipeline output: {fresh_features.shape[1]} features")
        fresh_feature_names = list(fresh_features.columns)
        print("   Fresh pipeline features:")
        for i, feat in enumerate(fresh_feature_names):
            print(f"   [{i+1:2d}] {feat}")
    except Exception as e:
        print(f"   ❌ Fresh pipeline failed: {e}")
        return
    
    # 7. Compare features
    print("\n7. Feature comparison:")
    print(f"   Model expects: {len(saved_model_features)} features")
    print(f"   Saved pipeline: {len(saved_feature_names)} features")  
    print(f"   Fresh pipeline: {len(fresh_feature_names)} features")
    
    # Check which features are missing/extra
    model_set = set(saved_model_features)
    saved_set = set(saved_feature_names)
    fresh_set = set(fresh_feature_names)
    
    print(f"\n   Features model expects but saved pipeline doesn't have:")
    missing_in_saved = model_set - saved_set
    for feat in missing_in_saved:
        print(f"   - {feat}")
    
    print(f"\n   Features saved pipeline has but model doesn't expect:")
    extra_in_saved = saved_set - model_set  
    for feat in extra_in_saved:
        print(f"   - {feat}")
    
    print(f"\n   Features fresh pipeline has but model doesn't expect:")
    extra_in_fresh = fresh_set - model_set
    for feat in extra_in_fresh:
        print(f"   - {feat}")
    
    print(f"\n   Features model expects but fresh pipeline doesn't have:")
    missing_in_fresh = model_set - fresh_set
    for feat in missing_in_fresh:
        print(f"   - {feat}")
    
    # 8. Test prediction with correctly matched features
    print("\n8. Testing prediction with feature matching...")
    try:
        # Use only the features the model expects, in the right order
        if model_set == saved_set:
            # Reorder saved features to match model
            matched_features = saved_features[saved_model_features]
            pred = xgb_model.predict(matched_features)
            print(f"   ✅ Prediction successful with saved pipeline: {pred[0]:.3f}")
        else:
            print(f"   ❌ Feature sets don't match, can't test prediction")
            
    except Exception as e:
        print(f"   ❌ Prediction failed: {e}")

if __name__ == "__main__":
    debug_feature_mismatch()