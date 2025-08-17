#!/usr/bin/env python3
"""
Test script to verify how Streamlit app loads the feature pipeline vs other methods.
"""

import sys
import os
sys.path.append('.')
import pandas as pd
from src.features.feature_engineering import FeaturePipeline
from src.data.data_ingestion import DataIngestion

def test_different_loading_methods():
    """Test different ways of loading the feature pipeline."""
    
    print("🧪 Testing different feature pipeline loading methods...")
    
    # Get sample data
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    sample_data = X.head(1)
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {list(sample_data.columns)}")
    
    # Method 1: How Streamlit currently loads (PROBLEMATIC)
    print("\n1. Streamlit method (create fresh instance then load):")
    try:
        streamlit_pipeline = FeaturePipeline()  # Fresh instance with defaults
        streamlit_pipeline.load_pipeline('models/trained/feature_pipeline.pkl')
        streamlit_features = streamlit_pipeline.transform(sample_data)
        print(f"   ✅ Success: {streamlit_features.shape[1]} features")
        print(f"   Features: {list(streamlit_features.columns)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: Direct joblib load (CORRECT)
    print("\n2. Direct joblib load method:")
    try:
        import joblib
        direct_pipeline = joblib.load('models/trained/feature_pipeline.pkl')
        direct_features = direct_pipeline.transform(sample_data)
        print(f"   ✅ Success: {direct_features.shape[1]} features")
        print(f"   Features: {list(direct_features.columns)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 3: Create fresh and fit (should be different)
    print("\n3. Fresh pipeline method:")
    try:
        fresh_pipeline = FeaturePipeline(
            engineer_config={'create_interaction_features': True, 'create_polynomial_features': False},
            selector_config={'selection_method': 'f_test', 'k_features': 15}
        )
        fresh_features = fresh_pipeline.fit_transform(sample_data, y.head(1))
        print(f"   ✅ Success: {fresh_features.shape[1]} features")
        print(f"   Features: {list(fresh_features.columns)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Compare results
    print("\n4. Comparison:")
    if 'streamlit_features' in locals() and 'direct_features' in locals():
        streamlit_cols = set(streamlit_features.columns)
        direct_cols = set(direct_features.columns)
        
        if streamlit_cols == direct_cols:
            print("   ✅ Streamlit method produces same features as direct load")
        else:
            print("   ❌ Streamlit method produces different features!")
            print(f"   Missing in streamlit: {direct_cols - streamlit_cols}")
            print(f"   Extra in streamlit: {streamlit_cols - direct_cols}")

if __name__ == "__main__":
    test_different_loading_methods()