#!/usr/bin/env python3
"""
Simple test script to verify that the XGBoost feature mismatch has been fixed.
"""

import sys
import os
sys.path.append('.')

from app.streamlit_app import HousePricePredictionApp
import pandas as pd

def test_prediction():
    """Test the prediction functionality without the streamlit UI."""
    
    print("🧪 Testing prediction functionality...")
    
    # Initialize the app
    app = HousePricePredictionApp()
    
    # Check if models are loaded
    if not app.models_loaded:
        print("❌ Models not loaded!")
        return False
    
    print("✅ Models loaded successfully")
    print(f"   Available models: {list(app.trained_models.keys())}")
    
    # Create test input data (using California average values)
    test_input = pd.DataFrame({
        'MedInc': [3.87],          # Median income
        'HouseAge': [28.6],        # House age
        'AveRooms': [5.43],        # Average rooms
        'AveBedrms': [1.1],        # Average bedrooms 
        'Population': [3119.0],    # Population
        'AveOccup': [3.07],        # Average occupancy
        'Latitude': [34.2],        # Latitude
        'Longitude': [-118.5]      # Longitude
    })
    
    print(f"📊 Test input shape: {test_input.shape}")
    print(f"📊 Test input columns: {list(test_input.columns)}")
    
    # Test feature pipeline
    if hasattr(app, 'feature_pipeline') and app.feature_pipeline:
        try:
            print("🔧 Testing feature engineering...")
            engineered_features = app.feature_pipeline.transform(test_input)
            print(f"✅ Feature engineering successful: {test_input.shape[1]} → {engineered_features.shape[1]} features")
            print(f"   Engineered feature names: {list(engineered_features.columns)}")
        except Exception as e:
            print(f"❌ Feature engineering failed: {str(e)}")
            return False
    
    # Test prediction with each model
    print("🔮 Testing predictions...")
    
    try:
        # Call the make_prediction method (this will print results to stdout)
        app.make_prediction(test_input)
        print("✅ Prediction test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    if success:
        print("\n🎉 All tests passed! XGBoost feature mismatch has been resolved.")
        sys.exit(0)
    else:
        print("\n💥 Tests failed! There are still issues to resolve.")
        sys.exit(1)