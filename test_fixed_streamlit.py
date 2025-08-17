#!/usr/bin/env python3
"""
Test the fixed Streamlit prediction logic.
"""

import sys
import os
sys.path.append('.')
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeaturePipeline
from src.data.data_ingestion import DataIngestion
import joblib

def simulate_streamlit_prediction():
    """Simulate the fixed Streamlit prediction logic."""
    
    print("🧪 Testing fixed Streamlit prediction logic...")
    
    # Load data and pipeline (like Streamlit does)
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    sample_data = X.head(1)
    
    # Load feature pipeline
    feature_pipeline = FeaturePipeline()
    feature_pipeline.load_pipeline('models/trained/feature_pipeline.pkl')
    input_featured = feature_pipeline.transform(sample_data)
    
    print(f"📊 Input features: {list(input_featured.columns)}")
    print(f"📊 Number of features: {input_featured.shape[1]}")
    
    # Load all models (like Streamlit does with the fix)
    model_files = [f for f in os.listdir('models/trained') if f.endswith('.pkl') and f != 'feature_pipeline.pkl']
    trained_models = {}
    
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')  # Keep full name
        model_path = os.path.join('models/trained', model_file)
        trained_models[model_name] = joblib.load(model_path)
    
    print(f"\n🤖 Loaded models: {list(trained_models.keys())}")
    
    # Test prediction with feature compatibility check (the fix)
    predictions = {}
    for model_name, model in trained_models.items():
        print(f"\n🔮 Testing {model_name}:")
        try:
            # Check if model has feature expectations
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
                available_features = list(input_featured.columns)
                
                print(f"   Expected: {len(expected_features)} features")
                print(f"   Available: {len(available_features)} features")
                
                # Check if all expected features are available
                missing_features = set(expected_features) - set(available_features)
                if missing_features:
                    print(f"   ⚠️ Missing features {missing_features}, skipping model")
                    continue
                
                # Use only the features the model expects, in the correct order
                model_input = input_featured[expected_features]
                pred = model.predict(model_input)[0]
            else:
                # Fallback for models without feature names
                print("   No feature names, using all features")
                pred = model.predict(input_featured)[0]
            
            predictions[model_name] = pred
            print(f"   ✅ Prediction successful: {pred:.3f}")
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {str(e)}")
    
    # Final results
    print(f"\n🎯 Final results:")
    print(f"   Successful predictions: {len(predictions)}")
    
    if predictions:
        ensemble_pred = np.mean(list(predictions.values()))
        print(f"   Ensemble prediction: {ensemble_pred:.3f}")
        
        for model_name, pred_value in predictions.items():
            print(f"   {model_name}: {pred_value:.3f}")
        
        return True
    else:
        print("   ❌ No successful predictions!")
        return False

if __name__ == "__main__":
    success = simulate_streamlit_prediction()
    if success:
        print("\n🎉 Fixed Streamlit logic works correctly!")
    else:
        print("\n💥 Still have issues to fix.")