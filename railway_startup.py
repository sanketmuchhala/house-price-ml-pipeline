#!/usr/bin/env python3
"""
Railway startup script to ensure models are trained and available for the web app.
"""

import os
import sys
import subprocess

def ensure_models_exist():
    """Ensure trained models exist, train them if they don't."""
    
    print("🚀 Railway startup: Checking for trained models...")
    
    models_dir = "models/trained"
    required_files = [
        "feature_pipeline.pkl",
        "xgboost.pkl", 
        "linear_regression.pkl",
        "random_forest.pkl",
        "model_info.json"
    ]
    
    # Check if all required model files exist
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(models_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️ Missing model files: {missing_files}")
        print("🏗️ Training models for Railway deployment...")
        
        try:
            # Train models using the training script
            result = subprocess.run([sys.executable, "train_and_save_models.py"], 
                                  capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Models trained successfully!")
                print(result.stdout)
            else:
                print("❌ Model training failed!")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Model training timed out!")
            return False
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            return False
    else:
        print("✅ All required model files found!")
    
    return True

def start_streamlit():
    """Start the Streamlit application."""
    
    print("🌐 Starting Streamlit application...")
    
    # Get port from environment (Railway sets this)
    port = os.environ.get("PORT", "8080")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "app/streamlit_app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"🚀 Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("👋 Streamlit application stopped.")

if __name__ == "__main__":
    print("🚂 Railway deployment startup...")
    
    # Ensure models exist
    if ensure_models_exist():
        # Start Streamlit
        start_streamlit()
    else:
        print("💥 Failed to prepare models. Exiting.")
        sys.exit(1)