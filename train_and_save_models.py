"""
Script to train models and save them with the proper feature pipeline for the web app.
"""

import sys
import os
sys.path.append('src')

from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.feature_engineering import FeaturePipeline
from src.models.ml_models import ModelTrainer
import joblib

def train_and_save_models():
    """Train models and save them with the feature pipeline for web app compatibility."""
    
    print("🚀 Training and saving models for web app...")
    
    # 1. Load data
    print("📊 Loading data...")
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    
    # 2. Preprocess data
    print("🧹 Preprocessing data...")
    preprocessor = DataPreprocessor()
    results = preprocessor.preprocess_pipeline(X, y, test_size=0.2)
    
    # 3. Feature engineering
    print("🔧 Engineering features...")
    feature_pipeline = FeaturePipeline(
        engineer_config={'create_interaction_features': True, 'create_polynomial_features': False},
        selector_config={'selection_method': 'f_test', 'k_features': 15}
    )
    
    X_train_featured = feature_pipeline.fit_transform(
        results['X_train_unscaled'], 
        results['y_train']
    )
    X_test_featured = feature_pipeline.transform(results['X_test_unscaled'])
    
    print(f"   Original features: {results['X_train_unscaled'].shape[1]}")
    print(f"   Engineered features: {X_train_featured.shape[1]}")
    
    # 4. Train models
    print("🤖 Training models...")
    trainer = ModelTrainer(cv_folds=3)
    
    # Train key models for web app
    key_models = ['linear_regression', 'random_forest', 'xgboost']
    original_models = trainer.models.copy()
    trainer.models = {k: v for k, v in trainer.models.items() if k in key_models}
    
    training_results = trainer.train_all_models(
        X_train_featured, results['y_train'],
        X_test_featured, results['y_test']
    )
    
    # 5. Create models directory
    os.makedirs('models/trained', exist_ok=True)
    
    # 6. Save feature pipeline
    print("💾 Saving feature pipeline...")
    feature_pipeline.save_pipeline('models/trained/feature_pipeline.pkl')
    
    # 7. Save individual models
    print("💾 Saving trained models...")
    for model_name in training_results.keys():
        model_path = f'models/trained/{model_name}.pkl'
        trainer.save_model(model_name, model_path)
        print(f"   ✅ Saved {model_name}")
    
    # 8. Save model compatibility info
    model_info = {
        'feature_pipeline_path': 'models/trained/feature_pipeline.pkl',
        'models': {
            name: {
                'path': f'models/trained/{name}.pkl',
                'performance': {
                    'cv_rmse': results['cv_scores']['mean_rmse'],
                    'test_rmse': results['test_metrics']['rmse'],
                    'test_r2': results['test_metrics']['r2']
                }
            }
            for name, results in training_results.items()
        },
        'feature_names': list(X_train_featured.columns),
        'original_features': list(results['X_train_unscaled'].columns)
    }
    
    with open('models/trained/model_info.json', 'w') as f:
        import json
        json.dump(model_info, f, indent=2)
    
    print("📋 Performance Summary:")
    for name, result in training_results.items():
        cv_rmse = result['cv_scores']['mean_rmse']
        test_rmse = result['test_metrics']['rmse']
        test_r2 = result['test_metrics']['r2']
        print(f"   {name:15} | CV RMSE: {cv_rmse:.4f} | Test RMSE: {test_rmse:.4f} | R²: {test_r2:.4f}")
    
    print("\n✅ Models and pipeline saved successfully!")
    print("🌐 Web app is now ready for predictions with proper feature engineering!")
    
    return training_results

if __name__ == "__main__":
    train_and_save_models()