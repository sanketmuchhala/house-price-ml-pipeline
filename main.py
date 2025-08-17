"""
Main execution script for the End-to-End ML Pipeline.
This script demonstrates the complete workflow from data ingestion to model deployment.
"""

import os
import sys
import logging
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import pipeline components
from src.utils.logging_config import setup_pipeline_logging
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.feature_engineering import FeaturePipeline
from src.models.ml_models import ModelTrainer
from src.models.hyperparameter_tuning import HyperparameterTuner
from src.models.model_evaluation import ModelEvaluator
from src.visualization.eda import EDAVisualizer


def main():
    """
    Execute the complete ML pipeline workflow.
    """
    # Setup logging
    logger_system = setup_pipeline_logging(log_level="INFO")
    logger = logger_system.get_logger(__name__)
    
    logger.info("="*70)
    logger.info("STARTING END-TO-END ML PIPELINE")
    logger.info("="*70)
    
    start_time = datetime.now()
    
    try:
        # 1. DATA INGESTION
        logger.info("Step 1: Data Ingestion")
        data_ingestion = DataIngestion()
        X, y = data_ingestion.download_data(save_to_disk=True)
        logger.info(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # 2. EXPLORATORY DATA ANALYSIS
        logger.info("Step 2: Exploratory Data Analysis")
        df = X.copy()
        df['target'] = y
        
        eda = EDAVisualizer()
        eda_report = eda.generate_comprehensive_report(df, target_col='target')
        logger.info(f"✅ EDA completed: {len(eda_report['insights'])} insights generated")
        
        # 3. DATA PREPROCESSING
        logger.info("Step 3: Data Preprocessing")
        preprocessor = DataPreprocessor(scaling_method="standard")
        preprocessing_results = preprocessor.preprocess_pipeline(
            X, y, test_size=0.2, handle_outliers=True
        )
        logger.info(f"✅ Preprocessing completed: Train({preprocessing_results['X_train'].shape[0]}), Test({preprocessing_results['X_test'].shape[0]})")
        
        # 4. FEATURE ENGINEERING
        logger.info("Step 4: Feature Engineering")
        feature_pipeline = FeaturePipeline(
            engineer_config={'create_interaction_features': True, 'create_polynomial_features': False},
            selector_config={'selection_method': 'f_test', 'k_features': 15}
        )
        
        X_train_featured = feature_pipeline.fit_transform(
            preprocessing_results['X_train_unscaled'], 
            preprocessing_results['y_train']
        )
        X_test_featured = feature_pipeline.transform(preprocessing_results['X_test_unscaled'])
        
        feature_info = feature_pipeline.get_feature_info()
        logger.info(f"✅ Feature engineering completed: {feature_info['n_original_features']} → {feature_info['n_selected_features']} features")
        
        # 5. MODEL TRAINING
        logger.info("Step 5: Model Training")
        trainer = ModelTrainer(cv_folds=5, random_state=42)
        
        # Train a subset of models for demonstration
        models_to_train = ['linear_regression', 'ridge', 'random_forest', 'xgboost']
        original_models = trainer.models.copy()
        trainer.models = {k: v for k, v in trainer.models.items() if k in models_to_train}
        
        training_results = trainer.train_all_models(
            X_train_featured, preprocessing_results['y_train'],
            X_test_featured, preprocessing_results['y_test']
        )
        
        # Model comparison
        comparison = trainer.compare_models()
        logger.info(f"✅ Model training completed: {len(training_results)} models trained")
        
        # Get best model
        best_name, best_model = trainer.get_best_model()
        logger.info(f"🏆 Best model: {best_name}")
        
        # 6. HYPERPARAMETER TUNING (Optional - for demonstration)
        logger.info("Step 6: Hyperparameter Tuning (Quick Demo)")
        tuner = HyperparameterTuner(cv_folds=3)
        
        # Quick tuning for Random Forest
        if 'random_forest' in training_results:
            optuna_results = tuner.optuna_tuning(
                'random_forest', 
                X_train_featured, 
                preprocessing_results['y_train'],
                n_trials=20  # Limited for demo
            )
            logger.info(f"✅ Hyperparameter tuning completed: Best RMSE = {optuna_results['best_score']:.4f}")
        
        # 7. MODEL EVALUATION
        logger.info("Step 7: Model Evaluation")
        evaluator = ModelEvaluator()
        
        # Evaluate all trained models
        for model_name in training_results.keys():
            model = trainer.trained_models[model_name]
            evaluation = evaluator.evaluate_model(
                model, model_name,
                X_train_featured, preprocessing_results['y_train'],
                X_test_featured, preprocessing_results['y_test']
            )
            
            # Save evaluation results
            evaluator.save_evaluation_results(model_name)
        
        # Generate comprehensive report for best model
        best_report = evaluator.generate_model_report(best_name)
        logger.info(f"✅ Model evaluation completed for {len(training_results)} models")
        
        # 8. MODEL PERSISTENCE
        logger.info("Step 8: Model Persistence")
        os.makedirs('models/trained', exist_ok=True)
        
        # Save best model
        trainer.save_model(best_name, f'models/trained/{best_name}_best.pkl')
        
        # Save feature pipeline
        feature_pipeline.save_pipeline('models/trained/feature_pipeline.pkl')
        
        # Save preprocessor
        preprocessor.save_preprocessor('models/trained/preprocessor.pkl')
        
        logger.info("✅ Models and pipelines saved successfully")
        
        # 9. FINAL SUMMARY
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info("="*70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} original features")
        logger.info(f"Final features: {feature_info['n_selected_features']}")
        logger.info(f"Models trained: {len(training_results)}")
        logger.info(f"Best model: {best_name}")
        logger.info(f"Best CV RMSE: {training_results[best_name]['cv_scores']['mean_rmse']:.4f}")
        logger.info(f"Test RMSE: {training_results[best_name]['test_metrics']['rmse']:.4f}")
        logger.info(f"Test R² Score: {training_results[best_name]['test_metrics']['r2']:.4f}")
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"🏆 Best model: {best_name}")
        print(f"📊 Test RMSE: {training_results[best_name]['test_metrics']['rmse']:.4f}")
        print(f"📈 Test R²: {training_results[best_name]['test_metrics']['r2']:.4f}")
        print("\n📁 Generated artifacts:")
        print("   • Trained models in models/trained/")
        print("   • Evaluation results in models/artifacts/")
        print("   • Visualizations in data/processed/")
        print("   • Logs in logs/")
        print("\n🚀 Next steps:")
        print("   • Run the web app: python run_app.py")
        print("   • View logs: tail -f logs/ml_pipeline.log")
        print("   • Run tests: python tests/run_tests.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("Check logs/errors.log for detailed error information")
        return False
    
    finally:
        # Restore original models if modified
        if 'original_models' in locals():
            trainer.models = original_models


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)