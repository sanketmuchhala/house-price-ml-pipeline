"""
Streamlit web application for house price prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.feature_engineering import FeaturePipeline
from src.models.ml_models import ModelTrainer
from src.models.model_evaluation import ModelEvaluator


class HousePricePredictionApp:
    """
    Streamlit application for house price prediction.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.setup_page_config()
        self.load_models_and_data()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="California House Price Predictor",
            page_icon="house",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .prediction-result {
            font-size: 2rem;
            font-weight: bold;
            color: #2e8b57;
            text-align: center;
            padding: 1rem;
            background-color: #f0fff0;
            border-radius: 0.5rem;
            border: 2px solid #2e8b57;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_models_and_data(self):
        """Load trained models and sample data."""
        try:
            # Initialize data ingestion
            self.data_ingestion = DataIngestion()
            
            # Load sample data for feature ranges
            X, y = self.data_ingestion.load_data()
            self.sample_data = X.copy()
            self.target_data = y
            
            # Calculate feature statistics for input validation
            self.feature_stats = {
                'MedInc': {'min': X['MedInc'].min(), 'max': X['MedInc'].max(), 'mean': X['MedInc'].mean()},
                'HouseAge': {'min': X['HouseAge'].min(), 'max': X['HouseAge'].max(), 'mean': X['HouseAge'].mean()},
                'AveRooms': {'min': X['AveRooms'].min(), 'max': X['AveRooms'].max(), 'mean': X['AveRooms'].mean()},
                'AveBedrms': {'min': X['AveBedrms'].min(), 'max': X['AveBedrms'].max(), 'mean': X['AveBedrms'].mean()},
                'Population': {'min': X['Population'].min(), 'max': X['Population'].max(), 'mean': X['Population'].mean()},
                'AveOccup': {'min': X['AveOccup'].min(), 'max': X['AveOccup'].max(), 'mean': X['AveOccup'].mean()},
                'Latitude': {'min': X['Latitude'].min(), 'max': X['Latitude'].max(), 'mean': X['Latitude'].mean()},
                'Longitude': {'min': X['Longitude'].min(), 'max': X['Longitude'].max(), 'mean': X['Longitude'].mean()}
            }
            
            # Try to load pre-trained models (if available)
            self.models_loaded = self.try_load_pretrained_models()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            self.sample_data = None
            self.models_loaded = False
    
    def try_load_pretrained_models(self):
        """Try to load pre-trained models and feature pipeline."""
        try:
            model_files = [f for f in os.listdir('models/trained') if f.endswith('.pkl') and f != 'feature_pipeline.pkl']
            feature_pipeline_path = os.path.join('models/trained', 'feature_pipeline.pkl')
            
            if model_files and os.path.exists(feature_pipeline_path):
                # Load feature pipeline
                from src.features.feature_engineering import FeaturePipeline
                self.feature_pipeline = FeaturePipeline()
                self.feature_pipeline.load_pipeline(feature_pipeline_path)
                
                # Load models
                self.trained_models = {}
                for model_file in model_files:
                    model_name = model_file.replace('.pkl', '')  # Keep full name to avoid conflicts
                    model_path = os.path.join('models/trained', model_file)
                    self.trained_models[model_name] = joblib.load(model_path)
                
                # Load model info if available
                model_info_path = os.path.join('models/trained', 'model_info.json')
                if os.path.exists(model_info_path):
                    import json
                    with open(model_info_path, 'r') as f:
                        self.model_info = json.load(f)
                else:
                    self.model_info = {}
                
                return True
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def run(self):
        """Run the Streamlit application."""
        # Header
        st.markdown('<h1 class="main-header">California House Price Predictor</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        Welcome to the California House Price Prediction app! This application uses machine learning 
        to predict median house values based on various features of California housing districts.
        """)
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["Home", "Make Prediction", "Data Exploration", "Model Performance", "About"]
        )
        
        if page == "Home":
            self.show_home_page()
        elif page == "Make Prediction":
            self.show_prediction_page()
        elif page == "Data Exploration":
            self.show_data_exploration_page()
        elif page == "Model Performance":
            self.show_model_performance_page()
        elif page == "About":
            self.show_about_page()
    
    def show_home_page(self):
        """Display the home page."""
        st.header("Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            if self.sample_data is not None:
                st.info(f"""
                **California Housing Dataset**
                - **Samples**: {len(self.sample_data):,}
                - **Features**: {len(self.sample_data.columns)}
                - **Target**: Median house value (in hundreds of thousands)
                - **Data Source**: Sklearn built-in dataset
                """)
                
                st.subheader("Features Description")
                feature_descriptions = {
                    'MedInc': 'Median income in block group',
                    'HouseAge': 'Median house age in block group',
                    'AveRooms': 'Average number of rooms per household',
                    'AveBedrms': 'Average number of bedrooms per household',
                    'Population': 'Block group population',
                    'AveOccup': 'Average number of household members',
                    'Latitude': 'Block group latitude',
                    'Longitude': 'Block group longitude'
                }
                
                for feature, description in feature_descriptions.items():
                    st.write(f"**{feature}**: {description}")
        
        with col2:
            st.subheader("Machine Learning Pipeline")
            st.success("""
            **Our ML Pipeline includes:**
            
            1. **Data Ingestion** - Loading California housing data
            2. **Data Preprocessing** - Cleaning and scaling features
            3. **Feature Engineering** - Creating new meaningful features
            4. **Model Training** - Multiple algorithms comparison
            5. **Hyperparameter Tuning** - Optimizing model performance
            6. **Model Evaluation** - Comprehensive performance analysis
            7. **Web Deployment** - This interactive application
            """)
            
            if self.sample_data is not None:
                st.subheader("Quick Stats")
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Avg House Value", f"${self.target_data.mean():.0f}k")
                    st.metric("Max House Value", f"${self.target_data.max():.0f}k")
                with col2_2:
                    st.metric("Min House Value", f"${self.target_data.min():.0f}k")
                    st.metric("Std Deviation", f"${self.target_data.std():.0f}k")
    
    def show_prediction_page(self):
        """Display the prediction page."""
        st.header("Make House Price Prediction")
        
        if not self.models_loaded:
            st.warning("""
            **No pre-trained models found!** 
            
            To use the prediction feature, you need to train models first. 
            Please run the training pipeline to generate trained models.
            
            You can still explore the data and understand the features below.
            """)
            
            # Show training instructions
            with st.expander("How to train models"):
                st.code("""
# Run the following commands to train models:
cd /path/to/your/project
python -m src.data.data_ingestion
python -m src.models.ml_models
                """)
        
        # Input form
        st.subheader("Enter House Characteristics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Economic & Demographic Features**")
            
            median_income = st.slider(
                "Median Income (tens of thousands)",
                min_value=float(self.feature_stats['MedInc']['min']),
                max_value=float(self.feature_stats['MedInc']['max']),
                value=float(self.feature_stats['MedInc']['mean']),
                step=0.1,
                help="Median income in the block group (in tens of thousands of dollars)"
            )
            
            population = st.number_input(
                "Population",
                min_value=int(self.feature_stats['Population']['min']),
                max_value=int(self.feature_stats['Population']['max']),
                value=int(self.feature_stats['Population']['mean']),
                help="Total population in the block group"
            )
            
            ave_occup = st.slider(
                "Average Occupancy",
                min_value=float(self.feature_stats['AveOccup']['min']),
                max_value=min(10.0, float(self.feature_stats['AveOccup']['max'])),  # Cap for better UX
                value=float(self.feature_stats['AveOccup']['mean']),
                step=0.1,
                help="Average number of people per household"
            )
        
        with col2:
            st.write("**Housing Features**")
            
            house_age = st.slider(
                "House Age (years)",
                min_value=float(self.feature_stats['HouseAge']['min']),
                max_value=float(self.feature_stats['HouseAge']['max']),
                value=float(self.feature_stats['HouseAge']['mean']),
                step=1.0,
                help="Median age of houses in the block group"
            )
            
            ave_rooms = st.slider(
                "Average Rooms per House",
                min_value=float(self.feature_stats['AveRooms']['min']),
                max_value=min(15.0, float(self.feature_stats['AveRooms']['max'])),  # Cap for better UX
                value=float(self.feature_stats['AveRooms']['mean']),
                step=0.1,
                help="Average number of rooms per household"
            )
            
            ave_bedrms = st.slider(
                "Average Bedrooms per House",
                min_value=float(self.feature_stats['AveBedrms']['min']),
                max_value=min(5.0, float(self.feature_stats['AveBedrms']['max'])),  # Cap for better UX
                value=float(self.feature_stats['AveBedrms']['mean']),
                step=0.1,
                help="Average number of bedrooms per household"
            )
        
        # Geographic features
        st.write("**Geographic Location**")
        
        # California city locations (latitude, longitude)
        california_cities = {
            "Los Angeles": (34.05, -118.24),
            "San Francisco": (37.77, -122.42),
            "San Diego": (32.72, -117.16),
            "Sacramento": (38.58, -121.49),
            "San Jose": (37.34, -121.89),
            "Fresno": (36.73, -119.79),
            "Long Beach": (33.77, -118.19),
            "Oakland": (37.80, -122.27),
            "Bakersfield": (35.37, -119.04),
            "Anaheim": (33.84, -117.91),
            "Riverside": (33.95, -117.40),
            "Stockton": (37.96, -121.29),
            "Chula Vista": (32.64, -117.08),
            "Irvine": (33.68, -117.83),
            "Fremont": (37.55, -121.99),
            "Santa Ana": (33.75, -117.87),
            "Average California Location": (34.2, -118.5)  # Default
        }
        
        selected_city = st.selectbox(
            "Select a California City/Region:",
            options=list(california_cities.keys()),
            index=len(california_cities) - 1,  # Default to "Average California Location"
            help="Choose a city to automatically set the geographic coordinates"
        )
        
        latitude, longitude = california_cities[selected_city]
        
        # Show selected coordinates for reference
        if selected_city != "Average California Location":
            st.info(f"{selected_city}: Lat {latitude:.2f}, Long {longitude:.2f}")
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'MedInc': [median_income],
            'HouseAge': [house_age],
            'AveRooms': [ave_rooms],
            'AveBedrms': [ave_bedrms],
            'Population': [population],
            'AveOccup': [ave_occup],
            'Latitude': [latitude],
            'Longitude': [longitude]
        })
        
        # Show input summary
        st.subheader("Input Summary")
        st.dataframe(input_data.T, use_container_width=True)
        
        # Prediction button
        if st.button("Predict House Price", type="primary"):
            if self.models_loaded:
                self.make_prediction(input_data)
            else:
                st.error("No trained models available for prediction!")
        
        # Show feature insights
        self.show_feature_insights(input_data.iloc[0])
    
    def make_prediction(self, input_data):
        """Make prediction using trained models."""
        try:
            st.subheader("Prediction Results")
            
            # Apply the same preprocessing and feature engineering pipeline used during training
            if hasattr(self, 'feature_pipeline') and self.feature_pipeline:
                st.info("Applying trained feature engineering pipeline...")
                try:
                    # Use the loaded feature pipeline
                    input_featured = self.feature_pipeline.transform(input_data)
                    st.success(f"Features engineered: {input_data.shape[1]} → {input_featured.shape[1]} features")
                    
                    # Show feature engineering details
                    if hasattr(self, 'model_info') and 'feature_names' in self.model_info:
                        with st.expander("Feature Engineering Details"):
                            st.write("**Original Features:**", self.model_info.get('original_features', []))
                            st.write("**Engineered Features:**", self.model_info.get('feature_names', []))
                    
                except Exception as e:
                    st.error(f"Feature engineering failed: {str(e)}")
                    st.info("Falling back to original features...")
                    input_featured = input_data
            else:
                st.warning("Feature pipeline not loaded. Using original features.")
                input_featured = input_data
            
            predictions = {}
            for model_name, model in self.trained_models.items():
                try:
                    # Check if model has feature expectations
                    if hasattr(model, 'feature_names_in_'):
                        expected_features = list(model.feature_names_in_)
                        available_features = list(input_featured.columns)
                        
                        # Check if all expected features are available
                        missing_features = set(expected_features) - set(available_features)
                        if missing_features:
                            st.warning(f"{model_name}: Missing features {missing_features}, skipping model")
                            continue
                        
                        # Use only the features the model expects, in the correct order
                        model_input = input_featured[expected_features]
                        pred = model.predict(model_input)[0]
                    else:
                        # Fallback for models without feature names
                        pred = model.predict(input_featured)[0]
                    
                    predictions[model_name] = pred
                except Exception as e:
                    st.warning(f"Could not get prediction from {model_name}: {str(e)}")
            
            if predictions:
                # Calculate ensemble prediction (average)
                ensemble_pred = np.mean(list(predictions.values()))
                
                # Display main prediction (model predicts in hundreds of thousands)
                predicted_value = ensemble_pred * 100000  # Convert to actual dollars
                st.markdown(f"""
                <div class="prediction-result">
                    Predicted House Value: ${predicted_value:,.0f}
                </div>
                """, unsafe_allow_html=True)
                
                # Show individual model predictions
                st.subheader("Individual Model Predictions")
                
                # Create DataFrame with predictions and performance info
                pred_data = []
                for model_name, pred_value in predictions.items():
                    row = {
                        'Model': model_name,
                        'Prediction ($)': f"${pred_value * 100000:,.0f}"
                    }
                    
                    # Add performance info if available
                    if hasattr(self, 'model_info') and model_name in self.model_info.get('models', {}):
                        perf = self.model_info['models'][model_name]['performance']
                        row['Test R²'] = f"{perf.get('test_r2', 0):.3f}"
                        row['Test RMSE'] = f"{perf.get('test_rmse', 0):.3f}"
                    
                    pred_data.append(row)
                
                # Display predictions table
                pred_df = pd.DataFrame(pred_data)
                st.dataframe(pred_df, use_container_width=True)
                
                # Create bar chart with actual values for comparison
                chart_data = []
                for model_name, pred_value in predictions.items():
                    chart_data.append({
                        'Model': model_name,
                        'Prediction': pred_value * 100000
                    })
                chart_df = pd.DataFrame(chart_data)
                
                fig = px.bar(chart_df, x='Model', y='Prediction',
                           title='Predictions by Different Models',
                           color='Prediction',
                           color_continuous_scale='viridis')
                fig.update_layout(height=400, yaxis_tickformat='$,.0f')
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction confidence
                pred_std = np.std(list(predictions.values())) * 100000  # Convert to dollars
                pred_range_low = (ensemble_pred - np.std(list(predictions.values()))) * 100000
                pred_range_high = (ensemble_pred + np.std(list(predictions.values()))) * 100000
                st.info(f"""
                **Prediction Confidence:**
                - Standard Deviation: ${pred_std:,.0f}
                - Prediction Range: ${pred_range_low:,.0f} - ${pred_range_high:,.0f}
                """)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    def show_feature_insights(self, input_row):
        """Show insights about the input features."""
        st.subheader("Feature Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Income analysis
            income_percentile = (self.sample_data['MedInc'] <= input_row['MedInc']).mean() * 100
            st.metric(
                "Income Percentile", 
                f"{income_percentile:.0f}%",
                help="Percentage of areas with lower or equal median income"
            )
            
            # House age analysis
            age_percentile = (self.sample_data['HouseAge'] <= input_row['HouseAge']).mean() * 100
            st.metric(
                "House Age Percentile", 
                f"{age_percentile:.0f}%",
                help="Percentage of areas with newer or equal house age"
            )
        
        with col2:
            # Rooms analysis
            rooms_percentile = (self.sample_data['AveRooms'] <= input_row['AveRooms']).mean() * 100
            st.metric(
                "Rooms Percentile", 
                f"{rooms_percentile:.0f}%",
                help="Percentage of areas with fewer or equal average rooms"
            )
            
            # Population density
            pop_percentile = (self.sample_data['Population'] <= input_row['Population']).mean() * 100
            st.metric(
                "Population Percentile", 
                f"{pop_percentile:.0f}%",
                help="Percentage of areas with lower or equal population"
            )
    
    def show_data_exploration_page(self):
        """Display data exploration page."""
        st.header("Data Exploration")
        
        if self.sample_data is None:
            st.error("No data available for exploration.")
            return
        
        # Dataset overview
        st.subheader("Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(self.sample_data):,}")
        with col2:
            st.metric("Features", len(self.sample_data.columns))
        with col3:
            st.metric("Missing Values", self.sample_data.isnull().sum().sum())
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        selected_feature = st.selectbox(
            "Select feature to explore:",
            self.sample_data.columns
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(self.sample_data, x=selected_feature, nbins=50,
                             title=f'Distribution of {selected_feature}')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(self.sample_data, y=selected_feature,
                        title=f'Box Plot of {selected_feature}')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        
        corr_matrix = self.sample_data.corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic visualization
        st.subheader("Geographic Distribution")
        
        if 'Latitude' in self.sample_data.columns and 'Longitude' in self.sample_data.columns:
            # Sample data for performance
            sample_size = min(1000, len(self.sample_data))
            sample_data = self.sample_data.sample(n=sample_size, random_state=42)
            sample_target = self.target_data.loc[sample_data.index]
            
            fig = px.scatter_mapbox(
                sample_data, 
                lat="Latitude", 
                lon="Longitude",
                color=sample_target,
                size=sample_target,
                hover_data=['MedInc', 'HouseAge', 'AveRooms'],
                color_continuous_scale="Viridis",
                size_max=15,
                zoom=5,
                mapbox_style="open-street-map",
                title="House Prices by Geographic Location"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_model_performance_page(self):
        """Display model performance page."""
        st.header("Model Performance")
        
        # Check if evaluation results exist
        if os.path.exists('models/artifacts'):
            artifact_files = os.listdir('models/artifacts')
            eval_files = [f for f in artifact_files if 'evaluation_results.json' in f]
            
            if eval_files:
                st.success("Model evaluation results found!")
                
                # Load and display evaluation results
                for eval_file in eval_files:
                    try:
                        with open(os.path.join('models/artifacts', eval_file), 'r') as f:
                            results = json.load(f)
                        
                        self.display_model_performance(results)
                    except Exception as e:
                        st.error(f"Error loading {eval_file}: {str(e)}")
            else:
                st.warning("No model evaluation results found.")
        else:
            st.warning("No model artifacts directory found.")
        
        # Performance comparison (if multiple models)
        if self.models_loaded:
            st.subheader("Model Comparison")
            st.info("""
            Model comparison would show here if evaluation results were available.
            This would include metrics like RMSE, MAE, R² score, and training time.
            """)
    
    def display_model_performance(self, results):
        """Display performance results for a single model."""
        model_name = results.get('model_name', 'Unknown Model')
        
        st.subheader(f"{model_name} Performance")
        
        # Key metrics
        test_metrics = results.get('metrics', {}).get('test', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RMSE", f"{test_metrics.get('rmse', 0):.4f}")
        with col2:
            st.metric("MAE", f"{test_metrics.get('mae', 0):.4f}")
        with col3:
            st.metric("R² Score", f"{test_metrics.get('r2', 0):.4f}")
        with col4:
            st.metric("MAPE", f"{test_metrics.get('mape', 0):.2f}%")
        
        # Feature importance (if available)
        feature_importance = results.get('feature_importance')
        if feature_importance and feature_importance.get('scores'):
            st.subheader("Feature Importance")
            
            importance_data = feature_importance['scores']
            top_features = dict(list(importance_data.items())[:10])
            
            fig = px.bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                title="Top 10 Most Important Features"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_about_page(self):
        """Display about page."""
        st.header("About This Project")
        
        st.markdown("""
        ## California House Price Prediction
        
        This is a comprehensive machine learning project that demonstrates an end-to-end ML pipeline
        for predicting house prices in California using the famous California housing dataset.
        
        ### Project Goals
        - Demonstrate complete ML workflow from data ingestion to deployment
        - Compare multiple machine learning algorithms
        - Provide interactive predictions through a web interface
        - Show best practices in ML engineering
        
        ### Technologies Used
        - **Python** - Primary programming language
        - **Scikit-learn** - Machine learning library
        - **XGBoost** - Gradient boosting framework
        - **Pandas & NumPy** - Data manipulation
        - **Streamlit** - Web application framework
        - **Plotly** - Interactive visualizations
        - **Optuna** - Hyperparameter optimization
        
        ### Machine Learning Pipeline
        
        1. **Data Ingestion** - Load California housing dataset
        2. **Data Preprocessing** - Clean, scale, and prepare data
        3. **Feature Engineering** - Create new meaningful features
        4. **Model Training** - Train multiple ML algorithms
        5. **Hyperparameter Tuning** - Optimize model performance
        6. **Model Evaluation** - Comprehensive performance analysis
        7. **Web Deployment** - Interactive Streamlit application
        
        ### Project Structure
        ```
        End-to-END-ML-Pipeline/
        ├── data/                   # Data storage
        ├── src/                    # Source code
        │   ├── data/              # Data processing modules
        │   ├── features/          # Feature engineering
        │   ├── models/            # ML models and evaluation
        │   └── visualization/     # EDA and plotting
        ├── app/                   # Streamlit application
        ├── models/                # Trained models and artifacts
        ├── notebooks/             # Jupyter notebooks
        ├── tests/                 # Unit tests
        └── requirements.txt       # Dependencies
        ```
        
        ### Available Models
        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - Elastic Net
        - Random Forest
        - Gradient Boosting
        - XGBoost
        - Support Vector Regression
        - K-Nearest Neighbors
        
        ### Model Evaluation Metrics
        - Root Mean Square Error (RMSE)
        - Mean Absolute Error (MAE)
        - R² Score (Coefficient of Determination)
        - Mean Absolute Percentage Error (MAPE)
        - Residual Analysis
        - Feature Importance
        
        ### Getting Started
        
        To run this project locally:
        
        1. Clone the repository
        2. Install dependencies: `pip install -r requirements.txt`
        3. Run the Streamlit app: `streamlit run app/streamlit_app.py`
        
        ### Note
        This project is designed for educational and demonstration purposes, 
        showcasing best practices in machine learning engineering and deployment.
        """)
        
        # Add footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Built with Streamlit and Python</p>
            <p>© 2024 - End-to-End ML Pipeline Project</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    app = HousePricePredictionApp()
    app.run()


if __name__ == "__main__":
    main()