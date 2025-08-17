# End-to-End ML Pipeline for House Price Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning pipeline for predicting California house prices, demonstrating production-ready ML engineering practices from data ingestion to web deployment.

## Project Overview

This project implements a complete end-to-end machine learning pipeline that:

- **Processes** the California housing dataset with comprehensive data cleaning and preprocessing
- **Engineers** meaningful features using domain knowledge and statistical techniques
- **Trains** and compares multiple ML algorithms with hyperparameter optimization
- **Evaluates** model performance with detailed metrics and visualizations
- **Deploys** an interactive web application for real-time predictions
- **Maintains** production-ready code with logging, error handling, and comprehensive testing

## Architecture

```
End-to-End ML Pipeline
├── Data Layer           │ Ingestion, Cleaning, Preprocessing
├── Feature Layer        │ Engineering, Selection, Transformation
├── Model Layer          │ Training, Tuning, Evaluation
├── Application Layer    │ Web Interface, API, Predictions
└── Infrastructure Layer │ Logging, Testing, Configuration
```

## Project Structure

```
End-to-END-ML-Pipeline/
├── data/                    # Data storage
│   ├── raw/                   # Raw datasets
│   └── processed/             # Processed datasets and artifacts
├── src/                     # Source code
│   ├── data/               # Data processing modules
│   │   ├── data_ingestion.py  # Dataset loading and management
│   │   └── data_preprocessing.py # Cleaning and preprocessing
│   ├── features/           # Feature engineering
│   │   └── feature_engineering.py # Feature creation and selection
│   ├── models/             # Machine learning models
│   │   ├── ml_models.py       # Model training and comparison
│   │   ├── hyperparameter_tuning.py # Optimization strategies
│   │   └── model_evaluation.py # Performance evaluation
│   ├── visualization/      # Data visualization
│   │   └── eda.py            # Exploratory data analysis
│   └── utils/             # Utilities
│       └── logging_config.py # Logging and error handling
├── app/                    # Web application
│   └── streamlit_app.py      # Interactive Streamlit interface
├── models/                 # Trained models and artifacts
│   ├── trained/              # Serialized models
│   └── artifacts/            # Evaluation results and metadata
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit tests
├── logs/                  # Application logs
├── requirements.txt          # Python dependencies
├── run_app.py               # Application launcher
└── README.md                # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/End-to-END-ML-Pipeline.git
   cd End-to-END-ML-Pipeline
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web application**
   ```bash
   python run_app.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run app/streamlit_app.py
   ```

5. **Open your browser** to `http://localhost:8501` to access the application

## Dataset Information

**California Housing Dataset**
- **Source**: Scikit-learn built-in dataset
- **Samples**: 20,640 housing districts
- **Features**: 8 numeric features
- **Target**: Median house value (in hundreds of thousands of dollars)

### Features Description

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group (tens of thousands) |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

## Core Components

### 1. Data Pipeline

**Data Ingestion** (`src/data/data_ingestion.py`)
- Downloads California housing dataset
- Saves data in multiple formats (CSV, Pickle)
- Provides dataset metadata and statistics

**Data Preprocessing** (`src/data/data_preprocessing.py`)
- Handles missing values with multiple strategies
- Detects and manages outliers using IQR and Z-score methods
- Scales features using StandardScaler, RobustScaler, or MinMaxScaler
- Splits data with optional stratification

### 2. Feature Engineering

**Feature Creation** (`src/features/feature_engineering.py`)
- **Domain-specific features**: Room density, bedroom ratio, population density
- **Geographic features**: Distance to major cities, coastal proximity
- **Economic features**: Income categories, per-person income
- **Interaction features**: Cross-products of important variables
- **Polynomial features**: Higher-order terms for non-linear relationships

**Feature Selection**
- **F-test**: Statistical significance-based selection
- **RFE**: Recursive feature elimination with Random Forest
- **Lasso**: L1 regularization-based selection
- **Mutual Information**: Non-linear dependency detection

### 3. Machine Learning Models

**Available Algorithms**:
- **Linear Models**: Linear Regression, Ridge, Lasso, Elastic Net
- **Tree-based**: Random Forest, Gradient Boosting, XGBoost, Decision Tree
- **Instance-based**: K-Nearest Neighbors
- **Kernel Methods**: Support Vector Regression

**Model Training** (`src/models/ml_models.py`)
- Cross-validation with configurable folds
- Comprehensive performance metrics
- Model comparison and selection
- Automated model persistence

### 4. Hyperparameter Optimization

**Optimization Strategies** (`src/models/hyperparameter_tuning.py`)
- **Grid Search**: Exhaustive parameter search
- **Random Search**: Probabilistic parameter sampling
- **Optuna**: Bayesian optimization with Tree-structured Parzen Estimator

### 5. Model Evaluation

**Comprehensive Evaluation** (`src/models/model_evaluation.py`)
- **Regression Metrics**: RMSE, MAE, R², MAPE
- **Residual Analysis**: Normality tests, autocorrelation, heteroscedasticity
- **Feature Importance**: Model-specific importance scores
- **Prediction Intervals**: Uncertainty quantification
- **Visualization**: Performance plots and diagnostic charts

### 6. Web Application

**Interactive Interface** (`app/streamlit_app.py`)
- **Real-time Predictions**: Input house characteristics and get price estimates
- **Data Exploration**: Interactive visualizations and statistics
- **Model Performance**: Comparison charts and evaluation metrics
- **Geographic Visualization**: Interactive maps with house price distributions

## Usage Examples

### Training Models

```python
from src.data.data_ingestion import DataIngestion
from src.data.data_preprocessing import DataPreprocessor
from src.features.feature_engineering import FeaturePipeline
from src.models.ml_models import ModelTrainer

# Load and preprocess data
data_ingestion = DataIngestion()
X, y = data_ingestion.load_data()

preprocessor = DataPreprocessor()
results = preprocessor.preprocess_pipeline(X, y)

# Feature engineering
feature_pipeline = FeaturePipeline()
X_train_featured = feature_pipeline.fit_transform(
    results['X_train_unscaled'], 
    results['y_train']
)

# Train models
trainer = ModelTrainer()
training_results = trainer.train_all_models(
    X_train_featured, 
    results['y_train'],
    feature_pipeline.transform(results['X_test_unscaled']),
    results['y_test']
)

# Get best model
best_name, best_model = trainer.get_best_model()
print(f"Best model: {best_name}")
```

### Hyperparameter Tuning

```python
from src.models.hyperparameter_tuning import HyperparameterTuner

# Initialize tuner
tuner = HyperparameterTuner()

# Optuna optimization
optuna_results = tuner.optuna_tuning(
    'random_forest', 
    X_train_featured, 
    results['y_train'],
    n_trials=100
)

print(f"Best parameters: {optuna_results['best_params']}")
print(f"Best RMSE: {optuna_results['best_score']:.4f}")
```

### Model Evaluation

```python
from src.models.model_evaluation import ModelEvaluator

# Evaluate model
evaluator = ModelEvaluator()
evaluation = evaluator.evaluate_model(
    best_model, best_name,
    X_train_featured, results['y_train'],
    X_test_featured, results['y_test']
)

# Generate comprehensive report
report = evaluator.generate_model_report(best_name)
print(report)

# Create evaluation plots
evaluator.create_evaluation_plots(best_name)
```

## Testing

The project includes comprehensive unit tests covering all major components.

### Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python tests/run_tests.py --module test_data_ingestion

# Run with coverage report
python tests/run_tests.py --coverage
```

### Test Coverage

- **Data Processing**: Data ingestion, preprocessing, and validation
- **Feature Engineering**: Feature creation, selection, and pipeline
- **Model Training**: Algorithm training, evaluation, and persistence
- **Error Handling**: Exception handling and logging functionality

## Performance Metrics

### Model Comparison Example

| Model | CV RMSE | Test RMSE | Test R² | Test MAE | Training Time |
|-------|---------|-----------|---------|----------|---------------|
| XGBoost | 0.524 | 0.531 | 0.782 | 0.389 | 2.34s |
| Random Forest | 0.539 | 0.545 | 0.771 | 0.401 | 1.87s |
| Gradient Boosting | 0.547 | 0.552 | 0.765 | 0.408 | 3.12s |
| Ridge Regression | 0.573 | 0.578 | 0.743 | 0.421 | 0.03s |
| Linear Regression | 0.574 | 0.579 | 0.742 | 0.422 | 0.02s |

## Monitoring and Logging

### Logging System

The project implements comprehensive logging with:

- **Structured Logging**: JSON-formatted logs with contextual information
- **Multiple Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Module-specific Logs**: Separate log files for different components
- **Rotating Logs**: Automatic log rotation to manage disk space

### Error Handling

- **Custom Exceptions**: Domain-specific error types
- **Graceful Degradation**: Fallback mechanisms for non-critical errors
- **Context Preservation**: Detailed error context for debugging

## Web Application Features

### Home Page
- Project overview and dataset information
- Quick statistics and performance summaries
- Feature descriptions and data quality metrics

### Prediction Interface
- Interactive input forms for house characteristics
- Real-time price predictions from multiple models
- Feature importance and prediction confidence intervals
- Input validation and feature insights

### Data Exploration
- Interactive visualizations of feature distributions
- Correlation matrices and statistical summaries
- Geographic visualizations with price distributions
- Customizable data filtering and analysis

### Model Performance Dashboard
- Model comparison charts and metrics
- Performance trends and evaluation plots
- Feature importance rankings
- Residual analysis and diagnostic plots

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Logging configuration
LOG_LEVEL=INFO
LOG_DIR=logs

# Model configuration
MODEL_RANDOM_STATE=42
CV_FOLDS=5

# Application configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
```

### Custom Model Configuration

```python
# Custom model parameters
models_config = {
    'random_forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5
        }
    }
}

trainer = ModelTrainer(models_config=models_config)
```

## Deployment Options

### Local Deployment
```bash
# Standard deployment
streamlit run app/streamlit_app.py --server.port 8501

# With custom configuration
streamlit run app/streamlit_app.py --server.port 8080 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0"]
```

### Cloud Deployment
- **Streamlit Cloud**: Direct deployment from GitHub repository
- **Heroku**: Using Procfile and requirements.txt
- **AWS/GCP/Azure**: Container-based deployment with Docker

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Run the test suite: `python tests/run_tests.py`
5. Submit a pull request

### Code Style

- **Formatting**: Black code formatter
- **Linting**: Flake8 for code quality
- **Type Hints**: Use type annotations where appropriate
- **Docstrings**: Google-style docstrings for all functions and classes

## Future Enhancements

### Planned Features
- [ ] **MLOps Integration**: MLflow for experiment tracking
- [ ] **Model Monitoring**: Drift detection and performance monitoring
- [ ] **API Deployment**: REST API for model serving
- [ ] **Advanced Visualizations**: Interactive dashboards with Plotly Dash
- [ ] **Model Interpretability**: SHAP values and LIME explanations
- [ ] **Automated Retraining**: Scheduled model updates
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Data Validation**: Great Expectations integration

### Potential Improvements
- **Feature Store**: Centralized feature management
- **Model Versioning**: Automated model lifecycle management
- **Distributed Training**: Support for large-scale datasets
- **Real-time Predictions**: Streaming prediction pipeline
- **Multi-model Ensemble**: Advanced ensemble techniques

## Learning Resources

### Machine Learning
- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)

### MLOps and Production ML
- [Building Machine Learning Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/)
- [Machine Learning Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115777/)
- [MLOps: Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html)

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the project root and virtual environment is activated
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Memory Issues with Large Datasets**
```python
# Use data sampling for development
X_sample = X.sample(n=1000, random_state=42)
```

**Streamlit Port Conflicts**
```bash
# Use a different port
streamlit run app/streamlit_app.py --server.port 8502
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Scikit-learn** team for the California housing dataset
- **Streamlit** team for the amazing web framework
- **XGBoost** developers for the gradient boosting implementation
- **Optuna** team for the hyperparameter optimization library
- **Open Source Community** for the fantastic ML ecosystem

## Contact

- **GitHub Issues**: [Create an issue](https://github.com/your-username/End-to-END-ML-Pipeline/issues)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

<div align="center">

**If you find this project helpful, please consider giving it a star!**

Built with Python, Scikit-learn, XGBoost, and Streamlit

</div>