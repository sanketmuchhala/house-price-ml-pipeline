"""
Comprehensive logging configuration for the ML pipeline.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import sys


class MLPipelineLogger:
    """
    Centralized logging configuration for the ML pipeline.
    """
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        """
        Initialize the logging configuration.
        
        Args:
            log_dir (str): Directory to store log files
            log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.setup_logging()
    
    def setup_logging(self):
        """Set up comprehensive logging configuration."""
        # Create logs directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Remove default handlers to avoid duplication
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler - general log
        general_log_file = os.path.join(self.log_dir, 'ml_pipeline.log')
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Error handler - errors only
        error_log_file = os.path.join(self.log_dir, 'errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Create module-specific loggers
        self.setup_module_loggers()
        
        # Log the setup completion
        logger = logging.getLogger(__name__)
        logger.info(f"Logging system initialized. Log directory: {self.log_dir}")
    
    def setup_module_loggers(self):
        """Set up specific loggers for different modules."""
        module_configs = {
            'src.data': {'level': logging.INFO, 'file': 'data_processing.log'},
            'src.features': {'level': logging.INFO, 'file': 'feature_engineering.log'},
            'src.models': {'level': logging.INFO, 'file': 'model_training.log'},
            'src.visualization': {'level': logging.INFO, 'file': 'visualization.log'},
            'app': {'level': logging.INFO, 'file': 'streamlit_app.log'}
        }
        
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        for module_name, config in module_configs.items():
            logger = logging.getLogger(module_name)
            logger.setLevel(config['level'])
            
            # Create file handler for this module
            log_file = os.path.join(self.log_dir, config['file'])
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            handler.setFormatter(detailed_formatter)
            logger.addHandler(handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name (str): Logger name
            
        Returns:
            logging.Logger: Logger instance
        """
        return logging.getLogger(name)


class MLPipelineError(Exception):
    """Base exception class for ML pipeline errors."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        """
        Initialize ML pipeline error.
        
        Args:
            message (str): Error message
            error_code (str): Error code for categorization
            context (Dict): Additional context information
        """
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp
        }


class DataIngestionError(MLPipelineError):
    """Error in data ingestion process."""
    pass


class DataPreprocessingError(MLPipelineError):
    """Error in data preprocessing."""
    pass


class FeatureEngineeringError(MLPipelineError):
    """Error in feature engineering."""
    pass


class ModelTrainingError(MLPipelineError):
    """Error in model training."""
    pass


class ModelEvaluationError(MLPipelineError):
    """Error in model evaluation."""
    pass


class PredictionError(MLPipelineError):
    """Error in making predictions."""
    pass


def log_function_call(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger (logging.Logger): Logger instance (uses function's module logger if None)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get logger
            func_logger = logger or logging.getLogger(func.__module__)
            
            # Log function start
            func_logger.debug(f"Starting {func.__name__} with args={args}, kwargs={kwargs}")
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                func_logger.debug(f"Completed {func.__name__} in {execution_time:.4f} seconds")
                return result
                
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                func_logger.error(f"Error in {func.__name__} after {execution_time:.4f} seconds: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_error(error: Exception, context: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
    """
    Log an error with context information.
    
    Args:
        error (Exception): The error to log
        context (Dict): Additional context information
        logger (logging.Logger): Logger instance (uses root logger if None)
    """
    error_logger = logger or logging.getLogger()
    
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # If it's an MLPipelineError, use its structured format
    if isinstance(error, MLPipelineError):
        error_info.update(error.to_dict())
    
    error_logger.error(f"Pipeline Error: {json.dumps(error_info, indent=2)}")


def setup_pipeline_logging(log_dir: str = "logs", log_level: str = "INFO") -> MLPipelineLogger:
    """
    Set up logging for the entire ML pipeline.
    
    Args:
        log_dir (str): Directory for log files
        log_level (str): Logging level
        
    Returns:
        MLPipelineLogger: Configured logger instance
    """
    return MLPipelineLogger(log_dir=log_dir, log_level=log_level)


class ErrorHandler:
    """
    Centralized error handling for the ML pipeline.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.
        
        Args:
            logger (logging.Logger): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
    
    def handle_error(self, error: Exception, operation: str, 
                    context: Dict[str, Any] = None, raise_error: bool = True):
        """
        Handle an error with logging and optional re-raising.
        
        Args:
            error (Exception): The error to handle
            operation (str): Operation where error occurred
            context (Dict): Additional context
            raise_error (bool): Whether to re-raise the error
        """
        # Count error types
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Prepare context
        full_context = {
            'operation': operation,
            'error_count': self.error_counts[error_type],
            **(context or {})
        }
        
        # Log error
        log_error(error, full_context, self.logger)
        
        # Re-raise if requested
        if raise_error:
            if isinstance(error, MLPipelineError):
                raise error
            else:
                # Wrap in appropriate pipeline error
                if 'data' in operation.lower():
                    if 'ingest' in operation.lower():
                        raise DataIngestionError(str(error), "DATA_INGESTION_FAILED", full_context)
                    else:
                        raise DataPreprocessingError(str(error), "DATA_PREPROCESSING_FAILED", full_context)
                elif 'feature' in operation.lower():
                    raise FeatureEngineeringError(str(error), "FEATURE_ENGINEERING_FAILED", full_context)
                elif 'model' in operation.lower() or 'train' in operation.lower():
                    raise ModelTrainingError(str(error), "MODEL_TRAINING_FAILED", full_context)
                elif 'predict' in operation.lower():
                    raise PredictionError(str(error), "PREDICTION_FAILED", full_context)
                else:
                    raise MLPipelineError(str(error), "GENERAL_PIPELINE_ERROR", full_context)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts."""
        return self.error_counts.copy()


def main():
    """
    Main function to demonstrate logging setup.
    """
    # Setup logging
    ml_logger = setup_pipeline_logging()
    
    # Get logger
    logger = ml_logger.get_logger(__name__)
    
    logger.info("Logging system demonstration")
    logger.warning("This is a warning message")
    
    # Test error handling
    error_handler = ErrorHandler(logger)
    
    try:
        # Simulate an error
        raise ValueError("This is a test error")
    except Exception as e:
        error_handler.handle_error(e, "test_operation", {"test": True}, raise_error=False)
    
    print("Logging demonstration completed. Check the logs directory for output files.")


if __name__ == "__main__":
    main()