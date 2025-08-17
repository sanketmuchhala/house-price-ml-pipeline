"""
Exploratory Data Analysis (EDA) module with comprehensive visualizations.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class EDAVisualizer:
    """
    Comprehensive Exploratory Data Analysis and Visualization class.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize EDAVisualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_data_overview(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data overview.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Data overview statistics
        """
        overview = {
            'basic_info': {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_column': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
            },
            'duplicates': {
                'total_duplicates': df.duplicated().sum(),
                'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
            },
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }
        
        # Categorical summary
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            overview['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency': df[col].value_counts().head().to_dict()
            }
        
        self.logger.info("Data overview generated successfully")
        return overview
    
    def plot_distribution_analysis(self, df: pd.DataFrame, target_col: str = None) -> None:
        """
        Create distribution plots for all numeric variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name for correlation analysis
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if target_col and target_col in numeric_cols:
            numeric_cols = numeric_cols.drop(target_col)
        
        n_cols = len(numeric_cols)
        if n_cols == 0:
            self.logger.warning("No numeric columns found for distribution analysis")
            return
        
        # Calculate subplot layout
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                # Histogram with KDE
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/processed/distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Distribution analysis plots created")
    
    def plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """
        Create correlation matrix heatmap.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            self.logger.warning("Insufficient numeric columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('data/processed/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Correlation matrix created")
    
    def plot_target_analysis(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Analyze target variable distribution and relationships.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
        """
        if target_col not in df.columns:
            self.logger.error(f"Target column '{target_col}' not found in dataframe")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Target distribution
        sns.histplot(data=df, x=target_col, kde=True, ax=axes[0,0])
        axes[0,0].set_title(f'Distribution of {target_col}')
        axes[0,0].grid(True, alpha=0.3)
        
        # Box plot
        sns.boxplot(y=df[target_col], ax=axes[0,1])
        axes[0,1].set_title(f'Box Plot of {target_col}')
        axes[0,1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(df[target_col], dist="norm", plot=axes[1,0])
        axes[1,0].set_title(f'Q-Q Plot of {target_col}')
        axes[1,0].grid(True, alpha=0.3)
        
        # Target vs strongest correlated feature
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_with_target = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
            strongest_feature = corr_with_target.index[1]  # Exclude target itself
            
            sns.scatterplot(data=df, x=strongest_feature, y=target_col, alpha=0.6, ax=axes[1,1])
            axes[1,1].set_title(f'{target_col} vs {strongest_feature}')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/processed/target_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Target analysis plots created")
    
    def plot_feature_relationships(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Plot relationships between features and target variable.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if len(feature_cols) == 0:
            self.logger.warning("No numeric features found for relationship analysis")
            return
        
        # Calculate correlations with target
        correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        correlations = correlations.sort_values(key=abs, ascending=False)
        
        # Plot top correlations
        top_features = correlations.head(6).index
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                sns.scatterplot(data=df, x=feature, y=target_col, alpha=0.6, ax=axes[i])
                
                # Add correlation coefficient to title
                corr_coef = correlations[feature]
                axes[i].set_title(f'{feature} vs {target_col}\nCorrelation: {corr_coef:.3f}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/processed/feature_relationships.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Feature relationship plots created")
    
    def plot_outlier_analysis(self, df: pd.DataFrame) -> None:
        """
        Create outlier analysis plots.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns found for outlier analysis")
            return
        
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 6*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                sns.boxplot(y=df[col], ax=axes[i])
                axes[i].set_title(f'Outliers in {col}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/processed/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info("Outlier analysis plots created")
    
    def create_interactive_scatter_matrix(self, df: pd.DataFrame, target_col: str = None) -> None:
        """
        Create interactive scatter matrix using Plotly.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column for color coding
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            self.logger.warning("Insufficient numeric columns for scatter matrix")
            return
        
        # Limit to reasonable number of features for visualization
        if len(numeric_cols) > 8:
            # Select top correlated features with target if available
            if target_col and target_col in numeric_cols:
                correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
                selected_cols = correlations.head(8).index.tolist()
            else:
                selected_cols = numeric_cols[:8].tolist()
        else:
            selected_cols = numeric_cols.tolist()
        
        # Create scatter matrix
        if target_col and target_col in df.columns:
            fig = px.scatter_matrix(df[selected_cols], 
                                  color=df[target_col] if target_col in selected_cols else None,
                                  title="Interactive Scatter Matrix")
        else:
            fig = px.scatter_matrix(df[selected_cols], title="Interactive Scatter Matrix")
        
        fig.update_layout(height=800, width=800)
        fig.write_html('data/processed/interactive_scatter_matrix.html')
        fig.show()
        
        self.logger.info("Interactive scatter matrix created")
    
    def create_geographic_analysis(self, df: pd.DataFrame, lat_col: str = 'Latitude', 
                                 lon_col: str = 'Longitude', target_col: str = None) -> None:
        """
        Create geographic analysis if latitude/longitude data is available.
        
        Args:
            df (pd.DataFrame): Input dataframe
            lat_col (str): Latitude column name
            lon_col (str): Longitude column name
            target_col (str): Target column for color coding
        """
        if lat_col not in df.columns or lon_col not in df.columns:
            self.logger.info("Geographic columns not found, skipping geographic analysis")
            return
        
        # Create geographic scatter plot
        fig = plt.figure(figsize=(15, 10))
        
        if target_col and target_col in df.columns:
            scatter = plt.scatter(df[lon_col], df[lat_col], c=df[target_col], 
                                alpha=0.6, s=30, cmap='viridis')
            plt.colorbar(scatter, label=target_col)
            plt.title(f'Geographic Distribution of {target_col}')
        else:
            plt.scatter(df[lon_col], df[lat_col], alpha=0.6, s=30)
            plt.title('Geographic Distribution of Data Points')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('data/processed/geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create interactive map using Plotly
        if target_col and target_col in df.columns:
            fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, color=target_col,
                                  size_max=15, zoom=5, mapbox_style="open-street-map",
                                  title=f"Interactive Map: {target_col}")
        else:
            fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col,
                                  size_max=15, zoom=5, mapbox_style="open-street-map",
                                  title="Interactive Map: Data Distribution")
        
        fig.write_html('data/processed/interactive_map.html')
        fig.show()
        
        self.logger.info("Geographic analysis plots created")
    
    def generate_comprehensive_report(self, df: pd.DataFrame, target_col: str = None) -> Dict:
        """
        Generate comprehensive EDA report with all visualizations.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            Dict: Comprehensive analysis report
        """
        self.logger.info("Starting comprehensive EDA analysis...")
        
        # Ensure processed directory exists
        import os
        os.makedirs('data/processed', exist_ok=True)
        
        # Generate data overview
        overview = self.generate_data_overview(df)
        
        # Create all visualizations
        self.plot_distribution_analysis(df, target_col)
        self.plot_correlation_matrix(df)
        
        if target_col:
            self.plot_target_analysis(df, target_col)
            self.plot_feature_relationships(df, target_col)
        
        self.plot_outlier_analysis(df)
        self.create_interactive_scatter_matrix(df, target_col)
        self.create_geographic_analysis(df, target_col=target_col)
        
        # Generate summary insights
        insights = self._generate_insights(df, target_col, overview)
        
        report = {
            'overview': overview,
            'insights': insights,
            'visualizations_created': [
                'distribution_analysis.png',
                'correlation_matrix.png',
                'target_analysis.png' if target_col else None,
                'feature_relationships.png' if target_col else None,
                'outlier_analysis.png',
                'interactive_scatter_matrix.html',
                'geographic_analysis.png',
                'interactive_map.html'
            ]
        }
        
        # Remove None values from visualizations list
        report['visualizations_created'] = [v for v in report['visualizations_created'] if v is not None]
        
        self.logger.info("Comprehensive EDA analysis completed")
        return report
    
    def _generate_insights(self, df: pd.DataFrame, target_col: str, overview: Dict) -> List[str]:
        """
        Generate actionable insights from the data analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Target column name
            overview (Dict): Data overview statistics
            
        Returns:
            List[str]: List of insights
        """
        insights = []
        
        # Data quality insights
        if overview['missing_data']['total_missing'] > 0:
            insights.append(f"Dataset has {overview['missing_data']['total_missing']} missing values that need attention")
        else:
            insights.append("Dataset has no missing values - excellent data quality")
        
        # Correlation insights
        if target_col and target_col in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            if len(feature_cols) > 0:
                correlations = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
                strongest_corr = correlations.abs().max()
                strongest_feature = correlations.abs().idxmax()
                
                insights.append(f"Strongest predictor is '{strongest_feature}' with correlation of {correlations[strongest_feature]:.3f}")
                
                high_corr_features = correlations[correlations.abs() > 0.7]
                if len(high_corr_features) > 0:
                    insights.append(f"Features with high correlation (>0.7): {list(high_corr_features.index)}")
        
        # Outlier insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            outlier_counts[col] = outliers
        
        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            max_outlier_col = max(outlier_counts, key=outlier_counts.get)
            insights.append(f"Total outliers detected: {total_outliers}. '{max_outlier_col}' has the most outliers ({outlier_counts[max_outlier_col]})")
        
        return insights


def main():
    """
    Main function to demonstrate EDA functionality.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Import required modules
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from data.data_ingestion import DataIngestion
    
    # Load data
    data_ingestion = DataIngestion()
    X, y = data_ingestion.load_data()
    
    # Combine for EDA
    df = X.copy()
    df['target'] = y
    
    # Initialize EDA
    eda = EDAVisualizer()
    
    # Generate comprehensive report
    report = eda.generate_comprehensive_report(df, target_col='target')
    
    print("EDA Report Generated Successfully!")
    print(f"Overview: {len(report['insights'])} insights generated")
    print(f"Visualizations: {len(report['visualizations_created'])} files created")


if __name__ == "__main__":
    main()