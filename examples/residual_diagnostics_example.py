"""
Example demonstrating residual diagnostics functionality for regression models.

This example shows how to use the residual diagnostics features to analyze
regression model performance and check assumptions.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from extended_sklearn_metrics import (
    calculate_residual_diagnostics,
    create_residual_summary_report,
    print_residual_diagnostics_report,
    create_residual_plots,
    create_residual_summary_plot
)


def example_synthetic_data():
    """Example with synthetic regression data"""
    print("=" * 70)
    print("EXAMPLE 1: SYNTHETIC REGRESSION DATA")
    print("=" * 70)
    
    # Create synthetic regression dataset
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=15,
        random_state=42
    )
    
    # Convert to DataFrame/Series for better display
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_series = pd.Series(y)
    
    # Create and fit model
    model = LinearRegression()
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Target range: [{y_series.min():.2f}, {y_series.max():.2f}]")
    print()
    
    # Calculate residual diagnostics
    diagnostics = calculate_residual_diagnostics(
        model=model,
        X=X_df,
        y=y_series,
        cv=5
    )
    
    # Print comprehensive report
    print_residual_diagnostics_report(diagnostics)
    print()
    
    # Show summary DataFrame
    summary_df = create_residual_summary_report(diagnostics)
    print("SUMMARY TABLE:")
    print("-" * 50)
    print(summary_df.to_string(index=False))
    print()


def example_diabetes_data():
    """Example with real-world diabetes dataset"""
    print("=" * 70)
    print("EXAMPLE 2: DIABETES DATASET (REAL DATA)")
    print("=" * 70)
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Convert to DataFrame for better handling
    X_df = pd.DataFrame(X, columns=diabetes.feature_names)
    y_series = pd.Series(y)
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Features: {list(X_df.columns)}")
    print(f"Target (diabetes progression): [{y_series.min():.1f}, {y_series.max():.1f}]")
    print()
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Calculate residual diagnostics
    diagnostics = calculate_residual_diagnostics(
        model=pipeline,
        X=X_df,
        y=y_series,
        cv=5
    )
    
    # Print report
    print_residual_diagnostics_report(diagnostics)
    print()


def example_poor_model():
    """Example demonstrating diagnostics on a poorly-specified model"""
    print("=" * 70)
    print("EXAMPLE 3: POORLY SPECIFIED MODEL (NON-LINEAR RELATIONSHIP)")
    print("=" * 70)
    
    # Create non-linear data but fit linear model
    np.random.seed(42)
    X = np.random.uniform(-3, 3, 150).reshape(-1, 1)
    y = X.ravel() ** 2 + 2 * X.ravel() + np.random.normal(0, 2, 150)  # Quadratic relationship
    
    X_df = pd.DataFrame(X, columns=['feature'])
    y_series = pd.Series(y)
    
    print(f"Data: Quadratic relationship y = x² + 2x + noise")
    print(f"Model: Linear regression (mis-specification)")
    print(f"Samples: {len(y_series)}")
    print()
    
    # Fit linear model to quadratic data (poor fit)
    model = LinearRegression()
    
    # Calculate diagnostics
    diagnostics = calculate_residual_diagnostics(
        model=model,
        X=X_df,
        y=y_series,
        cv=5
    )
    
    # Print report - should show issues
    print_residual_diagnostics_report(diagnostics)
    print()
    
    # Show specific statistics
    stats = diagnostics['residual_statistics']
    print("KEY STATISTICS:")
    print(f"  Residual mean: {stats['mean']:.6f} (should be ~0)")
    print(f"  Residual std:  {stats['std']:.4f}")
    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    
    # Show outlier information
    if 'outlier_analysis' in diagnostics:
        outliers = diagnostics['outlier_analysis']
        total = diagnostics['sample_size']
        print(f"  Outliers >2σ: {outliers['outliers_2std']} ({outliers['outliers_2std']/total*100:.1f}%)")
        print(f"  Outliers >3σ: {outliers['outliers_3std']} ({outliers['outliers_3std']/total*100:.1f}%)")
    print()


def example_with_plots():
    """Example showing how to create residual plots"""
    print("=" * 70)
    print("EXAMPLE 4: CREATING RESIDUAL PLOTS")
    print("=" * 70)
    
    # Use simple dataset for plotting
    X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=123)
    X_df = pd.DataFrame(X, columns=['feature_0', 'feature_1', 'feature_2'])
    y_series = pd.Series(y)
    
    model = LinearRegression()
    
    # Calculate diagnostics
    diagnostics = calculate_residual_diagnostics(model, X_df, y_series, cv=5)
    
    print("Generating residual diagnostic plots...")
    print("Note: In a Jupyter notebook, these would display automatically.")
    print()
    
    try:
        # Create residual plots (6-panel diagnostic plot)
        create_residual_plots(diagnostics)
        print("✅ Residual diagnostic plots created successfully")
        
        # Create summary plot
        create_residual_summary_plot(diagnostics)
        print("✅ Residual summary plot created successfully")
        
        print()
        print("Plots include:")
        print("  • Residuals vs Fitted values")
        print("  • Q-Q plot for normality assessment")
        print("  • Scale-Location plot")
        print("  • Residuals vs Leverage")
        print("  • Histogram of residuals")
        print("  • Autocorrelation plot")
        
    except ImportError as e:
        print(f"⚠️  Plotting not available: {e}")
        print("Install matplotlib to enable plotting: pip install matplotlib")
    except Exception as e:
        print(f"⚠️  Error creating plots: {e}")
    
    print()


def main():
    """Run all examples"""
    print("EXTENDED SKLEARN METRICS - RESIDUAL DIAGNOSTICS EXAMPLES")
    print("=" * 70)
    print()
    
    # Run examples
    example_synthetic_data()
    example_diabetes_data()
    example_poor_model()
    example_with_plots()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The residual diagnostics functionality provides:")
    print()
    print("✅ STATISTICAL TESTS:")
    print("   • Normality tests (Shapiro-Wilk, KS, Anderson-Darling)")
    print("   • Heteroscedasticity tests (Breusch-Pagan, Goldfeld-Quandt)")
    print("   • Autocorrelation tests (Durbin-Watson)")
    print("   • Outlier detection and analysis")
    print()
    print("✅ COMPREHENSIVE REPORTING:")
    print("   • Detailed statistical summaries")
    print("   • Automatic issue detection")
    print("   • Actionable recommendations")
    print("   • Export to pandas DataFrame")
    print()
    print("✅ VISUALIZATIONS:")
    print("   • 6-panel diagnostic plots")
    print("   • Q-Q plots for normality")
    print("   • Residual distribution analysis")
    print("   • Interactive summary plots")
    print()
    print("✅ INTEGRATION:")
    print("   • Works with scikit-learn pipelines")
    print("   • Cross-validation based analysis")
    print("   • Handles edge cases gracefully")
    print()
    print("Use residual diagnostics to validate regression assumptions!")
    print("=" * 70)


if __name__ == "__main__":
    main()