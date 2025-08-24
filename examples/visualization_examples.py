"""
Examples demonstrating visualization capabilities
"""
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from extended_sklearn_metrics import (
    evaluate_model_with_cross_validation,
    evaluate_classification_model_with_cross_validation,
    create_performance_summary_plot,
    create_model_comparison_plot,
    create_performance_radar_chart,
    print_performance_report
)
import warnings

# Suppress matplotlib warning if not available
warnings.filterwarnings("ignore", message="Matplotlib is required")


def example_1_single_model_visualization():
    """Example 1: Visualize single model performance"""
    print("=" * 60)
    print("EXAMPLE 1: Single Model Visualization")
    print("=" * 60)
    
    # Load data
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Evaluate
    result = evaluate_model_with_cross_validation(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=5
    )
    
    # Print formatted report
    print_performance_report(result)
    
    # Create visualizations (will show plots if matplotlib available)
    try:
        print("\nGenerating performance summary plot...")
        create_performance_summary_plot(result, title="Linear Regression Performance")
        
        print("Generating radar chart...")
        create_performance_radar_chart(result, title="Linear Regression Radar")
        
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("✅ Single model visualization example completed!")


def example_2_model_comparison_visualization():
    """Example 2: Compare multiple models with visualization"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Model Comparison Visualization")
    print("=" * 60)
    
    # Load data
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    # Define models
    models = {
        'Linear Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR())
        ])
    }
    
    # Evaluate all models
    results_dict = {}
    
    print("Evaluating models...")
    for name, model in models.items():
        print(f"  - {name}")
        result = evaluate_model_with_cross_validation(
            model=model,
            X=X_train,
            y=y_train,
            cv=3  # Reduced for faster execution
        )
        results_dict[name] = result
    
    # Print comparison summary
    print(f"\nMODEL COMPARISON SUMMARY:")
    print("-" * 30)
    
    for model_name, result in results_dict.items():
        r2_score = result.loc[result['Metric'] == 'R²', 'Value'].iloc[0]
        rmse_perf = result.loc[result['Metric'] == 'RMSE', 'Performance'].iloc[0]
        print(f"{model_name:18}: R² = {r2_score:.4f}, RMSE = {rmse_perf}")
    
    # Create comparison visualization
    try:
        print(f"\nGenerating model comparison plot...")
        create_model_comparison_plot(results_dict)
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("✅ Model comparison visualization example completed!")


def example_3_classification_visualization():
    """Example 3: Classification model visualization"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Classification Model Visualization")
    print("=" * 60)
    
    # Load data
    iris = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Evaluate
    result = evaluate_classification_model_with_cross_validation(
        model=model,
        X=X_train,
        y=y_train,
        cv=5
    )
    
    # Print report
    print_performance_report(result)
    
    # Create visualizations
    try:
        print(f"\nGenerating classification performance plots...")
        create_performance_summary_plot(
            result, 
            title="Random Forest Classifier - Iris Dataset"
        )
        
        create_performance_radar_chart(
            result,
            title="Classification Performance Radar"
        )
        
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("✅ Classification visualization example completed!")


def example_4_advanced_reporting():
    """Example 4: Advanced performance reporting"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Advanced Performance Reporting")
    print("=" * 60)
    
    # Load data
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    # Test different models with different performance levels
    models_and_configs = [
        ("Basic Linear", LinearRegression()),
        ("Scaled Linear", Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])),
        ("Random Forest", RandomForestRegressor(n_estimators=10, random_state=42))
    ]
    
    print("COMPARATIVE PERFORMANCE ANALYSIS:")
    print("=" * 40)
    
    for model_name, model in models_and_configs:
        print(f"\n{model_name.upper()}:")
        print("-" * len(model_name))
        
        result = evaluate_model_with_cross_validation(
            model=model,
            X=X_train,
            y=y_train,
            cv=3
        )
        
        # Show just the key metrics in a compact format
        for _, row in result.iterrows():
            metric = row['Metric']
            value = row['Value'] 
            performance = row['Performance']
            
            # Emoji indicators
            perf_emoji = {
                'Excellent': '🟢', 'Good': '🟢', 'Acceptable': '🟡',
                'Moderate': '🟡', 'Poor': '🔴', 'Very Poor': '🔴'
            }.get(performance, '⚪')
            
            print(f"  {perf_emoji} {metric:15}: {value:8.4f} ({performance})")
    
    print(f"\n{'=' * 60}")
    print("VISUALIZATION FEATURES AVAILABLE:")
    print("- create_performance_summary_plot(): Bar charts of metrics and performance")
    print("- create_model_comparison_plot(): Compare multiple models side-by-side") 
    print("- create_performance_radar_chart(): Radar/spider chart visualization")
    print("- print_performance_report(): Formatted console reports with recommendations")
    print("=" * 60)
    
    print("✅ Advanced reporting example completed!")


if __name__ == "__main__":
    print("Extended Sklearn Metrics - Visualization Examples\n")
    
    example_1_single_model_visualization()
    example_2_model_comparison_visualization()
    example_3_classification_visualization()
    example_4_advanced_reporting()
    
    print("\n" + "=" * 60)
    print("All visualization examples completed successfully!")
    print("💡 Install matplotlib to see interactive plots: pip install matplotlib")
    print("=" * 60)