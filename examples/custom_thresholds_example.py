"""
Examples demonstrating custom thresholds with extended-sklearn-metrics
"""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from extended_sklearn_metrics import evaluate_model_with_cross_validation, CustomThresholds


def example_1_default_vs_custom_thresholds():
    """Example 1: Compare default vs custom thresholds"""
    print("=" * 70)
    print("EXAMPLE 1: Default vs Custom Thresholds Comparison")
    print("=" * 70)
    
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
    
    print("DEFAULT THRESHOLDS:")
    print("-" * 30)
    result_default = evaluate_model_with_cross_validation(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=5
    )
    print(result_default[['Metric', 'Value', 'Performance']].to_string(index=False))
    
    print("\nCUSTOM THRESHOLDS (More Lenient):")
    print("-" * 40)
    # More lenient thresholds
    lenient_thresholds = CustomThresholds(
        error_thresholds=(20, 35, 50),  # More lenient error thresholds
        score_thresholds=(0.4, 0.6)     # More lenient score thresholds
    )
    
    result_lenient = evaluate_model_with_cross_validation(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        custom_thresholds=lenient_thresholds
    )
    print(result_lenient[['Metric', 'Value', 'Performance']].to_string(index=False))
    
    print("\nCUSTOM THRESHOLDS (More Strict):")
    print("-" * 40)
    # More strict thresholds
    strict_thresholds = CustomThresholds(
        error_thresholds=(5, 10, 15),   # More strict error thresholds
        score_thresholds=(0.7, 0.85)    # More strict score thresholds
    )
    
    result_strict = evaluate_model_with_cross_validation(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=5,
        custom_thresholds=strict_thresholds
    )
    print(result_strict[['Metric', 'Value', 'Performance']].to_string(index=False))


def example_2_domain_specific_thresholds():
    """Example 2: Domain-specific threshold requirements"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Domain-Specific Threshold Requirements")
    print("=" * 70)
    
    # Load data
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    
    # High-precision application (e.g., financial modeling)
    print("HIGH-PRECISION APPLICATION (Financial Modeling):")
    print("-" * 50)
    print("Requirements: Very low error tolerance, high R² requirement")
    
    high_precision_thresholds = CustomThresholds(
        error_thresholds=(2, 5, 8),     # Very strict error requirements
        score_thresholds=(0.85, 0.95)   # Very high R² requirements  
    )
    
    result_precision = evaluate_model_with_cross_validation(
        model=model,
        X=X_train,
        y=y_train,
        cv=5,
        custom_thresholds=high_precision_thresholds
    )
    
    # Show full details for high-precision case
    print(result_precision[['Metric', 'Value', 'Threshold', 'Performance']].to_string(index=False))
    
    # Exploratory data analysis (more lenient)
    print(f"\n{'EXPLORATORY DATA ANALYSIS:'}")
    print("-" * 35)
    print("Requirements: Focus on discovering patterns, less strict accuracy")
    
    exploratory_thresholds = CustomThresholds(
        error_thresholds=(25, 40, 60),  # More lenient for exploration
        score_thresholds=(0.3, 0.5)     # Lower R² requirements
    )
    
    result_exploratory = evaluate_model_with_cross_validation(
        model=model,
        X=X_train,
        y=y_train,
        cv=5,
        custom_thresholds=exploratory_thresholds
    )
    print(result_exploratory[['Metric', 'Value', 'Performance']].to_string(index=False))


def example_3_progressive_threshold_tuning():
    """Example 3: Progressive threshold tuning based on model development stage"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Progressive Threshold Tuning")
    print("=" * 70)
    
    # Load data
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    
    stages = [
        ("Proof of Concept", CustomThresholds(error_thresholds=(30, 50, 70), score_thresholds=(0.2, 0.4))),
        ("Development", CustomThresholds(error_thresholds=(20, 30, 45), score_thresholds=(0.4, 0.6))),
        ("Pre-Production", CustomThresholds(error_thresholds=(10, 20, 30), score_thresholds=(0.6, 0.8))),
        ("Production", CustomThresholds(error_thresholds=(5, 10, 15), score_thresholds=(0.8, 0.9)))
    ]
    
    results_summary = []
    
    for stage_name, thresholds in stages:
        print(f"\n{stage_name.upper()} STAGE:")
        print("-" * len(stage_name))
        
        result = evaluate_model_with_cross_validation(
            model=model,
            X=X_train,
            y=y_train,
            cv=3,  # Faster for demonstration
            custom_thresholds=thresholds
        )
        
        # Extract key metrics
        rmse_perf = result.loc[result['Metric'] == 'RMSE', 'Performance'].iloc[0]
        r2_perf = result.loc[result['Metric'] == 'R²', 'Performance'].iloc[0]
        
        print(f"RMSE Performance: {rmse_perf}")
        print(f"R² Performance: {r2_perf}")
        
        results_summary.append({
            'Stage': stage_name,
            'RMSE_Performance': rmse_perf,
            'R²_Performance': r2_perf,
            'Ready_for_Next_Stage': rmse_perf in ['Excellent', 'Good'] and r2_perf in ['Good', 'Acceptable']
        })
    
    # Summary table
    print(f"\n{'=' * 70}")
    print("STAGE PROGRESSION SUMMARY:")
    print("=" * 70)
    
    import pandas as pd
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # Check if model is ready for production
    production_ready = results_summary[-1]['Ready_for_Next_Stage']
    print(f"\nModel Production Ready: {'YES' if production_ready else 'NO'}")


def example_4_threshold_sensitivity_analysis():
    """Example 4: Understanding threshold sensitivity"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Threshold Sensitivity Analysis")
    print("=" * 70)
    
    # Load data
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    
    # Test different error thresholds while keeping score thresholds constant
    print("ERROR THRESHOLD SENSITIVITY:")
    print("-" * 30)
    
    error_threshold_sets = [
        ("Very Strict", (3, 7, 12)),
        ("Strict", (8, 15, 25)),
        ("Moderate", (15, 25, 40)),
        ("Lenient", (25, 40, 60))
    ]
    
    for name, error_thresholds in error_threshold_sets:
        thresholds = CustomThresholds(
            error_thresholds=error_thresholds,
            score_thresholds=(0.5, 0.7)  # Keep constant
        )
        
        result = evaluate_model_with_cross_validation(
            model=model,
            X=X_train,
            y=y_train,
            cv=3,
            custom_thresholds=thresholds
        )
        
        rmse_perf = result.loc[result['Metric'] == 'RMSE', 'Performance'].iloc[0]
        rmse_value = result.loc[result['Metric'] == 'RMSE', 'Value'].iloc[0]
        
        print(f"{name:12} ({error_thresholds}): RMSE = {rmse_value:.4f} → {rmse_perf}")


if __name__ == "__main__":
    print("Extended Sklearn Metrics - Custom Thresholds Examples\n")
    
    example_1_default_vs_custom_thresholds()
    example_2_domain_specific_thresholds()
    example_3_progressive_threshold_tuning()
    example_4_threshold_sensitivity_analysis()
    
    print("\n" + "=" * 70)
    print("All custom threshold examples completed successfully!")
    print("=" * 70)