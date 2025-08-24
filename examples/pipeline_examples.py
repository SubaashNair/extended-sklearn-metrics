"""
Examples demonstrating pipeline usage with extended-sklearn-metrics
"""
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from extended_sklearn_metrics import evaluate_model_with_cross_validation
import numpy as np
import pandas as pd

def example_1_basic_pipeline():
    """Example 1: Basic pipeline with scaling"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Pipeline with StandardScaler")
    print("=" * 60)
    
    # Load California Housing dataset
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
    
    print(result.to_string(index=False))
    print()

def example_2_complex_preprocessing():
    """Example 2: Complex preprocessing pipeline"""
    print("=" * 60)
    print("EXAMPLE 2: Complex Preprocessing Pipeline")
    print("=" * 60)
    
    # Create mixed data
    X_num, y = make_regression(n_samples=500, n_features=6, noise=0.2, random_state=42)
    X_cat = np.random.choice(['Low', 'Medium', 'High'], size=(500, 2))
    
    X = pd.DataFrame(X_num, columns=[f'feature_{i}' for i in range(6)])
    X['category_1'] = X_cat[:, 0]
    X['category_2'] = X_cat[:, 1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Complex preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), [f'feature_{i}' for i in range(6)]),
        ('cat', OneHotEncoder(drop='first'), ['category_1', 'category_2'])
    ])
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Evaluate
    result = evaluate_model_with_cross_validation(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=5
    )
    
    print("Pipeline steps:")
    for i, (name, transformer) in enumerate(pipeline.steps):
        print(f"  {i+1}. {name}: {type(transformer).__name__}")
    print()
    print(result.to_string(index=False))
    print()

def example_3_ensemble_pipeline():
    """Example 3: Pipeline with ensemble method"""
    print("=" * 60)
    print("EXAMPLE 3: Pipeline with Random Forest")
    print("=" * 60)
    
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    # Pipeline with ensemble method
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))
    ])
    
    # Evaluate
    result = evaluate_model_with_cross_validation(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=5
    )
    
    print(result.to_string(index=False))
    print()

def example_4_pipeline_comparison():
    """Example 4: Compare multiple pipelines"""
    print("=" * 60)
    print("EXAMPLE 4: Pipeline Comparison")
    print("=" * 60)
    
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    
    # Define multiple pipelines
    pipelines = {
        'Linear + Scaling': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Ridge + Scaling': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ]),
        'Poly + Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', Ridge(alpha=10.0))
        ])
    }
    
    # Compare pipelines
    for name, pipeline in pipelines.items():
        print(f"\n{name}:")
        print("-" * len(name))
        
        result = evaluate_model_with_cross_validation(
            model=pipeline,
            X=X_train,
            y=y_train,
            cv=3  # Reduced for faster computation
        )
        
        # Print key metrics
        r2_score = result.loc[result['Metric'] == 'R²', 'Value'].iloc[0]
        rmse_perf = result.loc[result['Metric'] == 'RMSE', 'Performance'].iloc[0]
        print(f"R² Score: {r2_score:.4f}")
        print(f"RMSE Performance: {rmse_perf}")

if __name__ == "__main__":
    print("Extended Sklearn Metrics - Pipeline Examples\n")
    
    example_1_basic_pipeline()
    example_2_complex_preprocessing()
    example_3_ensemble_pipeline()
    example_4_pipeline_comparison()
    
    print("=" * 60)
    print("All pipeline examples completed successfully!")
    print("=" * 60)