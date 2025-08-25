"""
Comprehensive Model Evaluation Example

This example demonstrates the full suite of evaluation capabilities including:
- Hold-out test evaluation
- Feature importance analysis (built-in + permutation)
- Model interpretation and complexity assessment
- Comprehensive error analysis
- Fairness evaluation by demographic segments
- Professional reporting and visualizations

Works for both classification and regression tasks.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from extended_sklearn_metrics import (
    final_model_evaluation,
    create_evaluation_report,
    print_evaluation_summary,
    create_feature_importance_report,
    create_fairness_report,
    create_comprehensive_evaluation_plots,
    create_feature_importance_plot,
    create_fairness_comparison_plot
)


def classification_evaluation_example():
    """Comprehensive evaluation example for classification with fairness analysis."""
    print("=" * 80)
    print("COMPREHENSIVE CLASSIFICATION EVALUATION EXAMPLE")
    print("=" * 80)
    
    # Create synthetic dataset with potential bias
    np.random.seed(42)
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],  # Class imbalance
        flip_y=0.05,  # Add some noise
        random_state=42
    )
    
    # Create feature names
    feature_names = [
        'income', 'education', 'age', 'employment_length', 'debt_ratio',
        'credit_score', 'loan_amount', 'property_value', 'savings',
        'dependents', 'residence_type', 'payment_history', 'account_balance',
        'investment_portfolio', 'insurance_coverage'
    ]
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='approved')
    
    # Create protected attributes (simulated demographic factors)
    np.random.seed(123)
    protected_attrs = {
        'gender': np.random.choice(['Male', 'Female'], size=len(y), p=[0.6, 0.4]),
        'age_group': np.random.choice(['Young', 'Middle', 'Senior'], size=len(y), p=[0.3, 0.5, 0.2]),
        'ethnicity': np.random.choice(['Group_A', 'Group_B', 'Group_C'], size=len(y), p=[0.7, 0.2, 0.1])
    }
    
    # Introduce subtle bias (higher approval rates for certain groups)
    bias_mask_gender = (protected_attrs['gender'] == 'Male') & (np.random.random(len(y)) < 0.1)
    y_series[bias_mask_gender] = 1
    
    bias_mask_age = (protected_attrs['age_group'] == 'Young') & (np.random.random(len(y)) < 0.05)
    y_series[bias_mask_age] = 0
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Features: {feature_names[:5]}... (financial/demographic)")
    print(f"Target: loan approval (0=denied, 1=approved)")
    print(f"Class distribution: {dict(y_series.value_counts())}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Get protected attributes for test set
    test_indices = y_test.index
    protected_attrs_test = {
        attr: values[test_indices] for attr, values in protected_attrs.items()
    }
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    print("Model trained. Running comprehensive evaluation...")
    print()
    
    # Comprehensive evaluation
    evaluation_results = final_model_evaluation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='classification',
        cv_folds=5,
        feature_names=feature_names,
        protected_attributes=protected_attrs_test,
        random_state=42
    )
    
    # Print comprehensive summary
    print_evaluation_summary(evaluation_results)
    print()
    
    # Create detailed reports
    print("DETAILED EVALUATION REPORTS:")
    print("-" * 30)
    
    # Main evaluation report
    eval_report = create_evaluation_report(evaluation_results)
    print("\\n1. MAIN EVALUATION REPORT:")
    print(eval_report.to_string(index=False))
    print()
    
    # Feature importance report
    fi_report = create_feature_importance_report(evaluation_results)
    if fi_report is not None:
        print("\\n2. FEATURE IMPORTANCE ANALYSIS:")
        print(fi_report.head(10).to_string(index=False))
        print()
    
    # Fairness report
    fairness_report = create_fairness_report(evaluation_results)
    if fairness_report is not None:
        print("\\n3. FAIRNESS ANALYSIS BY GROUP:")
        print(fairness_report.to_string(index=False))
        print()
    
    # Create visualizations
    print("Generating comprehensive visualizations...")
    try:
        # Main evaluation plots
        create_comprehensive_evaluation_plots(evaluation_results)
        
        # Detailed feature importance plot
        create_feature_importance_plot(evaluation_results, top_n=12)
        
        # Fairness comparison plots
        create_fairness_comparison_plot(evaluation_results)
        
        print("✅ All visualizations created successfully")
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualizations")
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")
    
    print()
    return evaluation_results


def regression_evaluation_example():
    """Comprehensive evaluation example for regression."""
    print("=" * 80)
    print("COMPREHENSIVE REGRESSION EVALUATION EXAMPLE")
    print("=" * 80)
    
    # Create synthetic regression dataset
    np.random.seed(42)
    X, y = make_regression(
        n_samples=1500,
        n_features=12,
        n_informative=8,
        noise=15,
        random_state=42
    )
    
    # Create meaningful feature names for housing price prediction
    feature_names = [
        'square_footage', 'bedrooms', 'bathrooms', 'age_of_house',
        'lot_size', 'garage_spaces', 'school_rating', 'crime_rate',
        'distance_to_downtown', 'property_tax', 'hoa_fees', 'neighborhood_income'
    ]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='house_price')
    
    # Create protected attributes for fairness analysis
    np.random.seed(456)
    protected_attrs = {
        'neighborhood_type': np.random.choice(['Urban', 'Suburban', 'Rural'], size=len(y), p=[0.4, 0.5, 0.1]),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], size=len(y), p=[0.3, 0.5, 0.2])
    }
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Features: housing characteristics and neighborhood factors")
    print(f"Target: house price (continuous)")
    print(f"Price range: [{y_series.min():.0f}, {y_series.max():.0f}]")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.25, random_state=42
    )
    
    # Get protected attributes for test set
    test_indices = y_test.index
    protected_attrs_test = {
        attr: values[test_indices] for attr, values in protected_attrs.items()
    }
    
    # Train model pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12))
    ])
    model.fit(X_train, y_train)
    
    print("Model pipeline trained. Running comprehensive evaluation...")
    print()
    
    # Comprehensive evaluation
    evaluation_results = final_model_evaluation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='regression',
        cv_folds=5,
        feature_names=feature_names,
        protected_attributes=protected_attrs_test,
        random_state=42
    )
    
    # Print comprehensive summary
    print_evaluation_summary(evaluation_results)
    print()
    
    # Create detailed reports
    print("DETAILED EVALUATION REPORTS:")
    print("-" * 30)
    
    # Main evaluation report
    eval_report = create_evaluation_report(evaluation_results)
    print("\\n1. MAIN EVALUATION REPORT:")
    print(eval_report.to_string(index=False))
    print()
    
    # Feature importance report
    fi_report = create_feature_importance_report(evaluation_results)
    if fi_report is not None:
        print("\\n2. TOP 8 MOST IMPORTANT FEATURES:")
        print(fi_report.head(8).to_string(index=False))
        print()
    
    # Create visualizations
    print("Generating comprehensive visualizations...")
    try:
        # Main evaluation plots
        create_comprehensive_evaluation_plots(evaluation_results)
        
        # Detailed feature importance plot
        create_feature_importance_plot(evaluation_results, top_n=10)
        
        print("✅ All visualizations created successfully")
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualizations")
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")
    
    print()
    return evaluation_results


def real_world_example():
    """Real-world example using breast cancer dataset."""
    print("=" * 80)
    print("REAL-WORLD EXAMPLE: BREAST CANCER DIAGNOSIS")
    print("=" * 80)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y_series = pd.Series(data.target, name='diagnosis')
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Features: Cell nuclei measurements (mean, SE, worst)")
    print(f"Target: {dict(zip(data.target_names, [sum(y_series==i) for i in range(2)]))}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    print("Logistic Regression model trained. Running evaluation...")
    print()
    
    # Comprehensive evaluation (no protected attributes for medical data)
    evaluation_results = final_model_evaluation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        task_type='classification',
        cv_folds=10,  # More folds for stability
        feature_names=list(data.feature_names),
        random_state=42
    )
    
    # Print summary
    print_evaluation_summary(evaluation_results)
    
    # Show top important features
    fi_report = create_feature_importance_report(evaluation_results)
    if fi_report is not None:
        print("\\nTOP 5 DIAGNOSTIC FEATURES:")
        print("-" * 25)
        top_features = fi_report.head(5)
        for _, row in top_features.iterrows():
            feature = row['Feature']
            if 'Permutation_Importance' in row:
                importance = row['Permutation_Importance']
                print(f"  {feature:30}: {importance:.4f}")
            elif 'Built_in_Importance' in row:
                importance = row['Built_in_Importance']
                print(f"  {feature:30}: {importance:.4f}")
    
    print()
    return evaluation_results


def main():
    """Run all comprehensive evaluation examples."""
    print("EXTENDED SKLEARN METRICS - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print("This example demonstrates advanced model evaluation capabilities:")
    print("• Hold-out test set evaluation with cross-validation stability")
    print("• Multi-method feature importance analysis")
    print("• Model interpretation and complexity assessment")
    print("• Comprehensive error analysis and diagnostics")
    print("• Fairness evaluation across demographic groups")
    print("• Professional reporting and visualizations")
    print()
    
    # Run examples
    print("🔬 Running Classification Example with Fairness Analysis...")
    classification_results = classification_evaluation_example()
    
    print("🏠 Running Regression Example...")
    regression_results = regression_evaluation_example()
    
    print("🏥 Running Real-World Medical Example...")
    medical_results = real_world_example()
    
    print("=" * 80)
    print("SUMMARY - COMPREHENSIVE EVALUATION CAPABILITIES")
    print("=" * 80)
    print("The comprehensive evaluation framework provides:")
    print()
    print("✅ HOLD-OUT EVALUATION:")
    print("   • Performance metrics on unseen test data")
    print("   • Cross-validation stability analysis")
    print("   • Automatic task type detection")
    print("   • Pipeline compatibility")
    print()
    print("✅ FEATURE ANALYSIS:")
    print("   • Built-in importance (tree-based models)")
    print("   • Permutation importance (model-agnostic)")
    print("   • Feature-target correlations")
    print("   • Feature interaction detection")
    print()
    print("✅ ERROR ANALYSIS:")
    print("   • Classification: confusion matrix, misclassification patterns")
    print("   • Regression: residual statistics, outlier analysis")
    print("   • Error correlation with features")
    print("   • Confidence and calibration analysis")
    print()
    print("✅ FAIRNESS EVALUATION:")
    print("   • Demographic parity analysis")
    print("   • Equal opportunity metrics")
    print("   • Group-specific performance")
    print("   • Disparity ratio calculations")
    print()
    print("✅ MODEL INTERPRETATION:")
    print("   • Complexity assessment")
    print("   • Prediction confidence analysis")
    print("   • Feature interaction detection")
    print("   • Model behavior insights")
    print()
    print("✅ PROFESSIONAL REPORTING:")
    print("   • Executive summaries with key findings")
    print("   • Detailed technical reports")
    print("   • Actionable recommendations")
    print("   • Export to pandas DataFrames")
    print()
    print("✅ COMPREHENSIVE VISUALIZATIONS:")
    print("   • Multi-panel evaluation dashboards")
    print("   • Feature importance plots")
    print("   • Fairness comparison charts")
    print("   • Error analysis visualizations")
    print()
    print("Use comprehensive evaluation for production-ready model assessment!")
    print("=" * 80)


if __name__ == "__main__":
    main()