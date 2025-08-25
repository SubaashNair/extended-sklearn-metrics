"""
Example demonstrating ROC/AUC analysis with threshold optimization for classification models.

This example shows how to use the ROC/AUC analysis features for binary and multiclass classification,
including threshold optimization and comprehensive visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from extended_sklearn_metrics import (
    calculate_roc_metrics,
    calculate_multiclass_roc_metrics,
    calculate_precision_recall_metrics,
    find_optimal_thresholds,
    create_threshold_analysis_report,
    print_roc_auc_summary,
    create_roc_curve_plot,
    create_precision_recall_plot,
    create_multiclass_roc_plot,
    create_threshold_analysis_plot
)


def example_binary_classification():
    """Example with binary classification using synthetic data"""
    print("=" * 70)
    print("EXAMPLE 1: BINARY CLASSIFICATION - SYNTHETIC DATA")
    print("=" * 70)
    
    # Create synthetic binary classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],  # Imbalanced dataset
        flip_y=0.05,  # Add some noise
        random_state=42
    )
    
    # Convert to DataFrame/Series for better display
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    y_series = pd.Series(y)
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Class distribution: {dict(y_series.value_counts())}")
    print(f"Class imbalance ratio: {y_series.value_counts()[0]/y_series.value_counts()[1]:.2f}:1")
    print()
    
    # Create and fit model
    model = LogisticRegression(random_state=42)
    
    # Calculate ROC metrics
    roc_metrics = calculate_roc_metrics(
        model=model,
        X=X_df,
        y=y_series,
        cv=5,
        pos_label=1
    )
    
    # Calculate Precision-Recall metrics
    pr_metrics = calculate_precision_recall_metrics(
        model=model,
        X=X_df,
        y=y_series,
        cv=5,
        pos_label=1
    )
    
    # Print comprehensive analysis
    print_roc_auc_summary(roc_metrics, pr_metrics)
    print()
    
    # Show optimal thresholds using different criteria
    optimal_thresholds = find_optimal_thresholds(
        roc_metrics, 
        criteria=['youden', 'closest_to_perfect', 'balanced_accuracy']
    )
    
    print("THRESHOLD OPTIMIZATION:")
    print("-" * 25)
    print(optimal_thresholds.to_string(index=False))
    print()
    
    # Create visualizations
    print("Generating ROC/AUC visualizations...")
    try:
        # ROC Curve
        create_roc_curve_plot(roc_metrics, title="Synthetic Binary Classification - ROC Curve")
        
        # Precision-Recall Curve
        create_precision_recall_plot(pr_metrics, title="Synthetic Binary Classification - PR Curve")
        
        # Comprehensive threshold analysis
        create_threshold_analysis_plot(
            roc_metrics, pr_metrics, 
            title="Comprehensive Threshold Analysis - Synthetic Data"
        )
        
        print("✅ All visualizations created successfully")
        
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualizations")
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {e}")
    
    print()


def example_breast_cancer_data():
    """Example with real-world breast cancer dataset"""
    print("=" * 70)
    print("EXAMPLE 2: BINARY CLASSIFICATION - BREAST CANCER DATASET")
    print("=" * 70)
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Convert to DataFrame for better handling
    X_df = pd.DataFrame(X, columns=cancer.feature_names)
    y_series = pd.Series(y)
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Features: {X_df.shape[1]} (mean, SE, and worst values for cell nuclei)")
    print(f"Classes: {dict(zip(cancer.target_names, [sum(y==i) for i in range(2)]))}")
    print(f"Target: 0=malignant, 1=benign")
    print()
    
    # Create pipeline with multiple models for comparison
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    best_auc = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Analyzing {name}...")
        
        # Calculate ROC metrics
        roc_metrics = calculate_roc_metrics(model, X_df, y_series, cv=5, pos_label=1)
        
        # Track best model
        if roc_metrics['roc_auc'] > best_auc:
            best_auc = roc_metrics['roc_auc']
            best_model_name = name
            best_roc_metrics = roc_metrics
        
        # Print summary for each model
        print(f"  ROC AUC: {roc_metrics['roc_auc']:.4f}")
        print(f"  Optimal Threshold: {roc_metrics['optimal_threshold']:.4f}")
        print(f"  Youden Index: {roc_metrics['optimal_youden_index']:.4f}")
        print()
    
    # Detailed analysis of best model
    print(f"DETAILED ANALYSIS - BEST MODEL: {best_model_name}")
    print("=" * 50)
    
    # Calculate both ROC and PR metrics for best model
    pr_metrics = calculate_precision_recall_metrics(
        models[best_model_name], X_df, y_series, cv=5, pos_label=1
    )
    
    # Print comprehensive report
    print_roc_auc_summary(best_roc_metrics, pr_metrics)
    
    # Create threshold analysis report
    threshold_report = create_threshold_analysis_report(best_roc_metrics, pr_metrics)
    print("\nTHRESHOLD ANALYSIS REPORT:")
    print("-" * 25)
    print(threshold_report.to_string(index=False))
    print()


def example_multiclass_classification():
    """Example with multiclass classification using Iris dataset"""
    print("=" * 70)
    print("EXAMPLE 3: MULTICLASS CLASSIFICATION - IRIS DATASET")
    print("=" * 70)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=iris.feature_names)
    y_series = pd.Series(y)
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Features: {list(X_df.columns)}")
    print(f"Classes: {dict(zip(iris.target_names, [sum(y==i) for i in range(3)]))}")
    print()
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Calculate multiclass ROC metrics
    multiclass_metrics = calculate_multiclass_roc_metrics(
        model=model,
        X=X_df,
        y=y_series,
        cv=5
    )
    
    print("MULTICLASS ROC ANALYSIS:")
    print("-" * 25)
    print(f"Number of classes: {multiclass_metrics['n_classes']}")
    print(f"Micro-average ROC AUC: {multiclass_metrics['micro_average']['roc_auc']:.4f}")
    print(f"Macro-average ROC AUC: {multiclass_metrics['macro_average']['roc_auc']:.4f}")
    print()
    
    print("PER-CLASS RESULTS:")
    print("-" * 18)
    class_results = multiclass_metrics['class_results']
    class_labels = multiclass_metrics['class_labels']
    
    for i, class_label in enumerate(class_labels):
        class_name = iris.target_names[class_label]
        class_data = class_results[class_label]
        print(f"  Class {class_label} ({class_name}):")
        print(f"    ROC AUC: {class_data['roc_auc']:.4f}")
        print(f"    Optimal Threshold: {class_data['optimal_threshold']:.4f}")
        print(f"    Optimal TPR: {class_data['optimal_tpr']:.4f}")
        print(f"    Optimal FPR: {class_data['optimal_fpr']:.4f}")
    print()
    
    # Create multiclass ROC plot
    try:
        create_multiclass_roc_plot(
            multiclass_metrics, 
            title="Multiclass ROC Analysis - Iris Dataset"
        )
        print("✅ Multiclass ROC plot created successfully")
    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"⚠️  Error creating visualization: {e}")
    
    print()


def example_threshold_sensitivity_analysis():
    """Example showing threshold sensitivity analysis"""
    print("=" * 70)
    print("EXAMPLE 4: THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Create dataset with clear decision boundary
    X, y = make_classification(
        n_samples=500,
        n_features=2,  # 2D for easy visualization
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=123
    )
    
    X_df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    y_series = pd.Series(y)
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print("Purpose: Analyze how threshold choice affects classification performance")
    print()
    
    # Create model
    model = LogisticRegression(random_state=42)
    
    # Calculate metrics
    roc_metrics = calculate_roc_metrics(model, X_df, y_series, cv=5)
    pr_metrics = calculate_precision_recall_metrics(model, X_df, y_series, cv=5)
    
    # Show different threshold optimization criteria
    optimal_thresholds = find_optimal_thresholds(
        roc_metrics, 
        criteria=['youden', 'closest_to_perfect', 'balanced_accuracy']
    )
    
    print("THRESHOLD OPTIMIZATION COMPARISON:")
    print("-" * 35)
    print(optimal_thresholds.to_string(index=False))
    print()
    
    # Analyze threshold sensitivity
    print("THRESHOLD SENSITIVITY ANALYSIS:")
    print("-" * 30)
    
    threshold_df = roc_metrics['threshold_metrics']
    
    # Show key threshold points
    interesting_thresholds = [0.3, 0.5, 0.7, roc_metrics['optimal_threshold']]
    
    print(f"{'Threshold':<12} {'TPR':<8} {'FPR':<8} {'TNR':<8} {'Precision*':<12} {'F1*':<8}")
    print("-" * 60)
    
    for thresh in interesting_thresholds:
        # Find closest threshold in our data
        idx = np.argmin(np.abs(threshold_df['threshold'].values - thresh))
        row = threshold_df.iloc[idx]
        
        # Calculate precision and F1 (approximation)
        tpr, fpr = row['tpr'], row['fpr']
        
        # Estimate precision (this is approximate without actual predictions)
        pos_rate = np.sum(y_series == 1) / len(y_series)  # Prior probability
        precision_est = (tpr * pos_rate) / (tpr * pos_rate + fpr * (1 - pos_rate))
        f1_est = 2 * (precision_est * tpr) / (precision_est + tpr) if (precision_est + tpr) > 0 else 0
        
        if thresh == roc_metrics['optimal_threshold']:
            print(f"{thresh:<12.3f} {tpr:<8.3f} {fpr:<8.3f} {row['tnr']:<8.3f} {precision_est:<12.3f} {f1_est:<8.3f} ← Optimal")
        else:
            print(f"{thresh:<12.3f} {tpr:<8.3f} {fpr:<8.3f} {row['tnr']:<8.3f} {precision_est:<12.3f} {f1_est:<8.3f}")
    
    print("*Precision and F1 are estimated based on population statistics")
    print()
    
    # Show recommendations
    print("THRESHOLD SELECTION RECOMMENDATIONS:")
    print("-" * 35)
    if roc_metrics['roc_auc'] > 0.8:
        print("✅ Good model performance - threshold optimization can provide significant gains")
        print(f"  • Use {roc_metrics['optimal_threshold']:.3f} for balanced classification")
        print(f"  • Adjust based on cost of false positives vs false negatives")
    else:
        print("⚠️  Moderate model performance - focus on feature engineering first")
        print("  • Threshold optimization provides limited gains with poor models")
    
    print()


def main():
    """Run all ROC/AUC analysis examples"""
    print("EXTENDED SKLEARN METRICS - ROC/AUC ANALYSIS EXAMPLES")
    print("=" * 70)
    print("This example demonstrates comprehensive ROC/AUC analysis capabilities:")
    print("• Binary classification with ROC and Precision-Recall curves")
    print("• Threshold optimization using multiple criteria")
    print("• Multiclass classification with one-vs-rest ROC analysis")  
    print("• Comprehensive visualizations with threshold sensitivity")
    print()
    
    # Run examples
    example_binary_classification()
    example_breast_cancer_data()
    example_multiclass_classification()
    example_threshold_sensitivity_analysis()
    
    print("=" * 70)
    print("SUMMARY - ROC/AUC ANALYSIS CAPABILITIES")
    print("=" * 70)
    print("The ROC/AUC analysis functionality provides:")
    print()
    print("✅ BINARY CLASSIFICATION:")
    print("   • ROC curve calculation and AUC scoring")
    print("   • Precision-Recall curves and AP scoring")
    print("   • Threshold optimization (Youden's Index, closest to perfect, balanced accuracy)")
    print("   • Cross-validated predictions for robust analysis")
    print()
    print("✅ MULTICLASS CLASSIFICATION:")
    print("   • One-vs-Rest ROC curves for each class")
    print("   • Micro and macro-averaged ROC curves")
    print("   • Per-class threshold optimization")
    print()
    print("✅ THRESHOLD ANALYSIS:")
    print("   • Multiple optimization criteria")
    print("   • Sensitivity analysis across threshold range")
    print("   • Trade-off analysis (TPR vs FPR vs Precision)")
    print("   • Customizable threshold selection")
    print()
    print("✅ COMPREHENSIVE VISUALIZATIONS:")
    print("   • ROC curves with optimal threshold highlighting")
    print("   • Precision-Recall curves with F1 optimization")
    print("   • Multi-panel threshold analysis plots")
    print("   • Score distribution analysis by true class")
    print("   • Multiclass ROC comparison plots")
    print()
    print("✅ INTEGRATION:")
    print("   • Works with any sklearn-compatible classifier")
    print("   • Cross-validation for robust estimates")
    print("   • Handles imbalanced datasets appropriately")
    print("   • Supports both probability and decision function outputs")
    print()
    print("Use ROC/AUC analysis to optimize classification thresholds and understand model performance!")
    print("=" * 70)


if __name__ == "__main__":
    main()