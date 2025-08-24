"""
Examples demonstrating classification evaluation with extended-sklearn-metrics
"""
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from extended_sklearn_metrics import evaluate_classification_model_with_cross_validation
import pandas as pd


def example_1_binary_classification():
    """Example 1: Binary classification with breast cancer dataset"""
    print("=" * 60)
    print("EXAMPLE 1: Binary Classification - Breast Cancer Dataset")
    print("=" * 60)
    
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Create and evaluate model
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    result = evaluate_classification_model_with_cross_validation(
        model=model,
        X=X_train,
        y=y_train,
        cv=5
    )
    
    print(f"Dataset: {len(X_train)} training samples, {X_train.shape[1]} features")
    print(f"Classes: {sorted(y_train.unique())}")
    print()
    print(result.to_string(index=False))
    print()


def example_2_multiclass_classification():
    """Example 2: Multiclass classification with Iris dataset"""
    print("=" * 60)
    print("EXAMPLE 2: Multiclass Classification - Iris Dataset")
    print("=" * 60)
    
    # Load dataset
    data = load_iris(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )
    
    # Create and evaluate model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    result = evaluate_classification_model_with_cross_validation(
        model=model,
        X=X_train,
        y=y_train,
        cv=5,
        average='weighted'  # Good for imbalanced multiclass
    )
    
    print(f"Dataset: {len(X_train)} training samples, {X_train.shape[1]} features")
    print(f"Classes: {sorted(y_train.unique())}")
    print("Averaging strategy: weighted")
    print()
    print(result.to_string(index=False))
    print()


def example_3_pipeline_classification():
    """Example 3: Classification with preprocessing pipeline"""
    print("=" * 60)
    print("EXAMPLE 3: Classification with Preprocessing Pipeline")
    print("=" * 60)
    
    # Create synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        n_redundant=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))  # probability=True for ROC AUC
    ])
    
    result = evaluate_classification_model_with_cross_validation(
        model=pipeline,
        X=X_train,
        y=y_train,
        cv=5
    )
    
    print(f"Dataset: {len(X_train)} training samples, {X_train.shape[1]} features")
    print("Pipeline: StandardScaler → SVM Classifier")
    print()
    print(result.to_string(index=False))
    print()


def example_4_averaging_strategies():
    """Example 4: Different averaging strategies for multiclass"""
    print("=" * 60)
    print("EXAMPLE 4: Averaging Strategies Comparison")
    print("=" * 60)
    
    # Create imbalanced multiclass dataset
    X, y = make_classification(
        n_samples=600,
        n_features=8,
        n_classes=3,
        n_informative=5,
        weights=[0.6, 0.3, 0.1],  # Imbalanced classes
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    averaging_strategies = ['micro', 'macro', 'weighted']
    
    for avg_strategy in averaging_strategies:
        print(f"\n{avg_strategy.upper()} AVERAGING:")
        print("-" * 20)
        
        result = evaluate_classification_model_with_cross_validation(
            model=model,
            X=X_train,
            y=y_train,
            cv=3,
            average=avg_strategy
        )
        
        # Print key metrics
        accuracy = result.loc[result['Metric'] == 'Accuracy', 'Value'].iloc[0]
        f1_score = result.loc[result['Metric'] == 'F1-Score', 'Value'].iloc[0]
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1_score:.4f}")


def example_5_model_comparison():
    """Example 5: Compare different classification models"""
    print("=" * 60)
    print("EXAMPLE 5: Classification Model Comparison")
    print("=" * 60)
    
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(probability=True, random_state=42)  # probability=True for ROC AUC
    }
    
    results_summary = []
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * len(name))
        
        result = evaluate_classification_model_with_cross_validation(
            model=model,
            X=X_train,
            y=y_train,
            cv=5
        )
        
        # Extract key metrics
        accuracy = result.loc[result['Metric'] == 'Accuracy', 'Value'].iloc[0]
        f1_score = result.loc[result['Metric'] == 'F1-Score', 'Value'].iloc[0]
        roc_auc = result.loc[result['Metric'] == 'ROC AUC', 'Value'].iloc[0]
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"ROC AUC:  {roc_auc:.4f}")
        
        results_summary.append({
            'Model': name,
            'Accuracy': accuracy,
            'F1-Score': f1_score,
            'ROC AUC': roc_auc
        })
    
    # Show comparison table
    print(f"\n{'=' * 60}")
    print("SUMMARY COMPARISON:")
    print("=" * 60)
    comparison_df = pd.DataFrame(results_summary)
    print(comparison_df.to_string(index=False, float_format='%.4f'))


if __name__ == "__main__":
    print("Extended Sklearn Metrics - Classification Examples\n")
    
    example_1_binary_classification()
    example_2_multiclass_classification()
    example_3_pipeline_classification()
    example_4_averaging_strategies()
    example_5_model_comparison()
    
    print("=" * 60)
    print("All classification examples completed successfully!")
    print("=" * 60)