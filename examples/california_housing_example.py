import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearnMetrics import evaluate_model_with_cross_validation

# Load dataset
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Split into training and testing sets
print("Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and evaluate model
print("Creating and evaluating model...")
model = LinearRegression()
target_range = y_train.max() - y_train.min()

# Evaluate model using cross-validation
performance_table = evaluate_model_with_cross_validation(
    model=model,
    X=X_train_scaled,
    y=y_train,
    cv=5,
    target_range=target_range
)

# Display results
print("\nModel Performance Summary:")
print("=" * 80)
print(performance_table.to_string(index=False))
print("=" * 80) 