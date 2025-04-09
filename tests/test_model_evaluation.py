import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearnMetrics import evaluate_model_with_cross_validation

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create synthetic regression dataset
        X, y = make_regression(
            n_samples=100,
            n_features=5,
            noise=0.1,
            random_state=42
        )
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.model = LinearRegression()
        self.target_range = np.max(self.y) - np.min(self.y)

    def test_output_format(self):
        """Test if the output DataFrame has the correct format"""
        result = evaluate_model_with_cross_validation(
            model=self.model,
            X=self.X,
            y=self.y,
            cv=5,
            target_range=self.target_range
        )
        
        # Check DataFrame structure
        expected_columns = ['Metric', 'Value', 'Threshold', 'Calculation', 'Performance']
        self.assertEqual(list(result.columns), expected_columns)
        
        # Check metrics are present
        expected_metrics = ['RMSE', 'MAE', 'R²', 'Explained Variance']
        self.assertEqual(list(result['Metric']), expected_metrics)
        
        # Check number of rows
        self.assertEqual(len(result), 4)

    def test_performance_categories(self):
        """Test if performance categories are correctly assigned"""
        result = evaluate_model_with_cross_validation(
            model=self.model,
            X=self.X,
            y=self.y,
            cv=5,
            target_range=self.target_range
        )
        
        # Check that performance categories are valid
        valid_categories = {'Excellent', 'Good', 'Moderate', 'Poor', 'Acceptable'}
        for performance in result['Performance']:
            self.assertIn(performance, valid_categories)

    def test_value_ranges(self):
        """Test if metric values are within expected ranges"""
        result = evaluate_model_with_cross_validation(
            model=self.model,
            X=self.X,
            y=self.y,
            cv=5,
            target_range=self.target_range
        )
        
        # R² and Explained Variance should be between 0 and 1
        r2_idx = result['Metric'] == 'R²'
        exp_var_idx = result['Metric'] == 'Explained Variance'
        
        self.assertGreaterEqual(result.loc[r2_idx, 'Value'].iloc[0], 0)
        self.assertLessEqual(result.loc[r2_idx, 'Value'].iloc[0], 1)
        self.assertGreaterEqual(result.loc[exp_var_idx, 'Value'].iloc[0], 0)
        self.assertLessEqual(result.loc[exp_var_idx, 'Value'].iloc[0], 1)
        
        # RMSE and MAE should be positive
        rmse_idx = result['Metric'] == 'RMSE'
        mae_idx = result['Metric'] == 'MAE'
        
        self.assertGreater(result.loc[rmse_idx, 'Value'].iloc[0], 0)
        self.assertGreater(result.loc[mae_idx, 'Value'].iloc[0], 0)

    def test_different_input_types(self):
        """Test if function works with different input types"""
        # Test with numpy arrays
        X_numpy = self.X.to_numpy()
        y_numpy = self.y.to_numpy()
        
        result_numpy = evaluate_model_with_cross_validation(
            model=self.model,
            X=X_numpy,
            y=y_numpy,
            cv=5,
            target_range=self.target_range
        )
        
        self.assertIsInstance(result_numpy, pd.DataFrame)
        
        # Test with pandas objects
        result_pandas = evaluate_model_with_cross_validation(
            model=self.model,
            X=self.X,
            y=self.y,
            cv=5,
            target_range=self.target_range
        )
        
        self.assertIsInstance(result_pandas, pd.DataFrame)

    def test_target_range_calculation(self):
        """Test if target_range is correctly calculated when not provided"""
        result = evaluate_model_with_cross_validation(
            model=self.model,
            X=self.X,
            y=self.y,
            cv=5
        )
        
        # Check if function runs without target_range
        self.assertIsInstance(result, pd.DataFrame)
        
        # Verify calculations use correct target range
        expected_range = np.max(self.y) - np.min(self.y)
        rmse_idx = result['Metric'] == 'RMSE'
        rmse_value = result.loc[rmse_idx, 'Value'].iloc[0]
        calculation = result.loc[rmse_idx, 'Calculation'].iloc[0]
        
        self.assertIn(f'{expected_range:.2f}', calculation)

if __name__ == '__main__':
    unittest.main() 