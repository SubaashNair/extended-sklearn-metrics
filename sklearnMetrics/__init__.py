"""
sklearn-metrics - A library for evaluating scikit-learn regression models with comprehensive metrics
"""

from .model_evaluation import evaluate_model_with_cross_validation

__version__ = "0.1.0"
__all__ = ["evaluate_model_with_cross_validation"] 