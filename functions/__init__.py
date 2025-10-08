"""
functions package: shared utilities for SVM, loss, optimization, and helpers.

This re-exports the main entry points so notebooks can do:
  from functions import *
"""

from .loss import phi
from .svm import svm, check_gradient
from .gradient_descent import gradient_descent, sigma_max_sq
from .linear_gradient import (
    linesearch,
    ls_gradient_descent,
    showMNISTImage,
    showMNISTImages_many,
    binary_classifier_accuracy,
)

__all__ = [
    "phi",
    "svm",
    "check_gradient",
    "gradient_descent",
    "sigma_max_sq",
    "linesearch",
    "ls_gradient_descent",
    "showMNISTImage",
    "showMNISTImages_many",
    "binary_classifier_accuracy",
]
