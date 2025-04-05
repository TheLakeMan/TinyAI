"""
TinyAI Model Conversion Package

This package provides tools for converting models from popular deep learning frameworks to TinyAI format.
Supported input formats:
- TensorFlow/Keras (.h5, .pb, SavedModel)
- PyTorch (.pt, .pth)
- ONNX (.onnx)
"""

import os
import sys

# Handle direct imports vs module imports
try:
    # When imported as a package
    from .convert_model import detect_format, validate_model_path
    from .config import get_converter_settings
except ImportError:
    # When running from this directory
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from convert_model import detect_format, validate_model_path
    from config import get_converter_settings

# Exposed modules
__all__ = [
    'detect_format',
    'validate_model_path',
    'get_converter_settings',
    # Converter classes are imported on-demand to avoid unnecessary dependencies
]

# Version information
__version__ = '1.0.0'
