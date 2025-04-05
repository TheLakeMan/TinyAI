#!/usr/bin/env python3
"""
TinyAI Model Conversion Tool

This tool converts models from popular deep learning frameworks to TinyAI format.
Supported input formats:
- TensorFlow/Keras (.h5, .pb, SavedModel)
- PyTorch (.pt, .pth)
- ONNX (.onnx)

Usage:
    python -m tools.conversion.convert_model [options] <input_model> <output_model>
    or
    cd tools/conversion && python convert_model.py [options] <input_model> <output_model>

Options:
    --input-format FORMAT    Input model format (tensorflow, pytorch, onnx, auto)
    --quantize BITS          Quantize weights to specified bits (4, 8, 16, 32)
    --mixed-precision        Use mixed precision quantization
    --prune THRESHOLD        Prune weights below threshold (0.0-1.0)
    --weight-clustering N    Apply weight clustering with N clusters
    --optimize               Apply additional optimizations
    --verbose                Show detailed conversion information
    --help                   Show this help message
"""

import os
import sys
import argparse
import importlib
import logging
from pathlib import Path

# Handle import paths when script is run directly vs as a module
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to sys.path for direct script execution
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from conversion.config import get_converter_settings
else:
    # Normal import when used as a module
    from .config import get_converter_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tinyai_converter')

def detect_format(model_path):
    """Detect the model format based on file extension."""
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext in ['.h5', '.pb', '.keras']:
        return 'tensorflow'
    elif ext in ['.pt', '.pth']:
        return 'pytorch'
    elif ext in ['.onnx']:
        return 'onnx'
    else:
        # Try to detect based on file content
        try:
            with open(model_path, 'rb') as f:
                header = f.read(10)
                
                # Check for ONNX magic number
                if header[:6] == b'ONNX-ML':
                    return 'onnx'
                    
                # Check for TensorFlow SavedModel
                if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, 'saved_model.pb')):
                    return 'tensorflow'
        except Exception:
            pass
    
    return None

def validate_model_path(model_path):
    """Validate that the model file exists."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    return True

def get_converter_module(format_name):
    """Import and return the appropriate converter module."""
    try:
        # Handle import paths for both direct execution and module import
        if __name__ == "__main__" and __package__ is None:
            # Direct script execution
            if format_name == 'tensorflow':
                from conversion.tensorflow.tensorflow_converter import TensorFlowConverter
                return TensorFlowConverter
            elif format_name == 'pytorch':
                from conversion.pytorch.pytorch_converter import PyTorchConverter
                return PyTorchConverter
            elif format_name == 'onnx':
                from conversion.onnx.onnx_converter import ONNXConverter
                return ONNXConverter
        else:
            # Module import
            if format_name == 'tensorflow':
                from .tensorflow.tensorflow_converter import TensorFlowConverter
                return TensorFlowConverter
            elif format_name == 'pytorch':
                from .pytorch.pytorch_converter import PyTorchConverter
                return PyTorchConverter
            elif format_name == 'onnx':
                from .onnx.onnx_converter import ONNXConverter
                return ONNXConverter
        
        # Handle unsupported formats
        logger.error(f"Unsupported model format: {format_name}")
        return None
        
    except ImportError as e:
        logger.error(f"Error importing converter module: {e}")
        logger.error(f"Make sure the required dependencies are installed.")
        
        # Provide more specific error guidance
        if format_name == 'tensorflow':
            logger.error("For TensorFlow conversion, install TensorFlow: pip install tensorflow")
        elif format_name == 'pytorch':
            logger.error("For PyTorch conversion, install PyTorch: pip install torch")
        elif format_name == 'onnx':
            logger.error("For ONNX conversion, install ONNX: pip install onnx onnxruntime")
        
        return None

def main():
    parser = argparse.ArgumentParser(description='Convert models to TinyAI format')
    parser.add_argument('input_model', help='Path to input model file')
    parser.add_argument('output_model', help='Path to output TinyAI model file')
    parser.add_argument('--input-format', choices=['tensorflow', 'pytorch', 'onnx', 'auto'], 
                        default='auto', help='Input model format')
    parser.add_argument('--quantize', type=int, choices=[4, 8, 16, 32],
                        default=4, help='Quantize weights to specified bits')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Use mixed precision quantization')
    parser.add_argument('--prune', type=float, 
                        help='Prune weights below threshold (0.0-1.0)')
    parser.add_argument('--weight-clustering', type=int,
                        help='Apply weight clustering with N clusters')
    parser.add_argument('--optimize', action='store_true',
                        help='Apply additional optimizations')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed conversion information')
    
    args = parser.parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate input model path
    if not validate_model_path(args.input_model):
        sys.exit(1)
    
    # Detect or validate input format
    input_format = args.input_format
    if input_format == 'auto':
        input_format = detect_format(args.input_model)
        if not input_format:
            logger.error(f"Could not automatically detect model format for {args.input_model}")
            logger.error("Please specify the format using --input-format")
            sys.exit(1)
        logger.info(f"Detected input format: {input_format}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output_model))
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the appropriate converter
    try:
        converter_module = get_converter_module(input_format)
        if not converter_module:
            sys.exit(1)
        
        # Get default settings from config and override with command-line args
        settings = get_converter_settings(input_format)
        
        # Override settings with command-line arguments when provided
        if args.quantize is not None:
            settings["quantize_bits"] = args.quantize
            
        if args.mixed_precision:
            settings["mixed_precision"] = True
            
        if args.prune is not None:
            settings["prune_threshold"] = args.prune
            
        if args.weight_clustering is not None:
            settings["weight_clustering"] = args.weight_clustering
            
        if args.optimize:
            settings["optimize"] = True
            
        # Set up converter with merged options
        converter = converter_module(
            quantize_bits=settings["quantize_bits"],
            mixed_precision=settings["mixed_precision"],
            prune_threshold=settings["prune_threshold"],
            weight_clustering=settings["weight_clustering"],
            optimize=settings["optimize"],
            verbose=args.verbose
        )
        
        # Log configuration info in verbose mode
        if args.verbose:
            logger.debug("Conversion settings:")
            for key, value in settings.items():
                if isinstance(value, dict):
                    logger.debug(f"  {key}:")
                    for subkey, subvalue in value.items():
                        logger.debug(f"    {subkey}: {subvalue}")
                else:
                    logger.debug(f"  {key}: {value}")
        
        # Convert the model
        logger.info(f"Converting {args.input_model} to TinyAI format...")
        result = converter.convert(args.input_model, args.output_model)
        
        if result:
            logger.info(f"Conversion successful!")
            logger.info(f"Model saved to: {args.output_model}")
            
            # Print model details
            logger.info("Model details:")
            for key, value in result.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.error("Conversion failed.")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
