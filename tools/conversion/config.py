"""
Configuration Settings for TinyAI Model Conversion

This module provides default configuration values and settings for model conversion.
These settings can be overridden via command-line arguments or environment variables.
"""

import os
import sys
from typing import Dict, Any

# Handle import paths when script is run directly
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Default quantization settings
DEFAULT_QUANTIZATION_BITS = 4  # 4-bit quantization by default
DEFAULT_MIXED_PRECISION = False  # Single precision by default
DEFAULT_PRUNE_THRESHOLD = None  # No pruning by default
DEFAULT_WEIGHT_CLUSTERING = None  # No weight clustering by default
DEFAULT_OPTIMIZE = True  # Apply optimizations by default

# Framework-specific settings
TENSORFLOW_SETTINGS = {
    "use_saved_model_format": True,  # Prefer SavedModel format over H5
    "include_optimizer": False,      # Don't include optimizer state in export
    "fuse_operations": True,         # Fuse operations where possible
    "strip_debug_ops": True,         # Remove debug operations
}

PYTORCH_SETTINGS = {
    "use_jit": True,                # Use TorchScript when possible
    "trace_inputs": None,           # Default input shapes for tracing
    "export_params": True,          # Include parameters in export
    "optimize_for_inference": True, # Apply inference optimizations
}

ONNX_SETTINGS = {
    "opset_version": 15,            # Default ONNX opset version
    "optimize_model": True,         # Run ONNX optimizations
    "check_model": True,            # Verify ONNX model validity
    "simplify": True,               # Run simplification passes
}

# Output settings
OUTPUT_FORMATS = ["tinyai", "json", "binary"]
DEFAULT_OUTPUT_FORMAT = "tinyai"

# Environment variable overrides
def get_env_settings() -> Dict[str, Any]:
    """Get settings from environment variables."""
    env_settings = {}
    
    # Quantization settings
    if "TINYAI_QUANTIZE_BITS" in os.environ:
        try:
            env_settings["quantize_bits"] = int(os.environ["TINYAI_QUANTIZE_BITS"])
        except ValueError:
            pass
    
    if "TINYAI_MIXED_PRECISION" in os.environ:
        env_settings["mixed_precision"] = os.environ["TINYAI_MIXED_PRECISION"].lower() in ["true", "1", "yes"]
    
    if "TINYAI_PRUNE_THRESHOLD" in os.environ:
        try:
            env_settings["prune_threshold"] = float(os.environ["TINYAI_PRUNE_THRESHOLD"])
        except ValueError:
            pass
    
    if "TINYAI_WEIGHT_CLUSTERING" in os.environ:
        try:
            env_settings["weight_clustering"] = int(os.environ["TINYAI_WEIGHT_CLUSTERING"])
        except ValueError:
            pass
    
    if "TINYAI_OPTIMIZE" in os.environ:
        env_settings["optimize"] = os.environ["TINYAI_OPTIMIZE"].lower() in ["true", "1", "yes"]
    
    # Framework specific settings
    if "TINYAI_TF_USE_SAVED_MODEL" in os.environ:
        env_settings["tf_use_saved_model"] = os.environ["TINYAI_TF_USE_SAVED_MODEL"].lower() in ["true", "1", "yes"]
    
    if "TINYAI_PT_USE_JIT" in os.environ:
        env_settings["pt_use_jit"] = os.environ["TINYAI_PT_USE_JIT"].lower() in ["true", "1", "yes"]
    
    if "TINYAI_ONNX_OPSET" in os.environ:
        try:
            env_settings["onnx_opset_version"] = int(os.environ["TINYAI_ONNX_OPSET"])
        except ValueError:
            pass
    
    # Output format
    if "TINYAI_OUTPUT_FORMAT" in os.environ:
        format_val = os.environ["TINYAI_OUTPUT_FORMAT"].lower()
        if format_val in OUTPUT_FORMATS:
            env_settings["output_format"] = format_val
    
    return env_settings

# Load settings from environment variables
ENV_SETTINGS = get_env_settings()

def get_converter_settings(framework=None):
    """Get conversion settings, optionally for a specific framework."""
    # Start with defaults
    settings = {
        "quantize_bits": ENV_SETTINGS.get("quantize_bits", DEFAULT_QUANTIZATION_BITS),
        "mixed_precision": ENV_SETTINGS.get("mixed_precision", DEFAULT_MIXED_PRECISION),
        "prune_threshold": ENV_SETTINGS.get("prune_threshold", DEFAULT_PRUNE_THRESHOLD),
        "weight_clustering": ENV_SETTINGS.get("weight_clustering", DEFAULT_WEIGHT_CLUSTERING),
        "optimize": ENV_SETTINGS.get("optimize", DEFAULT_OPTIMIZE),
        "output_format": ENV_SETTINGS.get("output_format", DEFAULT_OUTPUT_FORMAT),
    }
    
    # Add framework-specific settings if requested
    if framework == "tensorflow":
        tf_settings = TENSORFLOW_SETTINGS.copy()
        if "tf_use_saved_model" in ENV_SETTINGS:
            tf_settings["use_saved_model_format"] = ENV_SETTINGS["tf_use_saved_model"]
        settings["tensorflow"] = tf_settings
    
    elif framework == "pytorch":
        pt_settings = PYTORCH_SETTINGS.copy()
        if "pt_use_jit" in ENV_SETTINGS:
            pt_settings["use_jit"] = ENV_SETTINGS["pt_use_jit"]
        settings["pytorch"] = pt_settings
    
    elif framework == "onnx":
        onnx_settings = ONNX_SETTINGS.copy()
        if "onnx_opset_version" in ENV_SETTINGS:
            onnx_settings["opset_version"] = ENV_SETTINGS["onnx_opset_version"]
        settings["onnx"] = onnx_settings
    
    return settings
