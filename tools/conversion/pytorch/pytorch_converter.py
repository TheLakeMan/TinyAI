#!/usr/bin/env python3
"""
PyTorch to TinyAI Model Converter

This module converts PyTorch models to TinyAI format.
It handles:
- Loading PyTorch models (.pt, .pth)
- Extracting architecture and weights
- Applying quantization and optimization
- Creating TinyAI-compatible model files
"""

import os
import sys
import json
import struct
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

# Setup logging
logger = logging.getLogger('pytorch_converter')

class PyTorchConverter:
    """Converter for PyTorch models to TinyAI format."""
    
    def __init__(self, 
                 quantize_bits: int = 4, 
                 mixed_precision: bool = False,
                 prune_threshold: Optional[float] = None,
                 weight_clustering: Optional[int] = None,
                 optimize: bool = True,
                 verbose: bool = False):
        """
        Initialize the PyTorch converter.
        
        Args:
            quantize_bits: Number of bits for weight quantization (4, 8, 16, 32)
            mixed_precision: Whether to use mixed precision quantization
            prune_threshold: Prune weights below this threshold
            weight_clustering: Number of clusters for weight sharing
            optimize: Apply additional optimizations
            verbose: Show detailed conversion information
        """
        self.quantize_bits = quantize_bits
        self.mixed_precision = mixed_precision
        self.prune_threshold = prune_threshold
        self.weight_clustering = weight_clustering
        self.optimize = optimize
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Check for PyTorch installation
        try:
            import torch
            self.torch = torch
            logger.debug(f"PyTorch version: {torch.__version__}")
        except ImportError:
            logger.error("PyTorch is not installed. Please install PyTorch to use this converter.")
            sys.exit(1)
            
        # Import additional PyTorch modules
        try:
            import torch.nn as nn
            import torch.jit as jit
            self.nn = nn
            self.jit = jit
        except ImportError:
            logger.error("Failed to import PyTorch modules.")
            sys.exit(1)
    
    def convert(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Convert PyTorch model to TinyAI format.
        
        Args:
            input_path: Path to input PyTorch model
            output_path: Path to output TinyAI model file
            
        Returns:
            Dict with conversion statistics and metadata
        """
        # Load the model
        model = self._load_model(input_path)
        
        if model is None:
            return {}
        
        # Get model metadata
        metadata = self._extract_metadata(model)
        
        # Process weights
        processed_weights = self._process_weights(model)
        
        # Create TinyAI model structure
        tinyai_model = self._create_tinyai_model(model, metadata, processed_weights)
        
        # Save the model
        saved = self._save_model(tinyai_model, output_path)
        
        if not saved:
            return {}
        
        # Gather statistics
        stats = self._gather_statistics(model, tinyai_model, processed_weights)
        
        return stats
    
    def _load_model(self, model_path: str) -> Any:
        """Load a PyTorch model from file."""
        logger.info(f"Loading PyTorch model from {model_path}")
        
        try:
            # Try different loading methods
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # Standard PyTorch save format
                model = self.torch.load(model_path, map_location=self.torch.device('cpu'))
                
                # Handle different save formats
                if isinstance(model, dict):
                    # Check if this is a state_dict or full model with 'model' key
                    if 'model' in model:
                        model = model['model']
                    elif 'state_dict' in model:
                        logger.warning("Loaded state_dict without model architecture. Using dummy model for conversion.")
                        # Create a dummy model to attach state_dict to
                        from torch.nn import Module
                        dummy_model = Module()
                        dummy_model.load_state_dict(model['state_dict'])
                        model = dummy_model
                    else:
                        # Assume this is a state_dict directly
                        logger.warning("Loaded state_dict without model architecture. Using dummy model for conversion.")
                        dummy_model = self.nn.Module()
                        dummy_model.load_state_dict(model)
                        model = dummy_model
                
                # Put model in evaluation mode
                if hasattr(model, 'eval'):
                    model.eval()
                
                logger.debug("Loaded PyTorch model successfully")
                
            elif model_path.endswith('.jit') or model_path.endswith('.pt'):
                # Try loading as a TorchScript model
                try:
                    model = self.jit.load(model_path)
                    logger.debug("Loaded TorchScript model successfully")
                except Exception as e:
                    logger.error(f"Failed to load TorchScript model: {e}")
                    return None
            else:
                logger.error(f"Unsupported file format: {model_path}")
                return None
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from the PyTorch model."""
        metadata = {
            "framework": "pytorch",
            "version": self.torch.__version__,
            "model_type": type(model).__name__,
            "layers": []
        }
        
        # Extract model architecture information
        if hasattr(model, '_modules'):
            self._extract_module_info(model, metadata["layers"], prefix="")
        
        return metadata
    
    def _extract_module_info(self, module: Any, layers: List[Dict[str, Any]], prefix: str) -> None:
        """Recursively extract information from PyTorch modules."""
        for name, child in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            layer_type = child.__class__.__name__
            
            # Extract parameters and shapes
            params = {}
            for param_name, param in child.named_parameters(recurse=False):
                params[param_name] = {
                    "shape": list(param.shape),
                    "requires_grad": param.requires_grad
                }
            
            # Create layer info dictionary
            layer_info = {
                "name": layer_name,
                "type": layer_type,
                "parameters": params
            }
            
            # Extract additional configuration if available
            if hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                # Linear layer
                layer_info["config"] = {
                    "in_features": child.in_features,
                    "out_features": child.out_features,
                    "bias": child.bias is not None
                }
            elif hasattr(child, 'in_channels') and hasattr(child, 'out_channels'):
                # Convolutional layer
                layer_info["config"] = {
                    "in_channels": child.in_channels,
                    "out_channels": child.out_channels,
                    "kernel_size": child.kernel_size if hasattr(child, 'kernel_size') else None,
                    "stride": child.stride if hasattr(child, 'stride') else None,
                    "padding": child.padding if hasattr(child, 'padding') else None,
                    "bias": child.bias is not None if hasattr(child, 'bias') else None
                }
            
            layers.append(layer_info)
            
            # Recursively process child modules
            self._extract_module_info(child, layers, layer_name)
    
    def _process_weights(self, model: Any) -> Dict[str, Any]:
        """Process model weights with quantization and other optimizations."""
        processed_weights = {}
        
        # Collect all named parameters
        for name, param in model.named_parameters():
            # Skip parameters that don't require gradients (non-trainable)
            if not param.requires_grad and not self.verbose:
                continue
            
            # Get the parameter data as numpy array
            weight_data = param.detach().cpu().numpy()
            
            # Extract module name and parameter type
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                module_name, param_type = parts
            else:
                module_name = "model"
                param_type = parts[0]
            
            # Apply pruning if threshold is set
            if self.prune_threshold is not None:
                weight_data = self._prune_weights(weight_data, self.prune_threshold)
            
            # Apply weight clustering if specified
            if self.weight_clustering is not None:
                weight_data = self._apply_weight_clustering(weight_data, self.weight_clustering)
            
            # Apply quantization
            if "weight" in param_type or "bias" in param_type:
                if self.mixed_precision:
                    # For mixed precision, use different bit widths for different layers
                    # Weights generally benefit from higher precision than biases
                    if "weight" in param_type:
                        # Use specified bit width for weights
                        bits = self.quantize_bits
                    else:
                        # Use 8-bit for biases as a reasonable default
                        bits = 8
                    
                    quantized_data, scale, zero_point = self._quantize_weights(weight_data, bits)
                else:
                    # Use the same bit width for all weights
                    quantized_data, scale, zero_point = self._quantize_weights(weight_data, self.quantize_bits)
                
                # Prepare weight information
                weight_info = {
                    "data": quantized_data,
                    "scale": scale,
                    "zero_point": zero_point,
                    "bits": bits if self.mixed_precision else self.quantize_bits,
                    "shape": weight_data.shape
                }
            else:
                # For other parameters, just store as is
                weight_info = {
                    "data": weight_data,
                    "shape": weight_data.shape
                }
            
            # Store in the processed weights dictionary
            if module_name not in processed_weights:
                processed_weights[module_name] = {}
            
            processed_weights[module_name][param_type] = weight_info
        
        return processed_weights
    
    def _prune_weights(self, weights: np.ndarray, threshold: float) -> np.ndarray:
        """Prune weights below the threshold."""
        mask = np.abs(weights) > threshold
        pruned_weights = weights * mask
        sparsity = 1.0 - np.count_nonzero(mask) / mask.size
        logger.debug(f"Pruned weights with threshold {threshold}, sparsity: {sparsity:.2%}")
        return pruned_weights
    
    def _apply_weight_clustering(self, weights: np.ndarray, num_clusters: int) -> np.ndarray:
        """Apply weight clustering (weight sharing) to reduce model size."""
        from sklearn.cluster import KMeans
        
        original_shape = weights.shape
        flattened = weights.reshape(-1)
        
        # Skip if there are fewer unique values than clusters
        if len(np.unique(flattened)) <= num_clusters:
            return weights
        
        # Create clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(flattened.reshape(-1, 1))
        
        # Replace weights with cluster centroids
        clustered = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        clustered = clustered.reshape(original_shape)
        
        logger.debug(f"Applied weight clustering with {num_clusters} clusters")
        return clustered
    
    def _quantize_weights(self, weights: np.ndarray, bits: int) -> Tuple[np.ndarray, float, int]:
        """Quantize weights to the specified bit width."""
        if bits == 32:
            # No quantization needed, return float32
            return weights, 1.0, 0
        
        # Determine data type based on bits
        if bits == 16:
            dtype = np.float16
            return weights.astype(dtype), 1.0, 0
        
        # For 8-bit and 4-bit, use integer quantization
        data_range = np.max(weights) - np.min(weights)
        if data_range == 0:
            data_range = 1.0
        
        if bits == 8:
            # 8-bit quantization
            zero_point = 0
            scale = data_range / 255.0
            quantized = np.round(weights / scale + zero_point).astype(np.uint8)
            return quantized, scale, zero_point
        
        elif bits == 4:
            # 4-bit quantization (pack two 4-bit values into one byte)
            zero_point = 0
            scale = data_range / 15.0
            quantized_float = np.clip(np.round(weights / scale + zero_point), 0, 15)
            
            # Pack two 4-bit values into each byte
            if weights.size % 2 == 0:
                # Even number of elements
                quantized = np.zeros(weights.size // 2, dtype=np.uint8)
                for i in range(weights.size // 2):
                    quantized[i] = (int(quantized_float[i*2]) << 4) | int(quantized_float[i*2+1])
            else:
                # Odd number of elements, pad with zero
                quantized = np.zeros(weights.size // 2 + 1, dtype=np.uint8)
                for i in range(weights.size // 2):
                    quantized[i] = (int(quantized_float[i*2]) << 4) | int(quantized_float[i*2+1])
                # Handle the last element
                if weights.size > 1:
                    quantized[-1] = int(quantized_float[-1]) << 4
            
            return quantized, scale, zero_point
    
    def _create_tinyai_model(self, model: Any, metadata: Dict[str, Any], processed_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Create a TinyAI model structure from the processed PyTorch model."""
        # Generate layer mapping
        layer_mapping = self._generate_layer_mapping()
        
        # Create TinyAI model dictionary
        tinyai_model = {
            "format_version": "1.0",
            "source": {
                "framework": "pytorch",
                "version": metadata["version"],
                "model_type": metadata["model_type"]
            },
            "metadata": {
                "name": model.__class__.__name__,
                "quantization": {
                    "bits": self.quantize_bits,
                    "mixed_precision": self.mixed_precision
                }
            },
            "layers": [],
            "weights": {}
        }
        
        # Add additional metadata
        if self.prune_threshold is not None:
            tinyai_model["metadata"]["pruning"] = {
                "threshold": self.prune_threshold
            }
        
        if self.weight_clustering is not None:
            tinyai_model["metadata"]["clustering"] = {
                "num_clusters": self.weight_clustering
            }
        
        # Add layer information
        for layer_info in metadata["layers"]:
            # Map PyTorch layer type to TinyAI layer type
            original_type = layer_info["type"]
            mapped_type = layer_mapping.get(original_type, "Unknown")
            
            tinyai_layer = {
                "name": layer_info["name"],
                "type": mapped_type,
                "original_type": original_type,
                "parameters": layer_info.get("parameters", {}),
                "config": layer_info.get("config", {})
            }
            
            tinyai_model["layers"].append(tinyai_layer)
        
        # Add weight data
        tinyai_model["weights"] = processed_weights
        
        return tinyai_model
    
    def _generate_layer_mapping(self) -> Dict[str, str]:
        """Generate mapping from PyTorch layer types to TinyAI layer types."""
        # This is a simplified mapping, a real implementation would be more comprehensive
        return {
            "Linear": "FullyConnected",
            "Conv1d": "Conv1D",
            "Conv2d": "Conv2D",
            "Conv3d": "Conv3D",
            "ConvTranspose1d": "TransposedConv1D",
            "ConvTranspose2d": "TransposedConv2D",
            "ConvTranspose3d": "TransposedConv3D",
            "BatchNorm1d": "BatchNorm1D",
            "BatchNorm2d": "BatchNorm2D",
            "BatchNorm3d": "BatchNorm3D",
            "MaxPool1d": "MaxPool1D",
            "MaxPool2d": "MaxPool2D",
            "MaxPool3d": "MaxPool3D",
            "AvgPool1d": "AvgPool1D",
            "AvgPool2d": "AvgPool2D",
            "AvgPool3d": "AvgPool3D",
            "AdaptiveAvgPool1d": "AdaptiveAvgPool1D",
            "AdaptiveAvgPool2d": "AdaptiveAvgPool2D",
            "AdaptiveAvgPool3d": "AdaptiveAvgPool3D",
            "Dropout": "Dropout",
            "Dropout2d": "Dropout2D",
            "Dropout3d": "Dropout3D",
            "ReLU": "ReLU",
            "LeakyReLU": "LeakyReLU",
            "PReLU": "PReLU",
            "ELU": "ELU",
            "GELU": "GELU",
            "Sigmoid": "Sigmoid",
            "Tanh": "Tanh",
            "Softmax": "Softmax",
            "LogSoftmax": "LogSoftmax",
            "Flatten": "Flatten",
            "LSTM": "LSTM",
            "GRU": "GRU",
            "RNN": "RNN",
            "Embedding": "Embedding",
            "LayerNorm": "LayerNorm",
            "InstanceNorm1d": "InstanceNorm1D",
            "InstanceNorm2d": "InstanceNorm2D",
            "InstanceNorm3d": "InstanceNorm3D",
            "MultiheadAttention": "MultiHeadAttention",
            "Transformer": "Transformer",
            "TransformerEncoder": "TransformerEncoder",
            "TransformerDecoder": "TransformerDecoder",
            "Sequential": "Sequential"
        }
    
    def _save_model(self, model: Dict[str, Any], output_path: str) -> bool:
        """Save the TinyAI model to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Split into metadata and weights
            metadata = {key: value for key, value in model.items() if key != "weights"}
            weights = model["weights"]
            
            # Calculate total weights size for statistics
            total_weights_size = 0
            
            # Convert weights to a binary representation
            weight_data = bytearray()
            weight_offsets = {}
            
            for layer_name, layer_weights in weights.items():
                layer_offsets = {}
                
                for weight_name, weight_info in layer_weights.items():
                    # Store the offset into the weights file
                    layer_offsets[weight_name] = len(weight_data)
                    
                    # Store shape and other metadata
                    shape_bytes = struct.pack(f"{len(weight_info['shape'])}i", *weight_info['shape'])
                    weight_data.extend(shape_bytes)
                    
                    if "scale" in weight_info and "zero_point" in weight_info:
                        # Quantized weight
                        scale_bytes = struct.pack("f", weight_info["scale"])
                        zero_point_bytes = struct.pack("i", weight_info["zero_point"])
                        bits_bytes = struct.pack("i", weight_info["bits"])
                        
                        weight_data.extend(scale_bytes)
                        weight_data.extend(zero_point_bytes)
                        weight_data.extend(bits_bytes)
                    
                    # Store the actual weight data
                    if isinstance(weight_info["data"], np.ndarray):
                        data_bytes = weight_info["data"].tobytes()
                    else:
                        data_bytes = weight_info["data"]
                        
                    weight_data.extend(data_bytes)
                    total_weights_size += len(data_bytes)
                
                weight_offsets[layer_name] = layer_offsets
            
            # Update metadata with weight offsets
            metadata["weight_offsets"] = weight_offsets
            
            # Save metadata as JSON
            metadata_path = os.path.splitext(output_path)[0] + ".json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Save weights as binary file
            weights_path = os.path.splitext(output_path)[0] + ".bin"
            with open(weights_path, "wb") as f:
                f.write(weight_data)
            
            logger.info(f"Saved metadata to {metadata_path}")
            logger.info(f"Saved weights to {weights_path} ({total_weights_size} bytes)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def _gather_statistics(self, original_model: Any, tinyai_model: Dict[str, Any], processed_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Gather statistics about the conversion process."""
        stats = {
            "original_model_type": tinyai_model["source"]["model_type"],
            "framework": tinyai_model["source"]["framework"],
            "framework_version": tinyai_model["source"]["version"],
            "num_layers": len(tinyai_model["layers"]),
            "quantization_bits": tinyai_model["metadata"]["quantization"]["bits"],
            "mixed_precision": tinyai_model["metadata"]["quantization"]["mixed_precision"]
        }
        
        # Calculate parameter counts and memory usage
        original_size = 0
        converted_size = 0
        num_parameters = 0
        
        # Calculate original size
        for _, param in original_model.named_parameters():
            if param.requires_grad:
                size = param.numel()
                num_parameters += size
                original_size += size * 4  # Assuming float32 (4 bytes)
        
        stats["num_parameters"] = num_parameters
        stats["original_size_bytes"] = original_size
        
        # Calculate converted size
        for layer_name, layer_weights in processed_weights.items():
            for weight_name, weight_info in layer_weights.items():
                if isinstance(weight_info["data"], np.ndarray):
                    converted_size += weight_info["data"].nbytes
                else:
                    converted_size += len(weight_info["data"])
        
        stats["converted_size_bytes"] = converted_size
        stats["compression_ratio"] = original_size / converted_size if converted_size > 0 else 0
        
        # Additional statistics for optimizations
        if self.prune_threshold is not None:
            stats["pruning_threshold"] = self.prune_threshold
        
        if self.weight_clustering is not None:
            stats["weight_clustering_clusters"] = self.weight_clustering
        
        return stats
