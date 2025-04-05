#!/usr/bin/env python3
"""
TensorFlow to TinyAI Model Converter

This module converts TensorFlow/Keras models to TinyAI format.
It handles:
- Loading TensorFlow/Keras models (.h5, .pb, SavedModel)
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
logger = logging.getLogger('tensorflow_converter')

class TensorFlowConverter:
    """Converter for TensorFlow/Keras models to TinyAI format."""
    
    def __init__(self, 
                 quantize_bits: int = 4, 
                 mixed_precision: bool = False,
                 prune_threshold: Optional[float] = None,
                 weight_clustering: Optional[int] = None,
                 optimize: bool = True,
                 verbose: bool = False):
        """
        Initialize the TensorFlow converter.
        
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
        
        # Check for TensorFlow installation
        try:
            import tensorflow as tf
            self.tf = tf
            logger.debug(f"TensorFlow version: {tf.__version__}")
        except ImportError:
            logger.error("TensorFlow is not installed. Please install TensorFlow to use this converter.")
            sys.exit(1)
    
    def convert(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Convert TensorFlow model to TinyAI format.
        
        Args:
            input_path: Path to input TensorFlow model
            output_path: Path to output TinyAI model file
            
        Returns:
            Dict with conversion statistics and metadata
        """
        # Load the model
        model = self._load_model(input_path)
        
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
        """Load a TensorFlow/Keras model from file."""
        logger.info(f"Loading TensorFlow model from {model_path}")
        
        try:
            # Try different loading methods
            if os.path.isdir(model_path):
                # SavedModel format
                model = self.tf.keras.models.load_model(model_path)
                logger.debug("Loaded model from SavedModel directory")
            elif model_path.endswith('.h5') or model_path.endswith('.keras'):
                # H5 format
                model = self.tf.keras.models.load_model(model_path)
                logger.debug("Loaded model from H5 file")
            elif model_path.endswith('.pb'):
                # Frozen graph
                with self.tf.io.gfile.GFile(model_path, 'rb') as f:
                    graph_def = self.tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    
                with self.tf.compat.v1.Graph().as_default() as graph:
                    self.tf.import_graph_def(graph_def, name='')
                    
                # Create a session and wrap it into a Keras model
                sess = self.tf.compat.v1.Session(graph=graph)
                input_tensors = [graph.get_tensor_by_name(name) for name in graph.get_operations()[0].values()]
                output_tensors = [graph.get_tensor_by_name(name) for name in graph.get_operations()[-1].values()]
                
                model = self.tf.keras.Model(inputs=input_tensors, outputs=output_tensors)
                logger.debug("Loaded model from frozen graph (.pb)")
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
        """Extract metadata from the TensorFlow model."""
        metadata = {
            "framework": "tensorflow",
            "version": self.tf.__version__,
            "model_type": type(model).__name__,
            "input_shapes": [],
            "output_shapes": [],
            "layers": []
        }
        
        # Get input and output shapes
        for layer in model.inputs:
            shape = layer.shape.as_list()
            # Replace None with -1 for dynamic dimensions
            shape = [-1 if dim is None else dim for dim in shape]
            metadata["input_shapes"].append(shape)
        
        for layer in model.outputs:
            shape = layer.shape.as_list()
            shape = [-1 if dim is None else dim for dim in shape]
            metadata["output_shapes"].append(shape)
        
        # Get layer information
        for layer in model.layers:
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "config": {k: str(v) for k, v in layer.get_config().items()},
                "input_shapes": [],
                "output_shapes": []
            }
            
            # Input shapes
            if hasattr(layer, "input_shape"):
                if isinstance(layer.input_shape, list):
                    for shape in layer.input_shape:
                        if shape is not None:
                            shape_list = list(shape)
                            shape_list = [-1 if dim is None else dim for dim in shape_list]
                            layer_info["input_shapes"].append(shape_list)
                else:
                    shape_list = list(layer.input_shape)
                    shape_list = [-1 if dim is None else dim for dim in shape_list]
                    layer_info["input_shapes"].append(shape_list)
            
            # Output shapes
            if hasattr(layer, "output_shape"):
                if isinstance(layer.output_shape, list):
                    for shape in layer.output_shape:
                        if shape is not None:
                            shape_list = list(shape)
                            shape_list = [-1 if dim is None else dim for dim in shape_list]
                            layer_info["output_shapes"].append(shape_list)
                else:
                    shape_list = list(layer.output_shape)
                    shape_list = [-1 if dim is None else dim for dim in shape_list]
                    layer_info["output_shapes"].append(shape_list)
            
            metadata["layers"].append(layer_info)
        
        return metadata

    def _process_weights(self, model: Any) -> Dict[str, Any]:
        """Process model weights with quantization and other optimizations."""
        processed_weights = {}
        
        for layer in model.layers:
            if not layer.weights:
                continue
                
            layer_weights = {}
            
            for weight in layer.weights:
                weight_name = weight.name
                weight_data = weight.numpy()
                
                # Apply pruning if threshold is set
                if self.prune_threshold is not None:
                    weight_data = self._prune_weights(weight_data, self.prune_threshold)
                
                # Apply weight clustering if specified
                if self.weight_clustering is not None:
                    weight_data = self._apply_weight_clustering(weight_data, self.weight_clustering)
                
                # Apply quantization
                if "kernel" in weight_name or "bias" in weight_name:
                    if self.mixed_precision:
                        # For mixed precision, use different bit widths for different layers
                        # Kernels generally benefit from higher precision than biases
                        if "kernel" in weight_name:
                            # Use specified bit width for kernels
                            bits = self.quantize_bits
                        else:
                            # Use 8-bit for biases as a reasonable default
                            bits = 8
                        
                        quantized_data, scale, zero_point = self._quantize_weights(weight_data, bits)
                    else:
                        # Use the same bit width for all weights
                        quantized_data, scale, zero_point = self._quantize_weights(weight_data, self.quantize_bits)
                    
                    layer_weights[weight_name] = {
                        "data": quantized_data,
                        "scale": scale,
                        "zero_point": zero_point,
                        "bits": bits if self.mixed_precision else self.quantize_bits,
                        "shape": weight_data.shape
                    }
                else:
                    # For other weights (not kernels or biases), just store as is
                    layer_weights[weight_name] = {
                        "data": weight_data,
                        "shape": weight_data.shape
                    }
            
            if layer_weights:
                processed_weights[layer.name] = layer_weights
        
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
        """Create a TinyAI model structure from the processed TensorFlow model."""
        # Generate layer mapping
        layer_mapping = self._generate_layer_mapping(model, metadata)
        
        # Create TinyAI model dictionary
        tinyai_model = {
            "format_version": "1.0",
            "source": {
                "framework": "tensorflow",
                "version": metadata["version"],
                "model_type": metadata["model_type"]
            },
            "metadata": {
                "name": model.name if hasattr(model, "name") else "converted_model",
                "input_shapes": metadata["input_shapes"],
                "output_shapes": metadata["output_shapes"],
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
            tinyai_layer = {
                "name": layer_info["name"],
                "type": layer_mapping.get(layer_info["type"], "Unknown"),
                "input_shapes": layer_info["input_shapes"],
                "output_shapes": layer_info["output_shapes"],
                "config": layer_info["config"]
            }
            tinyai_model["layers"].append(tinyai_layer)
        
        # Add weight data
        tinyai_model["weights"] = processed_weights
        
        return tinyai_model
    
    def _generate_layer_mapping(self, model: Any, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Generate mapping from TensorFlow layer types to TinyAI layer types."""
        # This is a simplified mapping, a real implementation would be more comprehensive
        return {
            "Dense": "FullyConnected",
            "Conv2D": "Conv2D",
            "DepthwiseConv2D": "DepthwiseConv2D",
            "SeparableConv2D": "SeparableConv2D",
            "MaxPooling2D": "MaxPool2D",
            "AveragePooling2D": "AvgPool2D",
            "GlobalAveragePooling2D": "GlobalAvgPool2D",
            "BatchNormalization": "BatchNorm",
            "Dropout": "Dropout",
            "Flatten": "Flatten",
            "Reshape": "Reshape",
            "Activation": "Activation",
            "ReLU": "ReLU",
            "LeakyReLU": "LeakyReLU",
            "Softmax": "Softmax",
            "Add": "Add",
            "Concatenate": "Concat",
            "LayerNormalization": "LayerNorm",
            "MultiHeadAttention": "MultiHeadAttention",
            "LSTM": "LSTM",
            "GRU": "GRU",
            "Embedding": "Embedding",
            "InputLayer": "Input"
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
            "input_shapes": tinyai_model["metadata"]["input_shapes"],
            "output_shapes": tinyai_model["metadata"]["output_shapes"],
            "quantization_bits": tinyai_model["metadata"]["quantization"]["bits"],
            "mixed_precision": tinyai_model["metadata"]["quantization"]["mixed_precision"]
        }
        
        # Calculate parameter counts and memory usage
        original_size = 0
        converted_size = 0
        num_parameters = 0
        
        for layer in original_model.layers:
            for weight in layer.weights:
                shape = weight.shape.as_list()
                params = np.prod(shape)
                num_parameters += params
                original_size += params * 4  # Assuming float32 (4 bytes)
        
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
