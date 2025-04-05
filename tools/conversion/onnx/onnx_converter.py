#!/usr/bin/env python3
"""
ONNX to TinyAI Model Converter

This module converts ONNX models to TinyAI format.
It handles:
- Loading ONNX models (.onnx)
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
logger = logging.getLogger('onnx_converter')

class ONNXConverter:
    """Converter for ONNX models to TinyAI format."""
    
    def __init__(self, 
                 quantize_bits: int = 4, 
                 mixed_precision: bool = False,
                 prune_threshold: Optional[float] = None,
                 weight_clustering: Optional[int] = None,
                 optimize: bool = True,
                 verbose: bool = False):
        """
        Initialize the ONNX converter.
        
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
        
        # Check for ONNX installation
        try:
            import onnx
            self.onnx = onnx
            logger.debug(f"ONNX version: {onnx.__version__}")
            
            # Check for onnxruntime
            try:
                import onnxruntime as ort
                self.ort = ort
                logger.debug(f"ONNX Runtime version: {ort.__version__}")
            except ImportError:
                logger.warning("ONNX Runtime not installed. Some features may be limited.")
                self.ort = None
        except ImportError:
            logger.error("ONNX is not installed. Please install ONNX to use this converter.")
            sys.exit(1)
            
        # Check for scikit-learn (needed for clustering)
        if self.weight_clustering is not None:
            try:
                from sklearn.cluster import KMeans
                logger.debug("scikit-learn is installed.")
            except ImportError:
                logger.error("scikit-learn is not installed but required for weight clustering.")
                logger.error("Please install scikit-learn: pip install scikit-learn")
                sys.exit(1)
    
    def convert(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Convert ONNX model to TinyAI format.
        
        Args:
            input_path: Path to input ONNX model
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
        """Load an ONNX model from file."""
        logger.info(f"Loading ONNX model from {model_path}")
        
        try:
            # Load ONNX model
            model = self.onnx.load(model_path)
            
            # Check model
            self.onnx.checker.check_model(model)
            logger.debug("Model loaded and checked successfully")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from the ONNX model."""
        metadata = {
            "framework": "onnx",
            "version": self.onnx.__version__,
            "model_version": model.ir_version,
            "producer": model.producer_name,
            "inputs": [],
            "outputs": [],
            "nodes": []
        }
        
        # Extract input and output shapes
        for input_info in model.graph.input:
            input_shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    # Dynamic dimension
                    input_shape.append(-1)
                else:
                    input_shape.append(dim.dim_value)
            
            metadata["inputs"].append({
                "name": input_info.name,
                "shape": input_shape,
                "dtype": self._get_tensor_type(input_info.type.tensor_type.elem_type)
            })
        
        for output_info in model.graph.output:
            output_shape = []
            for dim in output_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    # Dynamic dimension
                    output_shape.append(-1)
                else:
                    output_shape.append(dim.dim_value)
            
            metadata["outputs"].append({
                "name": output_info.name,
                "shape": output_shape,
                "dtype": self._get_tensor_type(output_info.type.tensor_type.elem_type)
            })
        
        # Extract node information
        for node in model.graph.node:
            node_info = {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {}
            }
            
            # Extract node attributes
            for attr in node.attribute:
                if attr.type == self.onnx.AttributeProto.FLOAT:
                    node_info["attributes"][attr.name] = attr.f
                elif attr.type == self.onnx.AttributeProto.INT:
                    node_info["attributes"][attr.name] = attr.i
                elif attr.type == self.onnx.AttributeProto.STRING:
                    node_info["attributes"][attr.name] = attr.s.decode('utf-8')
                elif attr.type == self.onnx.AttributeProto.TENSOR:
                    node_info["attributes"][attr.name] = "tensor"  # Just indicate it's a tensor
                elif attr.type == self.onnx.AttributeProto.FLOATS:
                    node_info["attributes"][attr.name] = list(attr.floats)
                elif attr.type == self.onnx.AttributeProto.INTS:
                    node_info["attributes"][attr.name] = list(attr.ints)
                elif attr.type == self.onnx.AttributeProto.STRINGS:
                    node_info["attributes"][attr.name] = [s.decode('utf-8') for s in attr.strings]
            
            metadata["nodes"].append(node_info)
        
        return metadata
    
    def _get_tensor_type(self, elem_type: int) -> str:
        """Convert ONNX tensor type to string representation."""
        type_map = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            8: "string",
            9: "bool",
            10: "float16",
            11: "double",
            12: "uint32",
            13: "uint64",
            14: "complex64",
            15: "complex128"
        }
        return type_map.get(elem_type, f"unknown_{elem_type}")
    
    def _process_weights(self, model: Any) -> Dict[str, Any]:
        """Process model weights with quantization and other optimizations."""
        processed_weights = {}
        
        # Extract all initializers (weights) from the ONNX model
        for initializer in model.graph.initializer:
            # Get weight name
            weight_name = initializer.name
            
            # Extract node name and weight type from the initializer name
            parts = weight_name.rsplit('.', 1) if '.' in weight_name else (weight_name, 'weight')
            if len(parts) == 2:
                node_name, weight_type = parts
            else:
                node_name = "model"
                weight_type = parts[0]
            
            # Convert ONNX tensor to numpy array
            weight_data = self._tensor_to_array(initializer)
            
            # Apply pruning if threshold is set
            if self.prune_threshold is not None:
                weight_data = self._prune_weights(weight_data, self.prune_threshold)
            
            # Apply weight clustering if specified
            if self.weight_clustering is not None:
                weight_data = self._apply_weight_clustering(weight_data, self.weight_clustering)
            
            # Apply quantization based on weight type
            if "weight" in weight_type or "bias" in weight_type:
                if self.mixed_precision:
                    # For mixed precision, use different bit widths for different weight types
                    if "weight" in weight_type:
                        bits = self.quantize_bits
                    else:
                        bits = 8  # Use 8-bit for biases as a reasonable default
                    
                    quantized_data, scale, zero_point = self._quantize_weights(weight_data, bits)
                else:
                    # Use the same bit width for all weights
                    quantized_data, scale, zero_point = self._quantize_weights(weight_data, self.quantize_bits)
                
                # Store weight information
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
            if node_name not in processed_weights:
                processed_weights[node_name] = {}
            
            processed_weights[node_name][weight_type] = weight_info
        
        return processed_weights
    
    def _tensor_to_array(self, tensor: Any) -> np.ndarray:
        """Convert ONNX tensor to numpy array."""
        from onnx import numpy_helper
        return numpy_helper.to_array(tensor)
    
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
        """Create a TinyAI model structure from the processed ONNX model."""
        # Generate operator mapping
        op_mapping = self._generate_op_mapping()
        
        # Create TinyAI model dictionary
        tinyai_model = {
            "format_version": "1.0",
            "source": {
                "framework": "onnx",
                "version": metadata["version"],
                "model_version": metadata["model_version"],
                "producer": metadata["producer"]
            },
            "metadata": {
                "name": os.path.basename(model.graph.name) if model.graph.name else "onnx_model",
                "inputs": metadata["inputs"],
                "outputs": metadata["outputs"],
                "quantization": {
                    "bits": self.quantize_bits,
                    "mixed_precision": self.mixed_precision
                }
            },
            "nodes": [],
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
        
        # Add node information
        for node_info in metadata["nodes"]:
            # Map ONNX operator type to TinyAI operator type
            original_type = node_info["op_type"]
            mapped_type = op_mapping.get(original_type, "Unknown")
            
            tinyai_node = {
                "name": node_info["name"] or f"{original_type}_{len(tinyai_model['nodes'])}",
                "type": mapped_type,
                "original_type": original_type,
                "inputs": node_info["inputs"],
                "outputs": node_info["outputs"],
                "attributes": node_info["attributes"]
            }
            
            tinyai_model["nodes"].append(tinyai_node)
        
        # Add weight data
        tinyai_model["weights"] = processed_weights
        
        return tinyai_model
    
    def _generate_op_mapping(self) -> Dict[str, str]:
        """Generate mapping from ONNX operator types to TinyAI operator types."""
        # This is a simplified mapping, a real implementation would be more comprehensive
        return {
            "Conv": "Conv2D",
            "ConvTranspose": "TransposedConv2D",
            "BatchNormalization": "BatchNorm",
            "MaxPool": "MaxPool2D",
            "AveragePool": "AvgPool2D",
            "GlobalAveragePool": "GlobalAvgPool2D",
            "GlobalMaxPool": "GlobalMaxPool2D",
            "Relu": "ReLU",
            "LeakyRelu": "LeakyReLU",
            "PRelu": "PReLU",
            "Sigmoid": "Sigmoid",
            "Tanh": "Tanh",
            "Softmax": "Softmax",
            "MatMul": "MatMul",
            "Gemm": "FullyConnected",
            "Flatten": "Flatten",
            "Reshape": "Reshape",
            "Transpose": "Transpose",
            "Add": "Add",
            "Sub": "Subtract",
            "Mul": "Multiply",
            "Div": "Divide",
            "Concat": "Concat",
            "Slice": "Slice",
            "Squeeze": "Squeeze",
            "Unsqueeze": "Unsqueeze",
            "Pad": "Pad",
            "Clip": "Clip",
            "LSTM": "LSTM",
            "GRU": "GRU",
            "RNN": "RNN",
            "Dropout": "Dropout",
            "Gather": "Gather",
            "Split": "Split",
            "Resize": "Resize",
            "Upsample": "Upsample",
            "ReduceMean": "ReduceMean",
            "ReduceSum": "ReduceSum",
            "ReduceMax": "ReduceMax",
            "InstanceNormalization": "InstanceNorm"
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
            
            for node_name, node_weights in weights.items():
                node_offsets = {}
                
                for weight_name, weight_info in node_weights.items():
                    # Store the offset into the weights file
                    node_offsets[weight_name] = len(weight_data)
                    
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
                
                weight_offsets[node_name] = node_offsets
            
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
            "source_framework": tinyai_model["source"]["framework"],
            "source_version": tinyai_model["source"]["version"],
            "num_nodes": len(tinyai_model["nodes"]),
            "num_inputs": len(tinyai_model["metadata"]["inputs"]),
            "num_outputs": len(tinyai_model["metadata"]["outputs"]),
            "quantization_bits": tinyai_model["metadata"]["quantization"]["bits"],
            "mixed_precision": tinyai_model["metadata"]["quantization"]["mixed_precision"]
        }
        
        # Calculate parameter counts and memory usage
        original_size = 0
        converted_size = 0
        num_parameters = 0
        
        # Calculate original size and parameter count
        for initializer in original_model.graph.initializer:
            # Convert ONNX tensor to numpy array
            weight_data = self._tensor_to_array(initializer)
            params = weight_data.size
            num_parameters += params
            original_size += params * 4  # Assuming float32 (4 bytes)
        
        stats["num_parameters"] = num_parameters
        stats["original_size_bytes"] = original_size
        
        # Calculate converted size
        for node_name, node_weights in processed_weights.items():
            for weight_name, weight_info in node_weights.items():
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
