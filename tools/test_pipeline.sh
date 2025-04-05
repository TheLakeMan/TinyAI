#!/bin/bash
# TinyAI Pipeline Test Script
# This script demonstrates the full TinyAI pipeline with the sample data

echo "=== TinyAI Pipeline Test ==="
echo

# Step 1: Load the model and tokenizer
echo "Step 1: Load the model and tokenizer..."
cd ..
./build/tinyai model load data/model_config.json data/tiny_vocab.tok

# Step 2: Test tokenizer with sample text
echo
echo "Step 2: Testing tokenizer with sample text..."
./build/tinyai tokenize "TinyAI framework supports both local and remote execution with 4-bit quantization."

# Step 3: Generate text with local-only mode
echo
echo "Step 3: Generating text with local model..."
./build/tinyai config verbose 1
./build/tinyai generate "TinyAI is an ultra-lightweight AI framework that" 30 0.7 top_k

# Step 4: Test MCP connection (this is a simulated connection)
echo
echo "Step 4: Testing MCP connection..."
./build/tinyai mcp connect mock://localhost:8080
./build/tinyai mcp status

# Step 5: Enable hybrid mode and compare with local
echo
echo "Step 5: Testing hybrid generation mode..."
./build/tinyai hybrid on
./build/tinyai hybrid status
./build/tinyai generate "The key features of TinyAI include 4-bit quantization and" 20 0.8 top_p

# Step 6: Force local mode
echo
echo "Step 6: Forcing local mode for comparison..."
./build/tinyai hybrid force-local
./build/tinyai generate "TinyAI's tokenizer supports basic BPE operations and can" 20 0.8 top_p

# Step 7: Force remote mode
echo
echo "Step 7: Forcing remote mode for comparison..."
./build/tinyai hybrid force-remote
./build/tinyai generate "Hybrid execution mode allows TinyAI to balance performance with" 20 0.8 top_p

# Step 8: Disconnect from MCP server
echo
echo "Step 8: Disconnecting from MCP server..."
./build/tinyai mcp disconnect
./build/tinyai mcp status

# Step A: Run comprehensive tests
echo
echo "Step A: Running comprehensive tests..."
cd build
ctest -V

# Done
echo
echo "=== Test Sequence Complete ==="
