@echo off
REM TinyAI Pipeline Test Script
REM This script demonstrates the full TinyAI pipeline with the sample data

echo === TinyAI Pipeline Test ===
echo.

REM Step 1: Load the model and tokenizer
echo Step 1: Load the model and tokenizer...
cd ..
build\Debug\tinyai.exe model load data\model_config.json data\tiny_vocab.tok

REM Step 2: Test tokenizer with sample text
echo.
echo Step 2: Testing tokenizer with sample text...
build\Debug\tinyai.exe tokenize "TinyAI framework supports both local and remote execution with 4-bit quantization."

REM Step 3: Generate text with local-only mode
echo.
echo Step 3: Generating text with local model...
build\Debug\tinyai.exe config verbose 1
build\Debug\tinyai.exe generate "TinyAI is an ultra-lightweight AI framework that" 30 0.7 top_k

REM Step 4: Test MCP connection (this is a simulated connection)
echo.
echo Step 4: Testing MCP connection...
build\Debug\tinyai.exe mcp connect mock://localhost:8080
build\Debug\tinyai.exe mcp status

REM Step 5: Enable hybrid mode and compare with local
echo.
echo Step 5: Testing hybrid generation mode...
build\Debug\tinyai.exe hybrid on
build\Debug\tinyai.exe hybrid status
build\Debug\tinyai.exe generate "The key features of TinyAI include 4-bit quantization and" 20 0.8 top_p

REM Step 6: Force local mode
echo.
echo Step 6: Forcing local mode for comparison...
build\Debug\tinyai.exe hybrid force-local
build\Debug\tinyai.exe generate "TinyAI's tokenizer supports basic BPE operations and can" 20 0.8 top_p

REM Step 7: Force remote mode
echo.
echo Step 7: Forcing remote mode for comparison...
build\Debug\tinyai.exe hybrid force-remote
build\Debug\tinyai.exe generate "Hybrid execution mode allows TinyAI to balance performance with" 20 0.8 top_p

REM Step 8: Disconnect from MCP server
echo.
echo Step 8: Disconnecting from MCP server...
build\Debug\tinyai.exe mcp disconnect
build\Debug\tinyai.exe mcp status

REM Step A: Run comprehensive tests
echo.
echo Step A: Running comprehensive tests...
cd build
ctest -V

REM Done
echo.
echo === Test Sequence Complete ===
