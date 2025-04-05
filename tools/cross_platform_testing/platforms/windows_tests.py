#!/usr/bin/env python3
"""
Windows-specific Testing Script for TinyAI

This script runs TinyAI tests on Windows platforms:
- Sets up the Windows-specific environment
- Runs tests appropriate for Windows
- Collects and formats test results
- Returns results in JSON format for the main test framework

Usage:
    python windows_tests.py [options]

Options:
    --tests TESTS        Comma-separated list of tests to run (all if not provided)
    --verbose            Show detailed output
    --help               Show this help message
"""

import os
import sys
import subprocess
import json
import time
import platform
import argparse
import tempfile
from datetime import datetime

# Test categories for Windows
WINDOWS_TEST_CATEGORIES = [
    "core",         # Core functionalities
    "text",         # Text generation models
    "image",        # Image recognition models
    "audio",        # Audio processing models
    "multimodal",   # Multimodal models
    "memory",       # Memory optimization tests
    "performance",  # Performance benchmarks
    "edge_cases"    # Edge case tests (low memory, error handling, etc.)
]

def find_project_root():
    """Find the root directory of the TinyAI project."""
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        if os.path.isfile(os.path.join(current_dir, "CMakeLists.txt")) and \
           os.path.isdir(os.path.join(current_dir, "core")) and \
           os.path.isdir(os.path.join(current_dir, "models")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # If not found, default to the parent directory of the directory containing this script
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_windows_version():
    """Get detailed Windows version information."""
    version_info = {}
    
    # Get Windows version from platform module
    version_info["system"] = platform.system()
    version_info["release"] = platform.release()
    version_info["version"] = platform.version()
    
    # Get more detailed information from systeminfo command
    try:
        result = subprocess.run(
            ["systeminfo", "/fo", "list"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    version_info[key.strip()] = value.strip()
    except Exception as e:
        pass
    
    return version_info

def check_build_environment():
    """Check if the Windows build environment is properly set up."""
    environment_status = {
        "visual_studio": False,
        "cmake": False,
        "compiler": None,
        "simd_support": {
            "sse2": False,
            "avx": False,
            "avx2": False
        }
    }
    
    # Check for Visual Studio
    try:
        vswhere_result = subprocess.run(
            ["where", "vswhere"], 
            capture_output=True, 
            text=True
        )
        
        if vswhere_result.returncode == 0:
            vs_path_result = subprocess.run(
                ["vswhere", "-latest", "-property", "installationPath"], 
                capture_output=True, 
                text=True
            )
            
            if vs_path_result.returncode == 0 and vs_path_result.stdout.strip():
                environment_status["visual_studio"] = True
                vs_path = vs_path_result.stdout.strip()
                
                # Check for specific Visual Studio version
                vs_version_result = subprocess.run(
                    ["vswhere", "-latest", "-property", "catalog_productDisplayVersion"], 
                    capture_output=True, 
                    text=True
                )
                
                if vs_version_result.returncode == 0 and vs_version_result.stdout.strip():
                    environment_status["compiler"] = f"MSVC {vs_version_result.stdout.strip()}"
    except Exception:
        pass
    
    # Check for CMake
    try:
        cmake_result = subprocess.run(
            ["cmake", "--version"], 
            capture_output=True, 
            text=True
        )
        
        if cmake_result.returncode == 0:
            environment_status["cmake"] = True
    except Exception:
        pass
    
    # Check for SIMD support
    try:
        # This is a simple CPP program to check SIMD support
        simd_check_code = """
        #include <iostream>
        
        int main() {
            bool has_sse2 = false;
            bool has_avx = false;
            bool has_avx2 = false;
            
            #if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
                has_sse2 = true;
            #endif
            
            #if defined(__AVX__)
                has_avx = true;
            #endif
            
            #if defined(__AVX2__)
                has_avx2 = true;
            #endif
            
            std::cout << "SSE2:" << (has_sse2 ? "1" : "0") << std::endl;
            std::cout << "AVX:" << (has_avx ? "1" : "0") << std::endl;
            std::cout << "AVX2:" << (has_avx2 ? "1" : "0") << std::endl;
            
            return 0;
        }
        """
        
        # Create a temporary file for the SIMD check program
        with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as cpp_file:
            cpp_file.write(simd_check_code.encode())
            cpp_file_path = cpp_file.name
        
        # Compile the program
        exe_file_path = cpp_file_path.replace(".cpp", ".exe")
        compile_result = subprocess.run(
            ["cl", "/EHsc", "/nologo", cpp_file_path, "/Fe:" + exe_file_path], 
            capture_output=True, 
            text=True
        )
        
        if compile_result.returncode == 0:
            # Run the program to check SIMD support
            run_result = subprocess.run(
                [exe_file_path], 
                capture_output=True, 
                text=True
            )
            
            if run_result.returncode == 0:
                lines = run_result.stdout.splitlines()
                for line in lines:
                    if line.startswith("SSE2:"):
                        environment_status["simd_support"]["sse2"] = (line[5:] == "1")
                    elif line.startswith("AVX:"):
                        environment_status["simd_support"]["avx"] = (line[4:] == "1")
                    elif line.startswith("AVX2:"):
                        environment_status["simd_support"]["avx2"] = (line[5:] == "1")
        
        # Clean up temporary files
        try:
            os.remove(cpp_file_path)
            os.remove(exe_file_path)
        except Exception:
            pass
    except Exception:
        # If the SIMD check fails, try to determine based on CPU info
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            
            if "flags" in info:
                environment_status["simd_support"]["sse2"] = "sse2" in info["flags"]
                environment_status["simd_support"]["avx"] = "avx" in info["flags"]
                environment_status["simd_support"]["avx2"] = "avx2" in info["flags"]
        except Exception:
            pass
    
    return environment_status

def build_project():
    """Build TinyAI project on Windows."""
    project_root = find_project_root()
    build_dir = os.path.join(project_root, "build")
    
    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)
    
    # Configure with CMake
    configure_result = subprocess.run(
        ["cmake", "-G", "Visual Studio 17 2022", ".."], 
        cwd=build_dir, 
        capture_output=True, 
        text=True
    )
    
    if configure_result.returncode != 0:
        print(f"Failed to configure project: {configure_result.stderr}")
        return False, configure_result.stderr
    
    # Build the project
    build_result = subprocess.run(
        ["cmake", "--build", ".", "--config", "Debug"], 
        cwd=build_dir, 
        capture_output=True, 
        text=True
    )
    
    if build_result.returncode != 0:
        print(f"Failed to build project: {build_result.stderr}")
        return False, build_result.stderr
    
    return True, "Build successful"

def run_tests(test_categories=None, verbose=False):
    """Run TinyAI tests on Windows."""
    project_root = find_project_root()
    build_dir = os.path.join(project_root, "build")
    
    # Check if test_categories is provided and valid
    if test_categories:
        # Filter test categories
        valid_categories = [cat for cat in test_categories if cat in WINDOWS_TEST_CATEGORIES]
    else:
        # Use all categories
        valid_categories = WINDOWS_TEST_CATEGORIES
    
    # Test results
    results = {}
    
    # Run tests for each category
    for category in valid_categories:
        category_results = {}
        
        if category == "core":
            # Run core tests
            core_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "tinyai_tests.exe"), "core"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            category_results["core_tests"] = {
                "status": "pass" if core_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": core_test_result.stdout if verbose else ""
            }
        
        elif category == "text":
            # Run text model tests
            text_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "tinyai_tests.exe"), "text"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            category_results["text_model_tests"] = {
                "status": "pass" if text_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": text_test_result.stdout if verbose else ""
            }
        
        elif category == "image":
            # Run image model tests
            image_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "image_test.exe")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            category_results["image_model_tests"] = {
                "status": "pass" if image_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": image_test_result.stdout if verbose else ""
            }
        
        elif category == "audio":
            # Run audio model tests
            audio_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "audio_test.exe")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            category_results["audio_model_tests"] = {
                "status": "pass" if audio_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": audio_test_result.stdout if verbose else ""
            }
        
        elif category == "multimodal":
            # Run multimodal tests
            multimodal_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "multimodal_test.exe")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            category_results["multimodal_tests"] = {
                "status": "pass" if multimodal_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": multimodal_test_result.stdout if verbose else ""
            }
        
        elif category == "memory":
            # Run memory tests
            memory_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "mmap_loader_test.exe")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            category_results["memory_tests"] = {
                "status": "pass" if memory_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": memory_test_result.stdout if verbose else ""
            }
        
        elif category == "performance":
            # Run performance benchmark
            performance_benchmark_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "simd_benchmark_test.exe")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            category_results["performance_benchmark"] = {
                "status": "pass" if performance_benchmark_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": performance_benchmark_result.stdout if verbose else ""
            }
        
        elif category == "edge_cases":
            # Run edge case tests
            edge_case_results = {}
            
            # Low memory test
            low_memory_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "tinyai_tests.exe"), "memory_constraint"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            edge_case_results["low_memory_test"] = {
                "status": "pass" if low_memory_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": low_memory_test_result.stdout if verbose else ""
            }
            
            # Error handling test
            error_handling_test_result = subprocess.run(
                [os.path.join(build_dir, "Debug", "tinyai_tests.exe"), "error_handling"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            edge_case_results["error_handling_test"] = {
                "status": "pass" if error_handling_test_result.returncode == 0 else "fail",
                "duration_ms": 0,  # Not available in this simple example
                "details": error_handling_test_result.stdout if verbose else ""
            }
            
            category_results.update(edge_case_results)
        
        # Add category results to overall results
        results.update(category_results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Windows-specific Testing Script for TinyAI")
    parser.add_argument("--tests", help="Comma-separated list of tests to run (all if not provided)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    # Parse test categories
    test_categories = None
    if args.tests:
        test_categories = args.tests.split(",")
    
    # Get Windows version information
    windows_version = get_windows_version()
    
    if args.verbose:
        print(f"Running tests on Windows {windows_version.get('release', 'unknown')}")
    
    # Check build environment
    environment = check_build_environment()
    
    if args.verbose:
        print("Build environment:")
        print(f"  Visual Studio: {'Yes' if environment['visual_studio'] else 'No'}")
        print(f"  CMake: {'Yes' if environment['cmake'] else 'No'}")
        print(f"  Compiler: {environment['compiler'] or 'Unknown'}")
        print("  SIMD Support:")
        print(f"    SSE2: {'Yes' if environment['simd_support']['sse2'] else 'No'}")
        print(f"    AVX: {'Yes' if environment['simd_support']['avx'] else 'No'}")
        print(f"    AVX2: {'Yes' if environment['simd_support']['avx2'] else 'No'}")
    
    # Build the project
    build_success, build_message = build_project()
    
    if not build_success:
        # Return error results
        error_results = {
            "build_failed": {
                "status": "fail",
                "duration_ms": 0,
                "details": build_message
            }
        }
        
        print(json.dumps(error_results))
        sys.exit(1)
    
    # Run tests
    test_results = run_tests(test_categories, args.verbose)
    
    # Add environment information to results
    test_results["environment"] = {
        "windows_version": windows_version.get("release", "unknown"),
        "visual_studio": environment["visual_studio"],
        "cmake": environment["cmake"],
        "compiler": environment["compiler"],
        "simd_support": environment["simd_support"]
    }
    
    # Print JSON results
    print(json.dumps(test_results))
    
    # Check if any tests failed
    any_failed = any(
        isinstance(result, dict) and result.get("status") == "fail"
        for result in test_results.values()
        if isinstance(result, dict) and "status" in result
    )
    
    sys.exit(1 if any_failed else 0)

if __name__ == "__main__":
    main()
