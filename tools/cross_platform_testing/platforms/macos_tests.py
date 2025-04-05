#!/usr/bin/env python3
"""
macOS-specific Testing Script for TinyAI

This script runs TinyAI tests on macOS platforms:
- Sets up the macOS-specific environment
- Runs tests appropriate for macOS
- Collects and formats test results
- Returns results in JSON format for the main test framework

Usage:
    python macos_tests.py [options]

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

# Test categories for macOS
MACOS_TEST_CATEGORIES = [
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

def get_macos_info():
    """Get detailed macOS information."""
    macos_info = {}
    
    # Get basic system information
    macos_info["system"] = platform.system()
    macos_info["node"] = platform.node()
    macos_info["release"] = platform.release()
    macos_info["version"] = platform.version()
    macos_info["machine"] = platform.machine()
    macos_info["processor"] = platform.processor()
    
    # Get more detailed macOS information using sw_vers
    try:
        product_name = subprocess.run(
            ["sw_vers", "-productName"], 
            capture_output=True, 
            text=True
        )
        
        product_version = subprocess.run(
            ["sw_vers", "-productVersion"], 
            capture_output=True, 
            text=True
        )
        
        build_version = subprocess.run(
            ["sw_vers", "-buildVersion"], 
            capture_output=True, 
            text=True
        )
        
        if product_name.returncode == 0:
            macos_info["product_name"] = product_name.stdout.strip()
        
        if product_version.returncode == 0:
            macos_info["product_version"] = product_version.stdout.strip()
        
        if build_version.returncode == 0:
            macos_info["build_version"] = build_version.stdout.strip()
    except Exception:
        pass
    
    # Get architecture information
    try:
        arch_result = subprocess.run(
            ["uname", "-m"], 
            capture_output=True, 
            text=True
        )
        
        if arch_result.returncode == 0:
            macos_info["architecture"] = arch_result.stdout.strip()
    except Exception:
        pass
    
    # Check if running on Apple Silicon or Intel
    macos_info["is_apple_silicon"] = macos_info.get("architecture") == "arm64"
    
    return macos_info

def check_build_environment():
    """Check if the macOS build environment is properly set up."""
    environment_status = {
        "compiler": None,
        "cmake": False,
        "make": False,
        "xcode": False,
        "simd_support": {
            "sse2": False,
            "avx": False,
            "avx2": False,
            "neon": False
        },
        "libraries": {
            "openmp": False,
            "pthread": False
        },
        "accelerate_framework": False
    }
    
    # Check for compilers
    compiler_checks = [
        ("clang", "--version"),
        ("gcc", "--version"),
        ("g++", "--version")
    ]
    
    for compiler, arg in compiler_checks:
        try:
            result = subprocess.run(
                [compiler, arg], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                environment_status["compiler"] = compiler
                break
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
    
    # Check for Make
    try:
        make_result = subprocess.run(
            ["make", "--version"], 
            capture_output=True, 
            text=True
        )
        
        if make_result.returncode == 0:
            environment_status["make"] = True
    except Exception:
        pass
    
    # Check for Xcode
    try:
        xcode_result = subprocess.run(
            ["xcodebuild", "-version"], 
            capture_output=True, 
            text=True
        )
        
        if xcode_result.returncode == 0:
            environment_status["xcode"] = True
    except Exception:
        pass
    
    # Check for Accelerate framework
    try:
        # A simple approach to check if Accelerate framework is available
        accelerate_check_code = """
        #include <stdio.h>
        #include <Accelerate/Accelerate.h>
        
        int main() {
            printf("Accelerate:1\\n");
            return 0;
        }
        """
        
        # Create a temporary file for the Accelerate check
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as c_file:
            c_file.write(accelerate_check_code.encode())
            c_file_path = c_file.name
        
        # Compile the program
        compiler = environment_status["compiler"] or "clang"
        executable_path = c_file_path + ".out"
        compile_result = subprocess.run(
            [compiler, c_file_path, "-o", executable_path, "-framework", "Accelerate"], 
            capture_output=True, 
            text=True
        )
        
        if compile_result.returncode == 0:
            # Run the program
            run_result = subprocess.run(
                [executable_path], 
                capture_output=True, 
                text=True
            )
            
            if run_result.returncode == 0:
                for line in run_result.stdout.splitlines():
                    if line.strip() == "Accelerate:1":
                        environment_status["accelerate_framework"] = True
                        break
        
        # Clean up temporary files
        try:
            os.remove(c_file_path)
            os.remove(executable_path)
        except Exception:
            pass
    except Exception:
        pass
    
    # Check for SIMD support
    macos_info = get_macos_info()
    is_apple_silicon = macos_info.get("is_apple_silicon", False)
    
    if is_apple_silicon:
        # Apple Silicon always has NEON
        environment_status["simd_support"]["neon"] = True
        
        # Apple Silicon doesn't have AVX/SSE2
        environment_status["simd_support"]["sse2"] = False
        environment_status["simd_support"]["avx"] = False
        environment_status["simd_support"]["avx2"] = False
    else:
        # Intel Mac
        # Assume SSE2 is always available on Intel Macs
        environment_status["simd_support"]["sse2"] = True
        
        # Check for AVX/AVX2 on Intel
        try:
            # Create a simple C program to detect SIMD support
            simd_check_code = """
            #include <stdio.h>
            
            int main() {
                #if defined(__SSE2__)
                    printf("SSE2:1\\n");
                #else
                    printf("SSE2:0\\n");
                #endif
                
                #if defined(__AVX__)
                    printf("AVX:1\\n");
                #else
                    printf("AVX:0\\n");
                #endif
                
                #if defined(__AVX2__)
                    printf("AVX2:1\\n");
                #else
                    printf("AVX2:0\\n");
                #endif
                
                return 0;
            }
            """
            
            # Create a temporary file for the SIMD check program
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as c_file:
                c_file.write(simd_check_code.encode())
                c_file_path = c_file.name
            
            # Get the compiler (should be clang on macOS)
            compiler = environment_status["compiler"] or "clang"
            
            # Compile the program with SIMD flags
            executable_path = c_file_path + ".out"
            compile_result = subprocess.run(
                [compiler, "-msse2", "-mavx", "-mavx2", c_file_path, "-o", executable_path], 
                capture_output=True, 
                text=True
            )
            
            if compile_result.returncode == 0:
                # Run the program to check SIMD support
                run_result = subprocess.run(
                    [executable_path], 
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
                os.remove(c_file_path)
                os.remove(executable_path)
            except Exception:
                pass
        except Exception:
            # Default to basic Intel Mac support
            environment_status["simd_support"]["sse2"] = True
            environment_status["simd_support"]["avx"] = False
            environment_status["simd_support"]["avx2"] = False
    
    # Check for required libraries
    library_checks = [
        ("openmp", [compiler, "-fopenmp", "-dM", "-E", "-"]),
        ("pthread", [compiler, "-pthread", "-dM", "-E", "-"])
    ]
    
    for lib_name, cmd in library_checks:
        try:
            result = subprocess.run(
                cmd, 
                input="", 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                environment_status["libraries"][lib_name] = True
        except Exception:
            pass
    
    return environment_status

def build_project():
    """Build TinyAI project on macOS."""
    project_root = find_project_root()
    build_dir = os.path.join(project_root, "build")
    
    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)
    
    # Configure with CMake
    configure_result = subprocess.run(
        ["cmake", ".."], 
        cwd=build_dir, 
        capture_output=True, 
        text=True
    )
    
    if configure_result.returncode != 0:
        print(f"Failed to configure project: {configure_result.stderr}")
        return False, configure_result.stderr
    
    # Build the project with make
    build_result = subprocess.run(
        ["make", "-j4"], 
        cwd=build_dir, 
        capture_output=True, 
        text=True
    )
    
    if build_result.returncode != 0:
        print(f"Failed to build project: {build_result.stderr}")
        return False, build_result.stderr
    
    return True, "Build successful"

def run_tests(test_categories=None, verbose=False):
    """Run TinyAI tests on macOS."""
    project_root = find_project_root()
    build_dir = os.path.join(project_root, "build")
    
    # Check if test_categories is provided and valid
    if test_categories:
        # Filter test categories
        valid_categories = [cat for cat in test_categories if cat in MACOS_TEST_CATEGORIES]
    else:
        # Use all categories
        valid_categories = MACOS_TEST_CATEGORIES
    
    # Test results
    results = {}
    
    # Run tests for each category
    for category in valid_categories:
        category_results = {}
        
        if category == "core":
            # Run core tests
            core_test_result = subprocess.run(
                [os.path.join(build_dir, "tinyai_tests"), "core"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration from the output (assumes test outputs time taken)
            duration_ms = 0
            if "time:" in core_test_result.stdout:
                try:
                    time_line = [line for line in core_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            category_results["core_tests"] = {
                "status": "pass" if core_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": core_test_result.stdout if verbose else ""
            }
        
        elif category == "text":
            # Run text model tests
            text_test_result = subprocess.run(
                [os.path.join(build_dir, "tinyai_tests"), "text"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration
            duration_ms = 0
            if "time:" in text_test_result.stdout:
                try:
                    time_line = [line for line in text_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            category_results["text_model_tests"] = {
                "status": "pass" if text_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": text_test_result.stdout if verbose else ""
            }
        
        elif category == "image":
            # Run image model tests
            image_test_result = subprocess.run(
                [os.path.join(build_dir, "image_test")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration
            duration_ms = 0
            if "time:" in image_test_result.stdout:
                try:
                    time_line = [line for line in image_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            category_results["image_model_tests"] = {
                "status": "pass" if image_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": image_test_result.stdout if verbose else ""
            }
        
        elif category == "audio":
            # Run audio model tests
            audio_test_result = subprocess.run(
                [os.path.join(build_dir, "test_audio_model")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration
            duration_ms = 0
            if "time:" in audio_test_result.stdout:
                try:
                    time_line = [line for line in audio_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            category_results["audio_model_tests"] = {
                "status": "pass" if audio_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": audio_test_result.stdout if verbose else ""
            }
        
        elif category == "multimodal":
            # Run multimodal tests
            multimodal_test_result = subprocess.run(
                [os.path.join(build_dir, "test_multimodal")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration
            duration_ms = 0
            if "time:" in multimodal_test_result.stdout:
                try:
                    time_line = [line for line in multimodal_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            category_results["multimodal_tests"] = {
                "status": "pass" if multimodal_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": multimodal_test_result.stdout if verbose else ""
            }
        
        elif category == "memory":
            # Run memory tests
            memory_test_result = subprocess.run(
                [os.path.join(build_dir, "test_mmap_loader")], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration
            duration_ms = 0
            if "time:" in memory_test_result.stdout:
                try:
                    time_line = [line for line in memory_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            category_results["memory_tests"] = {
                "status": "pass" if memory_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": memory_test_result.stdout if verbose else ""
            }
        
        elif category == "performance":
            # Run performance benchmark
            # Check if running on Apple Silicon or Intel
            macos_info = get_macos_info()
            is_apple_silicon = macos_info.get("is_apple_silicon", False)
            
            # Use the appropriate test binary based on architecture
            if is_apple_silicon:
                test_binary = os.path.join(build_dir, "simd_benchmark_test_neon")
            else:
                test_binary = os.path.join(build_dir, "simd_benchmark_test")
                
            performance_benchmark_result = subprocess.run(
                [test_binary], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get benchmark duration (total time)
            duration_ms = 0
            if "Total time:" in performance_benchmark_result.stdout:
                try:
                    time_line = [line for line in performance_benchmark_result.stdout.splitlines() if "Total time:" in line][0]
                    duration_ms = float(time_line.split("Total time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            category_results["performance_benchmark"] = {
                "status": "pass" if performance_benchmark_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": performance_benchmark_result.stdout if verbose else ""
            }
        
        elif category == "edge_cases":
            # Run edge case tests
            edge_case_results = {}
            
            # Low memory test
            low_memory_test_result = subprocess.run(
                [os.path.join(build_dir, "tinyai_tests"), "memory_constraint"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration
            duration_ms = 0
            if "time:" in low_memory_test_result.stdout:
                try:
                    time_line = [line for line in low_memory_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            edge_case_results["low_memory_test"] = {
                "status": "pass" if low_memory_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": low_memory_test_result.stdout if verbose else ""
            }
            
            # Error handling test
            error_handling_test_result = subprocess.run(
                [os.path.join(build_dir, "tinyai_tests"), "error_handling"], 
                cwd=project_root, 
                capture_output=True, 
                text=True
            )
            
            # Get test duration
            duration_ms = 0
            if "time:" in error_handling_test_result.stdout:
                try:
                    time_line = [line for line in error_handling_test_result.stdout.splitlines() if "time:" in line][0]
                    duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                except Exception:
                    pass
            
            edge_case_results["error_handling_test"] = {
                "status": "pass" if error_handling_test_result.returncode == 0 else "fail",
                "duration_ms": duration_ms,
                "details": error_handling_test_result.stdout if verbose else ""
            }
            
            category_results.update(edge_case_results)
        
        # Add category results to overall results
        results.update(category_results)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="macOS-specific Testing Script for TinyAI")
    parser.add_argument("--tests", help="Comma-separated list of tests to run (all if not provided)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    # Parse test categories
    test_categories = None
    if args.tests:
        test_categories = args.tests.split(",")
    
    # Get macOS system information
    macos_info = get_macos_info()
    
    if args.verbose:
        print(f"Running tests on macOS {macos_info.get('product_version', 'unknown')}")
        print(f"Architecture: {macos_info.get('architecture', 'unknown')}")
    
    # Check build environment
    environment = check_build_environment()
    
    if args.verbose:
        print("Build environment:")
        print(f"  Compiler: {environment['compiler'] or 'Not found'}")
        print(f"  CMake: {'Yes' if environment['cmake'] else 'No'}")
        print(f"  Make: {'Yes' if environment['make'] else 'No'}")
        print(f"  Xcode: {'Yes' if environment['xcode'] else 'No'}")
        print("  SIMD Support:")
        if macos_info.get("is_apple_silicon", False):
            print(f"    NEON: {'Yes' if environment['simd_support']['neon'] else 'No'}")
        else:
            print(f"    SSE2: {'Yes' if environment['simd_support']['sse2'] else 'No'}")
            print(f"    AVX: {'Yes' if environment['simd_support']['avx'] else 'No'}")
            print(f"    AVX2: {'Yes' if environment['simd_support']['avx2'] else 'No'}")
        print("  Libraries:")
        print(f"    OpenMP: {'Yes' if environment['libraries']['openmp'] else 'No'}")
        print(f"    Pthread: {'Yes' if environment['libraries']['pthread'] else 'No'}")
        print(f"  Accelerate Framework: {'Yes' if environment['accelerate_framework'] else 'No'}")
    
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
        "os": f"macOS {macos_info.get('product_version', 'unknown')}",
        "architecture": macos_info.get("architecture", "unknown"),
        "is_apple_silicon": macos_info.get("is_apple_silicon", False),
        "compiler": environment["compiler"],
        "cmake": environment["cmake"],
        "make": environment["make"],
        "xcode": environment["xcode"],
        "simd_support": environment["simd_support"],
        "libraries": environment["libraries"],
        "accelerate_framework": environment["accelerate_framework"]
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
