#!/usr/bin/env python3
"""
Embedded Platform Testing Script for TinyAI

This script runs TinyAI tests on embedded platforms:
- Sets up the embedded-specific environment
- Runs tests appropriate for embedded systems
- Collects and formats test results
- Returns results in JSON format for the main test framework

Supported embedded platforms:
- Raspberry Pi
- Arduino (requires connected device)
- ESP32
- STM32
- Other ARM-based embedded systems

Usage:
    python embedded_tests.py [options]

Options:
    --tests TESTS        Comma-separated list of tests to run (all if not provided)
    --platform PLATFORM  Specific embedded platform (auto-detect if not provided)
    --device PATH        Path to connected device (e.g. /dev/ttyUSB0)
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
import glob
import re
from datetime import datetime

# Test categories for embedded platforms
EMBEDDED_TEST_CATEGORIES = [
    "core",         # Core functionalities (minimal subset)
    "micro_text",   # Text generation for microcontrollers
    "micro_image",  # Lightweight image recognition
    "memory",       # Memory constraint tests
    "power",        # Power consumption tests
    "performance",  # Performance benchmarks
    "edge_cases"    # Edge case tests (extremely low memory, etc.)
]

# Specific embedded platforms
EMBEDDED_PLATFORMS = [
    "raspberry_pi",
    "arduino",
    "esp32",
    "stm32",
    "generic_arm"
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

def detect_embedded_platform():
    """Detect the specific embedded platform."""
    # Check if running on Raspberry Pi
    if os.path.exists("/sys/firmware/devicetree/base/model"):
        with open("/sys/firmware/devicetree/base/model", "r") as f:
            model = f.read()
            if "raspberry pi" in model.lower():
                return "raspberry_pi"
    
    # Check for other ARM-based systems
    machine = platform.machine().lower()
    if "arm" in machine:
        # Generic ARM platform
        return "generic_arm"
    
    # Check for connected Arduino, ESP32, or STM32 devices
    # This is a simplified approach; real implementation would be more robust
    serial_devices = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    if serial_devices:
        # Try to identify the device by communicating with it
        for device in serial_devices:
            try:
                result = subprocess.run(
                    ["stty", "-F", device, "115200"], 
                    capture_output=True, 
                    text=True
                )
                
                # For ESP32, we might look for specific responses
                with open(device, "w") as f:
                    f.write("\r\n")
                
                time.sleep(0.5)
                
                with open(device, "r") as f:
                    response = f.read(100)
                    
                    if "esp" in response.lower():
                        return "esp32"
                    elif "stm32" in response.lower():
                        return "stm32"
                    elif "arduino" in response.lower():
                        return "arduino"
            except Exception:
                pass
    
    # Default to generic ARM if can't specifically identify
    return "generic_arm"

def get_embedded_info(platform_name="auto", device_path=None):
    """Get detailed information about the embedded platform."""
    embedded_info = {}
    
    # Detect platform if not specified
    if platform_name == "auto":
        platform_name = detect_embedded_platform()
    
    embedded_info["platform"] = platform_name
    
    # Get basic system information
    embedded_info["system"] = platform.system()
    embedded_info["release"] = platform.release()
    embedded_info["version"] = platform.version()
    embedded_info["machine"] = platform.machine()
    embedded_info["processor"] = platform.processor()
    
    # Platform-specific information gathering
    if platform_name == "raspberry_pi":
        # Get Raspberry Pi model and hardware details
        try:
            with open("/sys/firmware/devicetree/base/model", "r") as f:
                embedded_info["model"] = f.read().strip('\0')
            
            # Get processor info
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                match = re.search(r"Hardware\s+:\s+(.+)", cpuinfo)
                if match:
                    embedded_info["hardware"] = match.group(1)
                
                match = re.search(r"Revision\s+:\s+(.+)", cpuinfo)
                if match:
                    embedded_info["revision"] = match.group(1)
        except Exception:
            pass
        
        # Get memory information
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                match = re.search(r"MemTotal:\s+(\d+)", meminfo)
                if match:
                    embedded_info["memory_kb"] = int(match.group(1))
        except Exception:
            pass
    
    elif platform_name in ["arduino", "esp32", "stm32"]:
        # For these platforms, we need to communicate with the device
        if device_path and os.path.exists(device_path):
            embedded_info["device_path"] = device_path
            
            # Try to get device information (simplified example)
            try:
                # Set up serial communication
                subprocess.run(
                    ["stty", "-F", device_path, "115200"], 
                    capture_output=True, 
                    text=True
                )
                
                # Send a command to get device info
                with open(device_path, "w") as f:
                    if platform_name == "arduino":
                        f.write("INFO\r\n")
                    elif platform_name == "esp32":
                        f.write("print(ESP.getChipModel())\r\n")
                    elif platform_name == "stm32":
                        f.write("version\r\n")
                
                time.sleep(0.5)
                
                # Read the response
                with open(device_path, "r") as f:
                    response = f.read(200)
                    embedded_info["device_info"] = response.strip()
            except Exception:
                embedded_info["device_info"] = "Could not communicate with device"
    
    elif platform_name == "generic_arm":
        # For generic ARM devices, try to get as much information as possible
        try:
            # Get CPU information
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                
                # Extract relevant information
                matches = {
                    "processor": re.search(r"Processor\s+:\s+(.+)", cpuinfo),
                    "hardware": re.search(r"Hardware\s+:\s+(.+)", cpuinfo),
                    "revision": re.search(r"Revision\s+:\s+(.+)", cpuinfo)
                }
                
                for key, match in matches.items():
                    if match:
                        embedded_info[key] = match.group(1)
        except Exception:
            pass
        
        # Get memory information
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
                match = re.search(r"MemTotal:\s+(\d+)", meminfo)
                if match:
                    embedded_info["memory_kb"] = int(match.group(1))
        except Exception:
            pass
    
    return embedded_info

def check_build_environment(platform_name):
    """Check if the build environment is properly set up for the specified platform."""
    environment_status = {
        "compiler": None,
        "cross_compiler": None,
        "cmake": False,
        "make": False,
        "platform_sdk": False,
        "libraries": {}
    }
    
    # Check for basic tools
    tools_checks = [
        ("cmake", ["cmake", "--version"]),
        ("make", ["make", "--version"])
    ]
    
    for tool_name, cmd in tools_checks:
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                environment_status[tool_name] = True
        except Exception:
            pass
    
    # Platform-specific checks
    if platform_name == "raspberry_pi":
        # Check for ARM compiler
        compiler_checks = [
            ("gcc", ["gcc", "--version"]),
            ("g++", ["g++", "--version"])
        ]
        
        for compiler, cmd in compiler_checks:
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    environment_status["compiler"] = compiler
                    break
            except Exception:
                pass
        
        # Check for specific Raspberry Pi libraries
        libraries = [
            "wiringPi"
        ]
        
        for lib in libraries:
            try:
                result = subprocess.run(
                    ["ldconfig", "-p"], 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0 and lib in result.stdout:
                    environment_status["libraries"][lib] = True
                else:
                    environment_status["libraries"][lib] = False
            except Exception:
                environment_status["libraries"][lib] = False
    
    elif platform_name == "arduino":
        # Check for Arduino CLI or IDE
        for arduino_tool in ["arduino-cli", "arduino"]:
            try:
                result = subprocess.run(
                    [arduino_tool, "--version"], 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    environment_status["platform_sdk"] = True
                    environment_status["arduino_tool"] = arduino_tool
                    break
            except Exception:
                pass
    
    elif platform_name == "esp32":
        # Check for ESP-IDF
        try:
            result = subprocess.run(
                ["idf.py", "--version"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                environment_status["platform_sdk"] = True
                environment_status["esp_idf_version"] = result.stdout.strip()
        except Exception:
            pass
    
    elif platform_name == "stm32":
        # Check for STM32CubeMX or ARM compiler
        try:
            # First check if there's a cross-compiler
            result = subprocess.run(
                ["arm-none-eabi-gcc", "--version"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                environment_status["cross_compiler"] = "arm-none-eabi-gcc"
        except Exception:
            pass
    
    elif platform_name == "generic_arm":
        # Check for general ARM cross-compiler
        try:
            result = subprocess.run(
                ["arm-linux-gnueabihf-gcc", "--version"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                environment_status["cross_compiler"] = "arm-linux-gnueabihf-gcc"
        except Exception:
            try:
                result = subprocess.run(
                    ["arm-none-eabi-gcc", "--version"], 
                    capture_output=True, 
                    text=True
                )
                
                if result.returncode == 0:
                    environment_status["cross_compiler"] = "arm-none-eabi-gcc"
            except Exception:
                pass
    
    return environment_status

def build_project(platform_name, embedded_info, environment):
    """Build TinyAI project for the specified embedded platform."""
    project_root = find_project_root()
    build_dir = os.path.join(project_root, "build_embedded")
    
    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)
    
    # Platform-specific build commands
    if platform_name == "raspberry_pi":
        # For Raspberry Pi, we can use CMake directly
        configure_result = subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-DEMBEDDED=ON", "-DTARGET_PLATFORM=RASPBERRY_PI"], 
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
    
    elif platform_name == "arduino":
        # For Arduino, use the Arduino CLI
        if "arduino_tool" in environment and environment["arduino_tool"]:
            arduino_tool = environment["arduino_tool"]
            
            # Find the Arduino sketch file
            arduino_sketch = os.path.join(project_root, "embedded", "arduino", "TinyAI", "TinyAI.ino")
            
            if not os.path.exists(arduino_sketch):
                return False, f"Arduino sketch not found: {arduino_sketch}"
            
            # Compile the sketch
            build_result = subprocess.run(
                [arduino_tool, "compile", "--fqbn", "arduino:avr:uno", arduino_sketch], 
                capture_output=True, 
                text=True
            )
            
            if build_result.returncode != 0:
                print(f"Failed to build Arduino project: {build_result.stderr}")
                return False, build_result.stderr
        else:
            return False, "Arduino SDK not found"
    
    elif platform_name == "esp32":
        # For ESP32, use the ESP-IDF
        esp_project_path = os.path.join(project_root, "embedded", "esp32")
        
        if not os.path.exists(esp_project_path):
            return False, f"ESP32 project directory not found: {esp_project_path}"
        
        # Build the ESP32 project
        build_result = subprocess.run(
            ["idf.py", "build"], 
            cwd=esp_project_path, 
            capture_output=True, 
            text=True
        )
        
        if build_result.returncode != 0:
            print(f"Failed to build ESP32 project: {build_result.stderr}")
            return False, build_result.stderr
    
    elif platform_name == "stm32":
        # For STM32, check if we're using Makefile or CubeMX
        stm32_project_path = os.path.join(project_root, "embedded", "stm32")
        
        if not os.path.exists(stm32_project_path):
            return False, f"STM32 project directory not found: {stm32_project_path}"
        
        # Check for Makefile
        if os.path.exists(os.path.join(stm32_project_path, "Makefile")):
            # Build using make
            build_result = subprocess.run(
                ["make"], 
                cwd=stm32_project_path, 
                capture_output=True, 
                text=True
            )
            
            if build_result.returncode != 0:
                print(f"Failed to build STM32 project: {build_result.stderr}")
                return False, build_result.stderr
        else:
            return False, "STM32 build system not recognized"
    
    elif platform_name == "generic_arm":
        # For generic ARM, use CMake with appropriate options
        configure_result = subprocess.run(
            [
                "cmake", "..", 
                "-DCMAKE_BUILD_TYPE=Release", 
                "-DEMBEDDED=ON", 
                "-DTARGET_PLATFORM=ARM"
            ], 
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

def run_tests(platform_name, embedded_info, test_categories=None, device_path=None, verbose=False):
    """Run TinyAI tests on the embedded platform."""
    project_root = find_project_root()
    
    # Check if test_categories is provided and valid
    if test_categories:
        # Filter test categories
        valid_categories = [cat for cat in test_categories if cat in EMBEDDED_TEST_CATEGORIES]
    else:
        # Use all categories
        valid_categories = EMBEDDED_TEST_CATEGORIES
    
    # Test results
    results = {}
    
    # Platform-specific test execution
    if platform_name == "raspberry_pi":
        build_dir = os.path.join(project_root, "build_embedded")
        
        # Run tests for each category
        for category in valid_categories:
            if category == "core":
                # Run core tests
                core_test_result = subprocess.run(
                    [os.path.join(build_dir, "tinyai_embedded_tests"), "core"], 
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
                
                results["core_tests"] = {
                    "status": "pass" if core_test_result.returncode == 0 else "fail",
                    "duration_ms": duration_ms,
                    "details": core_test_result.stdout if verbose else ""
                }
            
            elif category == "micro_text":
                # Run microcontroller text model tests
                text_test_result = subprocess.run(
                    [os.path.join(build_dir, "tinyai_embedded_tests"), "micro_text"], 
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
                
                results["micro_text_tests"] = {
                    "status": "pass" if text_test_result.returncode == 0 else "fail",
                    "duration_ms": duration_ms,
                    "details": text_test_result.stdout if verbose else ""
                }
            
            elif category == "micro_image":
                # Run microcontroller image model tests
                image_test_result = subprocess.run(
                    [os.path.join(build_dir, "tinyai_embedded_tests"), "micro_image"], 
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
                
                results["micro_image_tests"] = {
                    "status": "pass" if image_test_result.returncode == 0 else "fail",
                    "duration_ms": duration_ms,
                    "details": image_test_result.stdout if verbose else ""
                }
            
            elif category == "memory":
                # Run memory constraint tests
                memory_test_result = subprocess.run(
                    [os.path.join(build_dir, "tinyai_embedded_tests"), "memory"], 
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
                
                results["memory_tests"] = {
                    "status": "pass" if memory_test_result.returncode == 0 else "fail",
                    "duration_ms": duration_ms,
                    "details": memory_test_result.stdout if verbose else ""
                }
            
            elif category == "performance":
                # Run performance benchmark
                performance_benchmark_result = subprocess.run(
                    [os.path.join(build_dir, "tinyai_embedded_benchmark")], 
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
                
                results["performance_benchmark"] = {
                    "status": "pass" if performance_benchmark_result.returncode == 0 else "fail",
                    "duration_ms": duration_ms,
                    "details": performance_benchmark_result.stdout if verbose else ""
                }
            
            elif category == "power":
                # For power consumption tests, we'd need specialized hardware
                # This is a simplified placeholder
                results["power_tests"] = {
                    "status": "skip",
                    "duration_ms": 0,
                    "details": "Power consumption tests require specialized hardware"
                }
            
            elif category == "edge_cases":
                # Run edge case tests
                edge_case_results = {}
                
                # Low memory test
                low_memory_test_result = subprocess.run(
                    [os.path.join(build_dir, "tinyai_embedded_tests"), "extreme_memory_constraint"], 
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
                
                edge_case_results["extreme_low_memory_test"] = {
                    "status": "pass" if low_memory_test_result.returncode == 0 else "fail",
                    "duration_ms": duration_ms,
                    "details": low_memory_test_result.stdout if verbose else ""
                }
                
                # Error recovery test
                error_recovery_test_result = subprocess.run(
                    [os.path.join(build_dir, "tinyai_embedded_tests"), "error_recovery"], 
                    cwd=project_root, 
                    capture_output=True, 
                    text=True
                )
                
                # Get test duration
                duration_ms = 0
                if "time:" in error_recovery_test_result.stdout:
                    try:
                        time_line = [line for line in error_recovery_test_result.stdout.splitlines() if "time:" in line][0]
                        duration_ms = float(time_line.split("time:")[1].strip().split()[0]) * 1000
                    except Exception:
                        pass
                
                edge_case_results["error_recovery_test"] = {
                    "status": "pass" if error_recovery_test_result.returncode == 0 else "fail",
                    "duration_ms": duration_ms,
                    "details": error_recovery_test_result.stdout if verbose else ""
                }
                
                results.update(edge_case_results)
    
    elif platform_name in ["arduino", "esp32", "stm32"]:
        # For these platforms, we need to upload the test firmware and communicate with the device
        if not device_path or not os.path.exists(device_path):
            results["device_error"] = {
                "status": "fail",
                "duration_ms": 0,
                "details": f"Device not found at {device_path}. Please specify correct device path."
            }
            return results
        
        # Check platform-specific test capabilities
        if platform_name == "arduino":
            # Arduino tests are more limited
            for category in valid_categories:
                if category in ["core", "micro_text", "memory"]:
                    # These tests can run on Arduino
                    # This is a placeholder for actual implementation that would
                    # upload test firmware and read results via serial
                    results[f"{category}_tests"] = {
                        "status": "skip",
                        "duration_ms": 0,
                        "details": f"{category} tests on Arduino not implemented in this version"
                    }
                else:
                    # Skip other tests
                    results[f"{category}_tests"] = {
                        "status": "skip",
                        "duration_ms": 0,
                        "details": f"{category} tests not supported on Arduino"
                    }
        
        elif platform_name == "esp32":
            # ESP32 has more capabilities
            for category in valid_categories:
                if category in ["core", "micro_text", "micro_image", "memory", "performance"]:
                    # These tests can run on ESP32
                    # This is a placeholder for actual implementation
                    results[f"{category}_tests"] = {
                        "status": "skip",
                        "duration_ms": 0,
                        "details": f"{category} tests on ESP32 not implemented in this version"
                    }
                else:
                    # Skip other tests
                    results[f"{category}_tests"] = {
                        "status": "skip",
                        "duration_ms": 0,
                        "details": f"{category} tests not supported on ESP32"
                    }
        
        elif platform_name == "stm32":
            # STM32 tests
            for category in valid_categories:
                if category in ["core", "micro_text", "memory"]:
                    # These tests can run on STM32
                    # This is a placeholder for actual implementation
                    results[f"{category}_tests"] = {
                        "status": "skip",
                        "duration_ms": 0,
                        "details": f"{category} tests on STM32 not implemented in this version"
                    }
                else:
                    # Skip other tests
                    results[f"{category}_tests"] = {
                        "status": "skip",
                        "duration_ms": 0,
                        "details": f"{category} tests not supported on STM32"
                    }
    
    elif platform_name == "generic_arm":
        # For generic ARM, similar to Raspberry Pi but more limited
        build_dir = os.path.join(project_root, "build_embedded")
        
        # Run tests for each category
        for category in valid_categories:
            if category in ["core", "micro_text", "memory", "performance"]:
                # Run the test if available
                test_binary = os.path.join(build_dir, f"tinyai_embedded_{category}_test")
                
                if os.path.exists(test_binary):
                    test_result = subprocess.run(
                        [test_binary], 
                        cwd=project_root, 
                        capture_output=True, 
                        text=True
                    )
                    
                    results[f"{category}_tests"] = {
                        "status": "pass" if test_result.returncode == 0 else "fail",
                        "duration_ms": 0,  # Duration not available
                        "details": test_result.stdout if verbose else ""
                    }
                else:
                    # Skip if test binary not available
                    results[f"{category}_tests"] = {
                        "status": "skip",
                        "duration_ms": 0,
                        "details": f"Test binary not found: {test_binary}"
                    }
            else:
                # Skip unsupported categories
                results[f"{category}_tests"] = {
                    "status": "skip",
                    "duration_ms": 0,
                    "details": f"{category} tests not supported on generic ARM"
                }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Embedded Platform Testing Script for TinyAI")
    parser.add_argument("--tests", help="Comma-separated list of tests to run (all if not provided)")
    parser.add_argument("--platform", choices=EMBEDDED_PLATFORMS + ["auto"], default="auto", 
                        help="Specific embedded platform (auto-detect if not provided)")
    parser.add_argument("--device", help="Path to connected device (e.g. /dev/ttyUSB0)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    # Parse test categories
    test_categories = None
    if args.tests:
        test_categories = args.tests.split(",")
    
    # Detect or use specified platform
    platform_name = args.platform
    if platform_name == "auto":
        platform_name = detect_embedded_platform()
    
    if platform_name not in EMBEDDED_PLATFORMS:
        print(f"Error: Unsupported or undetected embedded platform: {platform_name}")
        sys.exit(1)
    
    # Get embedded platform information
    embedded_info = get_embedded_info(platform_name, args.device)
    
    if args.verbose:
        print(f"Running tests on {platform_name} platform")
        for key, value in embedded_info.items():
            print(f"  {key}: {value}")
    
    # Check build environment
    environment = check_build_environment(platform_name)
    
    if args.verbose:
        print("Build environment:")
        print(f"  Compiler: {environment['compiler'] or environment['cross_compiler'] or 'Not found'}")
        print(f"  CMake: {'Yes' if environment['cmake'] else 'No'}")
        print(f"  Make: {'Yes' if environment['make'] else 'No'}")
        print(f"  Platform SDK: {'Yes' if environment['platform_sdk'] else 'No'}")
        if environment['libraries']:
            print("  Libraries:")
            for lib, available in environment['libraries'].items():
                print(f"    {lib}: {'Yes' if available else 'No'}")
    
    # Build the project
    build_success, build_message = build_project(platform_name, embedded_info, environment)
    
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
    test_results = run_tests(platform_name, embedded_info, test_categories, args.device, args.verbose)
    
    # Add environment information to results
    test_results["environment"] = {
        "platform": platform_name,
        "embedded_info": embedded_info,
        "compiler": environment.get("compiler") or environment.get("cross_compiler"),
        "platform_sdk": environment["platform_sdk"]
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
