#!/usr/bin/env python3
"""
Cross-Platform Testing Framework for TinyAI

This script automates testing across multiple platforms:
- Detects the current platform
- Runs platform-specific test scripts
- Collects and analyzes test results
- Generates compatibility reports

Usage:
    python run_tests.py [options]

Options:
    --platform PLATFORM     Specify platform (auto-detect if not provided)
    --tests TESTS           Comma-separated list of tests to run (all if not provided)
    --report FORMAT         Output report format (text, html, json) (default: text)
    --verbose               Show detailed output
    --help                  Show this help message
"""

import os
import sys
import platform
import subprocess
import json
import time
import argparse
from datetime import datetime

# Platform identifiers
PLATFORM_LINUX = "linux"
PLATFORM_MACOS = "macos"
PLATFORM_WINDOWS = "windows"
PLATFORM_RASPBERRY_PI = "raspberry_pi"
PLATFORM_ANDROID = "android"
PLATFORM_IOS = "ios"
PLATFORM_EMBEDDED = "embedded"  # Generic embedded

# Test categories
TEST_CATEGORIES = [
    "core",         # Core functionalities
    "text",         # Text generation models
    "image",        # Image recognition models
    "audio",        # Audio processing models
    "multimodal",   # Multimodal models
    "memory",       # Memory optimization tests
    "performance",  # Performance benchmarks
    "edge_cases"    # Edge case tests (low memory, error handling, etc.)
]

def detect_platform():
    """Detect the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # Detect Raspberry Pi
    if system == "linux" and os.path.exists("/sys/firmware/devicetree/base/model"):
        with open("/sys/firmware/devicetree/base/model", "r") as f:
            model = f.read()
            if "raspberry pi" in model.lower():
                return PLATFORM_RASPBERRY_PI
    
    # Detect standard platforms
    if system == "linux":
        return PLATFORM_LINUX
    elif system == "darwin":
        return PLATFORM_MACOS
    elif system == "windows":
        return PLATFORM_WINDOWS
    elif system == "android":
        return PLATFORM_ANDROID
    
    # Embedded detection could be based on specific environment variables
    # or custom identification files
    if os.environ.get("TINYAI_EMBEDDED") == "1":
        return PLATFORM_EMBEDDED
        
    # Default to the detected system
    return system

def get_platform_specific_script(platform_name):
    """Get the platform-specific test script path."""
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "platforms")
    
    # Map of platform names to script files
    platform_scripts = {
        PLATFORM_LINUX: os.path.join(script_dir, "linux_tests.py"),
        PLATFORM_MACOS: os.path.join(script_dir, "macos_tests.py"),
        PLATFORM_WINDOWS: os.path.join(script_dir, "windows_tests.py"),
        PLATFORM_EMBEDDED: os.path.join(script_dir, "embedded_tests.py")
    }
    
    # Raspberry Pi is handled by the embedded tests script
    if platform_name == PLATFORM_RASPBERRY_PI:
        return platform_scripts[PLATFORM_EMBEDDED]
    
    # Android and iOS platforms are not yet implemented
    if platform_name in [PLATFORM_ANDROID, PLATFORM_IOS]:
        print(f"Warning: Tests for {platform_name} platform are not yet implemented")
        return None
    
    return platform_scripts.get(platform_name)

def find_project_root():
    """Find the root directory of the TinyAI project."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        if os.path.isfile(os.path.join(current_dir, "CMakeLists.txt")) and \
           os.path.isdir(os.path.join(current_dir, "core")) and \
           os.path.isdir(os.path.join(current_dir, "models")):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # If not found, default to the directory containing this script
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_platform_tests(platform_name, test_categories=None, verbose=False):
    """Run tests for the specified platform."""
    platform_script = get_platform_specific_script(platform_name)
    
    if not platform_script or not os.path.isfile(platform_script):
        print(f"Error: No test script found for platform: {platform_name}")
        return False, {}
    
    # Prepare command
    cmd = [sys.executable, platform_script]
    if test_categories:
        cmd.extend(["--tests", ",".join(test_categories)])
    if verbose:
        cmd.append("--verbose")
    
    # Run the platform-specific test script
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=find_project_root()
        )
        
        if result.returncode != 0:
            print(f"Platform tests failed with exit code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False, {}
        
        # Try to parse JSON results
        try:
            test_results = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse test results as JSON")
            print(f"Output: {result.stdout}")
            return True, {"raw_output": result.stdout}
        
        return True, test_results
    
    except Exception as e:
        print(f"Error running platform tests: {e}")
        return False, {}

def generate_report(results, format="text"):
    """Generate a test report in the specified format."""
    report_dir = os.path.join(find_project_root(), "test_reports")
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"cross_platform_test_report_{timestamp}"
    
    if format == "json":
        report_path = os.path.join(report_dir, f"{report_filename}.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
    
    elif format == "html":
        report_path = os.path.join(report_dir, f"{report_filename}.html")
        with open(report_path, "w") as f:
            f.write("<html><head><title>TinyAI Cross-Platform Test Report</title>")
            f.write("<style>body{font-family:Arial,sans-serif;margin:40px;}")
            f.write("table{border-collapse:collapse;width:100%;}")
            f.write("th,td{text-align:left;padding:8px;border:1px solid #ddd;}")
            f.write("th{background-color:#f2f2f2;}")
            f.write(".pass{color:green;}.fail{color:red;}")
            f.write("</style></head><body>")
            f.write(f"<h1>TinyAI Cross-Platform Test Report</h1>")
            f.write(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Summary table
            f.write("<h2>Summary</h2>")
            f.write("<table><tr><th>Platform</th><th>Status</th><th>Tests Passed</th><th>Tests Failed</th></tr>")
            
            total_passed = 0
            total_failed = 0
            
            for platform_name, platform_results in results.items():
                platform_status = platform_results.get("status", "Unknown")
                tests_passed = platform_results.get("tests_passed", 0)
                tests_failed = platform_results.get("tests_failed", 0)
                
                total_passed += tests_passed
                total_failed += tests_failed
                
                status_class = "pass" if platform_status == "pass" else "fail"
                f.write(f"<tr><td>{platform_name}</td>")
                f.write(f"<td class='{status_class}'>{platform_status}</td>")
                f.write(f"<td>{tests_passed}</td><td>{tests_failed}</td></tr>")
            
            f.write("</table>")
            
            # Platform details
            for platform_name, platform_results in results.items():
                f.write(f"<h2>Platform: {platform_name}</h2>")
                
                if "test_results" in platform_results:
                    f.write("<table><tr><th>Test</th><th>Status</th><th>Duration (ms)</th><th>Details</th></tr>")
                    
                    for test_name, test_data in platform_results["test_results"].items():
                        status = test_data.get("status", "Unknown")
                        duration = test_data.get("duration_ms", 0)
                        details = test_data.get("details", "")
                        
                        status_class = "pass" if status == "pass" else "fail"
                        f.write(f"<tr><td>{test_name}</td>")
                        f.write(f"<td class='{status_class}'>{status}</td>")
                        f.write(f"<td>{duration}</td><td>{details}</td></tr>")
                    
                    f.write("</table>")
                else:
                    f.write("<p>No detailed test results available.</p>")
            
            f.write("</body></html>")
    
    else:  # Default to text format
        report_path = os.path.join(report_dir, f"{report_filename}.txt")
        with open(report_path, "w") as f:
            f.write("TinyAI Cross-Platform Test Report\n")
            f.write("===============================\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("Summary:\n")
            f.write("--------\n")
            
            total_passed = 0
            total_failed = 0
            
            for platform_name, platform_results in results.items():
                platform_status = platform_results.get("status", "Unknown")
                tests_passed = platform_results.get("tests_passed", 0)
                tests_failed = platform_results.get("tests_failed", 0)
                
                total_passed += tests_passed
                total_failed += tests_failed
                
                f.write(f"Platform: {platform_name}\n")
                f.write(f"Status: {platform_status}\n")
                f.write(f"Tests Passed: {tests_passed}\n")
                f.write(f"Tests Failed: {tests_failed}\n\n")
            
            f.write(f"Total Tests Passed: {total_passed}\n")
            f.write(f"Total Tests Failed: {total_failed}\n\n")
            
            # Platform details
            f.write("Detailed Results:\n")
            f.write("----------------\n\n")
            
            for platform_name, platform_results in results.items():
                f.write(f"Platform: {platform_name}\n")
                f.write("=" * (len(platform_name) + 10) + "\n\n")
                
                if "test_results" in platform_results:
                    for test_name, test_data in platform_results["test_results"].items():
                        status = test_data.get("status", "Unknown")
                        duration = test_data.get("duration_ms", 0)
                        details = test_data.get("details", "")
                        
                        f.write(f"Test: {test_name}\n")
                        f.write(f"Status: {status}\n")
                        f.write(f"Duration: {duration} ms\n")
                        if details:
                            f.write(f"Details: {details}\n")
                        f.write("\n")
                else:
                    f.write("No detailed test results available.\n\n")
    
    print(f"Report generated: {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Cross-Platform Testing Framework for TinyAI")
    parser.add_argument("--platform", help="Specify platform (auto-detect if not provided)")
    parser.add_argument("--tests", help="Comma-separated list of tests to run (all if not provided)")
    parser.add_argument("--report", default="text", choices=["text", "html", "json"], help="Output report format")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()
    
    # Detect platform if not specified
    platform_name = args.platform or detect_platform()
    print(f"Running tests for platform: {platform_name}")
    
    # Parse test categories
    test_categories = None
    if args.tests:
        test_categories = args.tests.split(",")
        print(f"Running test categories: {', '.join(test_categories)}")
    
    # Run tests
    results = {}
    if platform_name:
        print(f"Starting tests for {platform_name}...")
        success, platform_results = run_platform_tests(platform_name, test_categories, args.verbose)
        
        # Store results
        results[platform_name] = {
            "status": "pass" if success else "fail",
            "test_results": platform_results
        }
        
        # Process results to get pass/fail counts
        tests_passed = 0
        tests_failed = 0
        
        if isinstance(platform_results, dict):
            for test_name, test_data in platform_results.items():
                if isinstance(test_data, dict) and "status" in test_data:
                    if test_data["status"] == "pass":
                        tests_passed += 1
                    else:
                        tests_failed += 1
        
        results[platform_name]["tests_passed"] = tests_passed
        results[platform_name]["tests_failed"] = tests_failed
        
        print(f"Tests completed for {platform_name}")
        print(f"Tests passed: {tests_passed}")
        print(f"Tests failed: {tests_failed}")
    else:
        print("Error: Could not determine platform")
    
    # Generate report
    report_path = generate_report(results, args.report)
    
    # Exit with appropriate code
    if any(platform_data.get("status") == "fail" for platform_data in results.values()):
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
