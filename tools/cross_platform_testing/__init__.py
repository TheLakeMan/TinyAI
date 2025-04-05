"""
TinyAI Cross-Platform Testing Framework

This package provides utilities for testing the TinyAI framework across multiple platforms:
- Run tests on various operating systems (Windows, Linux, macOS)
- Test on embedded platforms (Raspberry Pi, Arduino, etc.)
- Generate cross-platform compatibility reports
"""

from .run_tests import detect_platform, find_project_root, run_platform_tests, generate_report

__all__ = [
    'detect_platform',
    'find_project_root',
    'run_platform_tests',
    'generate_report',
]

# Version information
__version__ = '1.0.0'
