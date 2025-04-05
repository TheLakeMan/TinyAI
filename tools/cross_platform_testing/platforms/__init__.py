"""
TinyAI Platform-Specific Testing Modules

This package provides platform-specific testing implementations for the TinyAI framework.
Each module implements the testing logic for a specific platform or device type.

Available platform modules:
- windows_tests.py: Testing for Windows platforms
- linux_tests.py: Testing for Linux platforms
- macos_tests.py: Testing for macOS platforms
- embedded_tests.py: Testing for embedded platforms (Raspberry Pi, Arduino, etc.)
"""

# Import platform modules conditionally to avoid ImportError when missing platform-specific dependencies
import platform
import os

# Detect current platform for conditional imports
system = platform.system().lower()

# Import only the modules for the current platform to avoid unnecessary dependencies
if system == 'windows':
    try:
        from .windows_tests import main as run_windows_tests
        __all__ = ['run_windows_tests']
    except ImportError:
        pass
elif system == 'linux':
    try:
        from .linux_tests import main as run_linux_tests
        __all__ = ['run_linux_tests']
    except ImportError:
        pass
elif system == 'darwin':
    try:
        from .macos_tests import main as run_macos_tests
        __all__ = ['run_macos_tests']
    except ImportError:
        pass

# Check for embedded platforms
if (system == 'linux' and 
    os.path.exists("/sys/firmware/devicetree/base/model") and
    "raspberry pi" in open("/sys/firmware/devicetree/base/model", "r").read().lower()):
    try:
        from .embedded_tests import main as run_embedded_tests
        __all__ = ['run_embedded_tests']
    except ImportError:
        pass
elif os.environ.get("TINYAI_EMBEDDED") == "1":
    try:
        from .embedded_tests import main as run_embedded_tests
        __all__ = ['run_embedded_tests']
    except ImportError:
        pass
