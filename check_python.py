#!/usr/bin/env python3
"""
Check Python version compatibility
"""

import sys

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10
MAX_MINOR = 11

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info

    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    print(f"Required: Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}.x")

    if version.major != REQUIRED_MAJOR:
        print(f"\n❌ ERROR: Python {REQUIRED_MAJOR}.x is required")
        print(f"   You are using Python {version.major}.{version.minor}")
        return False

    if version.minor < REQUIRED_MINOR or version.minor > MAX_MINOR:
        print(f"\n⚠️  WARNING: Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}.x is recommended")
        print(f"   You are using Python {version.major}.{version.minor}.{version.micro}")
        print(f"   Some features may not work correctly\n")

        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            return False

    print(f"✓ Python version compatible\n")
    return True


if __name__ == "__main__":
    if not check_python_version():
        print("\nPlease install Python 3.10.x:")
        print("  Download from: https://www.python.org/downloads/")
        print("  Or use pyenv: pyenv install 3.10.11")
        sys.exit(1)
    else:
        print("You can proceed with installation!")
