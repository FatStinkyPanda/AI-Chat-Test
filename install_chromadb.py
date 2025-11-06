#!/usr/bin/env python3
"""
Separate ChromaDB installer to handle compatibility issues
"""

import subprocess
import sys


def install_chromadb():
    """
    Install ChromaDB with compatibility handling for Python 3.14
    """
    print("Installing ChromaDB with compatibility fixes...")
    print()

    # Try different approaches
    approaches = [
        # Approach 1: Try latest version
        {
            'name': 'Latest ChromaDB',
            'commands': [
                [sys.executable, '-m', 'pip', 'install', 'chromadb', '--no-deps'],
                [sys.executable, '-m', 'pip', 'install', 'httpx', 'hnswlib',
                 'opentelemetry-api', 'opentelemetry-sdk', 'opentelemetry-exporter-otlp-proto-grpc',
                 'opentelemetry-instrumentation-fastapi', 'httptools', 'typer', 'build',
                 'uvloop', 'watchfiles', 'websockets', 'bcrypt']
            ]
        },
        # Approach 2: Specific compatible version
        {
            'name': 'ChromaDB 0.5.5',
            'commands': [
                [sys.executable, '-m', 'pip', 'install', 'chromadb==0.5.5']
            ]
        },
    ]

    for approach in approaches:
        print(f"Trying: {approach['name']}")
        try:
            for cmd in approach['commands']:
                print(f"  Running: {' '.join(cmd)}")
                subprocess.check_call(cmd)

            # Test if it works
            try:
                import chromadb
                print()
                print(f"✓ Success! ChromaDB installed via: {approach['name']}")
                print(f"  Version: {chromadb.__version__}")
                return True
            except ImportError as e:
                print(f"  Installation succeeded but import failed: {e}")
                continue

        except subprocess.CalledProcessError as e:
            print(f"  Failed: {e}")
            print()
            continue

    print()
    print("=" * 60)
    print("ChromaDB installation failed")
    print("=" * 60)
    print()
    print("The system will work in LIMITED MODE without ChromaDB:")
    print("  - Memory will not persist across sessions")
    print("  - Vector search will use in-memory only")
    print()
    print("To enable full functionality later:")
    print("  1. Install Visual Studio Build Tools")
    print("  2. Run: pip install chromadb")
    print()
    return False


if __name__ == "__main__":
    install_chromadb()
