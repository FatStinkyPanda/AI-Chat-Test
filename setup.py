"""
Setup script for Brain-Inspired Conversational AI
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("=" * 60)
    print("Brain-Inspired Conversational AI - Setup")
    print("=" * 60)
    print()

    requirements_file = "requirements.txt"

    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found!")
        return False

    print("Installing required packages...")
    print("This may take several minutes on first run...")
    print()

    try:
        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_file
        ])

        print()
        print("=" * 60)
        print("Installation complete!")
        print("=" * 60)
        print()
        print("To start the AI:")
        print("  python chat.py")
        print()
        print("For help:")
        print("  python chat.py --help")
        print()

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False


def verify_installation():
    """Verify that key packages are installed"""
    print("Verifying installation...")

    required_packages = [
        'sentence_transformers',
        'chromadb',
        'numpy',
        'sklearn'
    ]

    all_ok = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            all_ok = False

    print()

    return all_ok


def create_directories():
    """Create necessary directories"""
    directories = [
        './brain_memory',
    ]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def main():
    """Main setup function"""
    print()
    print("This script will install all required dependencies.")
    print()

    response = input("Continue with installation? (y/n): ").lower().strip()

    if response != 'y':
        print("Installation cancelled.")
        return

    print()

    # Install requirements
    if not install_requirements():
        print("Installation failed. Please check the error messages above.")
        sys.exit(1)

    # Verify installation
    print()
    if not verify_installation():
        print("Some packages failed to install. Please try manual installation:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Create directories
    print()
    create_directories()

    print()
    print("=" * 60)
    print("Setup complete! You're ready to start.")
    print("=" * 60)
    print()
    print("Run: python chat.py")
    print()


if __name__ == "__main__":
    main()
