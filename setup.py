"""
Setup script for Brain-Inspired Conversational AI
"""

import subprocess
import sys
import os

# Check Python version first
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10
MAX_MINOR = 11


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info

    print(f"Checking Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != REQUIRED_MAJOR:
        print(f"\n❌ ERROR: Python {REQUIRED_MAJOR}.x is required")
        print(f"   You are using Python {version.major}.{version.minor}")
        print("\n   This project requires Python 3.10.x for full compatibility")
        print("   with ChromaDB, numpy, and other dependencies.\n")
        return False

    if version.minor < REQUIRED_MINOR or version.minor > MAX_MINOR:
        print(f"\n⚠️  WARNING: Python {REQUIRED_MAJOR}.{REQUIRED_MINOR}.x is recommended")
        print(f"   You are using Python {version.major}.{version.minor}.{version.micro}")
        print(f"   Some features may not work correctly")
        print(f"   ChromaDB and numpy may have compatibility issues\n")
        return False

    print(f"✓ Python version compatible ({version.major}.{version.minor}.{version.micro})\n")
    return True


def offer_python_installation():
    """Offer to install correct Python version"""
    print()
    print("Would you like to automatically install Python 3.10.11?")
    print()
    print("Options:")
    print("  1. Yes, install Python 3.10.11 automatically")
    print("  2. No, I'll install it manually")
    print()

    choice = input("Choose (1 or 2): ").strip()

    if choice == '1':
        print()
        print("Launching Python installer...")
        print()

        try:
            # Run the install_python.py script
            subprocess.run([sys.executable, 'install_python.py'], check=True)
            print()
            print("=" * 60)
            print("Python installation process completed!")
            print("=" * 60)
            print()
            print("IMPORTANT: You must restart your terminal/command prompt")
            print("Then run this setup script again:")
            print("  python setup.py")
            print()
            sys.exit(0)

        except subprocess.CalledProcessError as e:
            print(f"\n✗ Installation script failed: {e}")
            return False
        except FileNotFoundError:
            print("\n✗ install_python.py not found!")
            print("Please download it from the repository.")
            return False

    return False


def install_requirements():
    """Install required packages"""
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
    print("=" * 60)
    print("Brain-Inspired Conversational AI - Setup")
    print("=" * 60)
    print()

    # Check Python version first
    if not check_python_version():
        print("\n❌ Installation aborted due to Python version incompatibility")
        print()

        # Offer automatic installation
        if not offer_python_installation():
            print("\nPlease install Python 3.10.x manually:")
            print("  Windows: https://www.python.org/downloads/release/python-31011/")
            print("  macOS/Linux: Run 'python install_python.py' for guided installation")
            print("  Or use pyenv: pyenv install 3.10.11")
            print("\nThen run setup again:")
            print("  python setup.py")
            print()

        sys.exit(1)

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
