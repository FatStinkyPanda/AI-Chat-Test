#!/usr/bin/env python3
"""
Automatic Python Version Installer
Installs Python 3.10.x if not available, with user permission
"""

import sys
import os
import platform
import subprocess
import urllib.request
import shutil
from pathlib import Path


REQUIRED_VERSION = "3.10.11"
REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10


class PythonInstaller:
    """Handles Python installation across different platforms"""

    def __init__(self):
        self.system = platform.system().lower()
        self.machine = platform.machine().lower()

    def check_current_version(self):
        """Check the current Python version"""
        version = sys.version_info
        return version.major, version.minor, version.micro

    def is_compatible_version(self):
        """Check if current version is compatible"""
        major, minor, micro = self.check_current_version()
        return major == REQUIRED_MAJOR and minor == REQUIRED_MINOR

    def find_compatible_python(self):
        """Try to find a compatible Python installation"""
        possible_commands = [
            'python3.10',
            'python310',
            'py -3.10',
            f'C:\\Python310\\python.exe',
            f'{os.path.expanduser("~")}\\.pyenv\\versions\\3.10.11\\python.exe',
            '/usr/bin/python3.10',
            '/usr/local/bin/python3.10',
        ]

        for cmd in possible_commands:
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and '3.10' in result.stdout:
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
                continue

        return None

    def ask_permission(self):
        """Ask user for permission to install Python"""
        major, minor, micro = self.check_current_version()

        print("\n" + "=" * 70)
        print("PYTHON VERSION INCOMPATIBILITY DETECTED")
        print("=" * 70)
        print(f"\nCurrent Python: {major}.{minor}.{micro}")
        print(f"Required Python: {REQUIRED_MAJOR}.{REQUIRED_MINOR}.x")
        print()
        print("This project requires Python 3.10.x for full compatibility with:")
        print("  - ChromaDB (vector database)")
        print("  - numpy (scientific computing)")
        print("  - All AI/ML dependencies")
        print()

        # Check if compatible version exists
        compatible_python = self.find_compatible_python()

        if compatible_python:
            print(f"✓ Found compatible Python installation: {compatible_python}")
            print()
            response = input("Use this Python version for the project? (y/n): ").lower().strip()

            if response == 'y':
                return 'use_existing', compatible_python
            else:
                print("\nWould you like to install a fresh Python 3.10.11?")
        else:
            print("✗ No compatible Python 3.10.x installation found.")
            print()
            print("This script can automatically install Python 3.10.11 for you.")

        print()
        print("Installation options:")
        print("  1. Auto-install Python 3.10.11 (recommended)")
        print("  2. Install using pyenv (for developers)")
        print("  3. Manual installation instructions")
        print("  4. Cancel and exit")
        print()

        choice = input("Choose option (1-4): ").strip()

        if choice == '1':
            return 'auto_install', None
        elif choice == '2':
            return 'pyenv_install', None
        elif choice == '3':
            return 'manual_instructions', None
        else:
            return 'cancel', None

    def install_windows(self):
        """Install Python on Windows"""
        print("\nPreparing to install Python 3.10.11 for Windows...")

        # Determine architecture
        if '64' in self.machine or 'amd64' in self.machine:
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
            installer_name = "python-3.10.11-amd64.exe"
        else:
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11.exe"
            installer_name = "python-3.10.11.exe"

        download_path = Path.home() / "Downloads" / installer_name

        print(f"\nDownloading Python installer...")
        print(f"URL: {installer_url}")
        print(f"Destination: {download_path}")
        print()

        try:
            # Download installer
            print("Downloading... (this may take a few minutes)")
            urllib.request.urlretrieve(installer_url, download_path)
            print("✓ Download complete!")

            print()
            print("=" * 70)
            print("IMPORTANT: Installation Instructions")
            print("=" * 70)
            print()
            print("The Python installer will now launch. Please:")
            print("  1. ✓ CHECK 'Add Python 3.10 to PATH' (IMPORTANT!)")
            print("  2. Click 'Install Now'")
            print("  3. Wait for installation to complete")
            print("  4. Click 'Close' when done")
            print()
            input("Press Enter to launch the installer...")

            # Launch installer
            subprocess.run([str(download_path), '/passive', 'PrependPath=1', 'Include_test=0'])

            print()
            print("✓ Python installation initiated!")
            print()
            print("After installation completes:")
            print("  1. Close this window")
            print("  2. Open a NEW terminal/command prompt")
            print("  3. Run: python --version")
            print("  4. Run: python setup.py")
            print()

            return True

        except Exception as e:
            print(f"\n✗ Error downloading installer: {e}")
            return False

    def install_macos(self):
        """Install Python on macOS"""
        print("\nPreparing to install Python 3.10.11 for macOS...")

        # Check if Homebrew is installed
        try:
            subprocess.run(['brew', '--version'], capture_output=True, check=True)
            has_brew = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            has_brew = False

        if has_brew:
            print("\nHomebrew detected. Installing Python 3.10...")
            print()
            print("Running: brew install python@3.10")

            try:
                subprocess.run(['brew', 'install', 'python@3.10'], check=True)
                print("\n✓ Python 3.10 installed via Homebrew!")
                print("\nTo use it, run:")
                print("  python3.10 setup.py")
                return True
            except subprocess.CalledProcessError as e:
                print(f"\n✗ Homebrew installation failed: {e}")
                return False
        else:
            # Download macOS installer
            installer_url = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg"
            installer_name = "python-3.10.11-macos11.pkg"
            download_path = Path.home() / "Downloads" / installer_name

            print(f"\nDownloading Python installer...")

            try:
                urllib.request.urlretrieve(installer_url, download_path)
                print("✓ Download complete!")
                print()
                print("Opening installer... Follow the installation prompts.")

                subprocess.run(['open', str(download_path)])

                print()
                print("After installation completes:")
                print("  1. Close this terminal")
                print("  2. Open a NEW terminal")
                print("  3. Run: python3.10 --version")
                print("  4. Run: python3.10 setup.py")
                return True

            except Exception as e:
                print(f"\n✗ Error: {e}")
                return False

    def install_linux(self):
        """Install Python on Linux"""
        print("\nPreparing to install Python 3.10.11 for Linux...")

        # Detect package manager
        package_managers = {
            'apt-get': ['sudo', 'apt-get', 'update', '&&', 'sudo', 'apt-get', 'install', '-y', 'python3.10', 'python3.10-venv', 'python3-pip'],
            'apt': ['sudo', 'apt', 'update', '&&', 'sudo', 'apt', 'install', '-y', 'python3.10', 'python3.10-venv', 'python3-pip'],
            'yum': ['sudo', 'yum', 'install', '-y', 'python310', 'python310-pip'],
            'dnf': ['sudo', 'dnf', 'install', '-y', 'python3.10', 'python3-pip'],
            'pacman': ['sudo', 'pacman', '-S', '--noconfirm', 'python310'],
        }

        for pm, cmd in package_managers.items():
            if shutil.which(pm):
                print(f"\nDetected package manager: {pm}")
                print(f"Running: {' '.join(cmd)}")
                print()

                try:
                    subprocess.run(' '.join(cmd), shell=True, check=True)
                    print("\n✓ Python 3.10 installed!")
                    print("\nTo use it, run:")
                    print("  python3.10 setup.py")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"\n✗ Installation failed: {e}")
                    return False

        print("\n✗ Could not detect package manager.")
        print("Please install Python 3.10 manually or use pyenv.")
        return False

    def install_with_pyenv(self):
        """Install Python using pyenv"""
        print("\nInstalling Python 3.10.11 using pyenv...")

        # Check if pyenv is installed
        try:
            subprocess.run(['pyenv', '--version'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("\n✗ pyenv is not installed.")
            print("\nTo install pyenv:")

            if self.system == 'windows':
                print("  Download from: https://github.com/pyenv-win/pyenv-win")
            else:
                print("  curl https://pyenv.run | bash")

            return False

        print("\nInstalling Python 3.10.11...")

        try:
            # Install Python version
            subprocess.run(['pyenv', 'install', '3.10.11'], check=True)

            # Set local version
            subprocess.run(['pyenv', 'local', '3.10.11'], check=True)

            print("\n✓ Python 3.10.11 installed and activated via pyenv!")
            print("\nRun setup again:")
            print("  python setup.py")

            return True

        except subprocess.CalledProcessError as e:
            print(f"\n✗ pyenv installation failed: {e}")
            return False

    def show_manual_instructions(self):
        """Show manual installation instructions"""
        print("\n" + "=" * 70)
        print("MANUAL INSTALLATION INSTRUCTIONS")
        print("=" * 70)
        print()

        if self.system == 'windows':
            print("Windows:")
            print("  1. Go to: https://www.python.org/downloads/release/python-31011/")
            print("  2. Download 'Windows installer (64-bit)' or '(32-bit)'")
            print("  3. Run the installer")
            print("  4. ✓ CHECK 'Add Python 3.10 to PATH' (VERY IMPORTANT!)")
            print("  5. Click 'Install Now'")
            print()
        elif self.system == 'darwin':
            print("macOS:")
            print("  Option 1 - Homebrew (recommended):")
            print("    brew install python@3.10")
            print()
            print("  Option 2 - Official installer:")
            print("    1. Go to: https://www.python.org/downloads/release/python-31011/")
            print("    2. Download 'macOS 64-bit universal2 installer'")
            print("    3. Open the .pkg file and follow prompts")
            print()
        else:
            print("Linux:")
            print("  Ubuntu/Debian:")
            print("    sudo apt update")
            print("    sudo apt install python3.10 python3.10-venv python3-pip")
            print()
            print("  Fedora:")
            print("    sudo dnf install python3.10")
            print()
            print("  Arch:")
            print("    sudo pacman -S python310")
            print()

        print("After installation, restart your terminal and run:")
        print("  python --version")
        print("  python setup.py")
        print()

    def install(self):
        """Main installation flow"""
        # Check if already compatible
        if self.is_compatible_version():
            print("✓ Already using compatible Python version!")
            return True

        # Ask permission
        action, python_path = self.ask_permission()

        if action == 'cancel':
            print("\nInstallation cancelled.")
            return False

        elif action == 'use_existing':
            print(f"\n✓ Using existing Python: {python_path}")
            print()
            print("To use this Python version, run:")
            print(f"  {python_path} setup.py")
            print()
            return False  # Return False so main setup doesn't continue

        elif action == 'auto_install':
            print()
            print("=" * 70)
            print("AUTOMATIC INSTALLATION")
            print("=" * 70)
            print()
            print("This will download and install Python 3.10.11 on your system.")
            print()
            confirm = input("Continue with automatic installation? (yes/no): ").lower().strip()

            if confirm != 'yes':
                print("\nInstallation cancelled.")
                return False

            # Platform-specific installation
            if self.system == 'windows':
                return self.install_windows()
            elif self.system == 'darwin':
                return self.install_macos()
            elif self.system == 'linux':
                return self.install_linux()
            else:
                print(f"\n✗ Unsupported platform: {self.system}")
                return False

        elif action == 'pyenv_install':
            return self.install_with_pyenv()

        elif action == 'manual_instructions':
            self.show_manual_instructions()
            return False

        return False


def main():
    """Main entry point"""
    print("=" * 70)
    print("Python Version Installer")
    print("=" * 70)
    print()

    installer = PythonInstaller()

    major, minor, micro = installer.check_current_version()
    print(f"Current Python version: {major}.{minor}.{micro}")
    print(f"Required version: {REQUIRED_MAJOR}.{REQUIRED_MINOR}.x")
    print()

    if installer.is_compatible_version():
        print("✓ Python version is compatible!")
        print("\nYou can proceed with installation:")
        print("  python setup.py")
        sys.exit(0)

    success = installer.install()

    if success:
        print("\n" + "=" * 70)
        print("Installation complete!")
        print("=" * 70)
        print()
        print("IMPORTANT: Restart your terminal/command prompt")
        print("Then run: python setup.py")
        print()
    else:
        print("\nPlease install Python 3.10.x manually and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
