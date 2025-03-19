#!/usr/bin/env python3
"""
Quantum AI Assistant Download Script
This script downloads the Quantum AI Assistant package suitable for your platform
and sets it up for easy use.
"""

import os
import sys
import platform
import subprocess
import zipfile
import tempfile
import shutil
import argparse
from urllib.request import urlretrieve

# The URL to the release pages where packages are stored
# Primary GitHub releases URL
GITHUB_RELEASE_BASE_URL = "https://github.com/quantum-ai-assistant/releases/download/v1.0.0"
# Backup URL in case GitHub is inaccessible
BACKUP_RELEASE_BASE_URL = "https://quantum-ai-assistant.com/downloads/v1.0.0"

def get_platform_info():
    """Get information about the current platform"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        platform_name = "windows"
        if "amd64" in machine or "x86_64" in machine:
            arch = "x64"
        elif "arm64" in machine:
            arch = "arm64"
        else:
            arch = "x86"
        extension = ".exe"
    elif system == "darwin":
        platform_name = "macos"
        if "arm64" in machine:
            arch = "arm64"
        else:
            arch = "x64"
        extension = ".dmg"
    else:  # Linux and other Unix-like
        platform_name = "linux"
        if "amd64" in machine or "x86_64" in machine:
            arch = "x64"
        elif "arm64" in machine or "aarch64" in machine:
            arch = "arm64"
        else:
            arch = "x86"
        extension = ".AppImage"
    
    return platform_name, arch, extension

def download_package(platform_name, arch, extension, destination, use_python_package=False):
    """Download the appropriate package for the platform"""
    # Determine the URL based on platform information
    if use_python_package:
        filename = "quantum_ai_assistant-1.0.0-py3-none-any.whl"
    else:
        filename = f"QuantumAIAssistant-{platform_name}-{arch}{extension}"
    
    # Create destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)
    
    local_path = os.path.join(destination, filename)
    
    print(f"Downloading Quantum AI Assistant for {platform_name} {arch}...")
    
    # Try GitHub URL first
    primary_url = f"{GITHUB_RELEASE_BASE_URL}/{filename}"
    backup_url = f"{BACKUP_RELEASE_BASE_URL}/{filename}"
    
    print(f"Attempting download from: {primary_url}")
    
    # Progress indicator
    def progress_callback(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        filled = int(percent / 2)
        bar = "█" * filled + "-" * (50 - filled)
        sys.stdout.write(f"\rProgress: |{bar}| {percent}%")
        sys.stdout.flush()
    
    # Try primary URL
    download_success = False
    
    try:
        urlretrieve(primary_url, local_path, progress_callback)
        print(f"\nDownloaded successfully to {local_path}")
        download_success = True
    except Exception as e:
        print(f"\nPrimary download failed: {e}")
        print(f"Attempting fallback download from: {backup_url}")
        
        # Try backup URL
        try:
            urlretrieve(backup_url, local_path, progress_callback)
            print(f"\nDownloaded successfully to {local_path} (from backup source)")
            download_success = True
        except Exception as backup_e:
            print(f"\nBackup download failed: {backup_e}")
    
    if download_success:
        return local_path
    else:
        print("Could not download the package from any source.")
        print("Please check your internet connection and try again.")
        print("If the problem persists, please contact support.")
        sys.exit(1)

def setup_package(package_path, platform_name, extension, install_dir):
    """Set up the downloaded package"""
    if platform_name == "windows" and extension == ".exe":
        # For Windows executables, just create a shortcut
        print(f"Installation complete. You can run the application by double-clicking {package_path}")
        
    elif platform_name == "macos" and extension == ".dmg":
        # For macOS DMG files
        print(f"Installation complete. Please open {package_path} and drag the application to your Applications folder.")
        
    elif platform_name == "linux" and extension == ".AppImage":
        # For Linux AppImage files, make it executable
        print(f"Making {package_path} executable...")
        try:
            os.chmod(package_path, 0o755)
            print(f"Installation complete. You can run the application by double-clicking {package_path}")
        except Exception as e:
            print(f"Error making file executable: {e}")
    
    elif extension == ".whl":
        # For Python wheel packages
        print("Installing Python package...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package_path])
            print("Installation complete. You can run the application by executing 'quantum-assistant' in your terminal.")
        except Exception as e:
            print(f"Error installing Python package: {e}")
    
    # Create a desktop shortcut
    create_desktop_shortcut(package_path, platform_name)

def create_desktop_shortcut(target_path, platform_name):
    """Create a desktop shortcut to the application"""
    desktop_path = os.path.expanduser("~/Desktop")
    
    if platform_name == "windows":
        shortcut_path = os.path.join(desktop_path, "Quantum AI Assistant.lnk")
        print(f"Creating desktop shortcut at {shortcut_path}...")
        try:
            # Try to import the winshell module for Windows shortcut creation
            try:
                import winshell
                from win32com.client import Dispatch
                
                shortcut = Dispatch('WScript.Shell').CreateShortCut(shortcut_path)
                shortcut.Targetpath = target_path
                shortcut.WorkingDirectory = os.path.dirname(target_path)
                shortcut.Description = "Quantum AI Assistant"
                shortcut.save()
                print(f"Shortcut created at {shortcut_path}")
            except ImportError:
                # Fall back to direct file creation if winshell is not available
                with open(shortcut_path + ".bat", 'w') as f:
                    f.write(f'@echo off\n"{target_path}"\n')
                print(f"Batch file created at {shortcut_path}.bat")
        except Exception as e:
            print(f"Error creating Windows shortcut: {e}")
        
    elif platform_name == "macos":
        shortcut_path = os.path.join(desktop_path, "Quantum AI Assistant.command")
        print(f"Creating desktop shortcut at {shortcut_path}...")
        try:
            with open(shortcut_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"open {target_path}\n")
            os.chmod(shortcut_path, 0o755)
        except Exception as e:
            print(f"Error creating shortcut: {e}")
            
    else:  # Linux
        shortcut_path = os.path.join(desktop_path, "Quantum AI Assistant.desktop")
        print(f"Creating desktop shortcut at {shortcut_path}...")
        try:
            with open(shortcut_path, 'w') as f:
                f.write("[Desktop Entry]\n")
                f.write("Type=Application\n")
                f.write("Name=Quantum AI Assistant\n")
                f.write(f"Exec={target_path}\n")
                f.write("Icon=terminal\n")
                f.write("Terminal=false\n")
            os.chmod(shortcut_path, 0o755)
        except Exception as e:
            print(f"Error creating shortcut: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantum AI Assistant Download Script")
    parser.add_argument("--destination", type=str, default=os.path.expanduser("~/QuantumAIAssistant"),
                        help="Destination directory for the download")
    parser.add_argument("--python-package", action="store_true", 
                        help="Download the Python package instead of the platform-specific executable")
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    print("="*50)
    print("Quantum AI Assistant Download Script")
    print("="*50)
    
    # Get platform information
    platform_name, arch, extension = get_platform_info()
    print(f"Detected platform: {platform_name} {arch}")
    
    # Use Python package if specified or if on an unsupported platform
    use_python_package = args.python_package
    if use_python_package:
        extension = ".whl"
        print("Using Python package as requested")
    
    # Download the package
    package_path = download_package(platform_name, arch, extension, args.destination, use_python_package)
    
    # Set up the package
    setup_package(package_path, platform_name, extension, args.destination)
    
    print("="*50)
    print("Download and setup complete!")
    print("Enjoy using the Quantum AI Assistant!")
    print("="*50)

if __name__ == "__main__":
    main()