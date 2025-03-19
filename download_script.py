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
import time
from urllib.request import urlretrieve

# Release URLs - Main and Backup
GITHUB_RELEASE_BASE_URL = "https://github.com/quantum-ai-assistant/releases/download/v1.0.0"
BACKUP_RELEASE_BASE_URL = "https://quantum-ai-assistant.com/downloads/v1.0.0"

# Import Windows-specific functions if on Windows
if platform.system().lower() == "windows":
    try:
        # Try to import our specialized Windows functions
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from quantum_ai_assistant.windows_updater import (
            is_admin, elevate_privileges, download_update, 
            install_update, create_windows_shortcut,
            setup_update_task, terminate_running_instance
        )
    except ImportError:
        # Define minimal versions of the functions for standalone use
        def is_admin():
            """Check if running with admin privileges"""
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except:
                return False
        
        def elevate_privileges(script_path, *args):
            """Request elevation to admin privileges"""
            try:
                import ctypes
                arguments = ' '.join(args)
                ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, 
                                                  f'"{script_path}" {arguments}', None, 1)
                return True
            except:
                return False
        
        def download_update(url, callback=None):
            """Download file with progress reporting"""
            try:
                # Simple download implementation
                if callback:
                    callback(f"Downloading from {url}...")
                
                temp_dir = tempfile.mkdtemp()
                filename = os.path.basename(url)
                local_path = os.path.join(temp_dir, filename)
                
                # Download with basic progress
                def report_progress(count, block_size, total_size):
                    if total_size > 0 and callback:
                        percent = min(100, int(count * block_size * 100 / total_size))
                        callback(f"Downloaded {percent}%")
                
                urlretrieve(url, local_path, report_progress)
                
                if callback:
                    callback("Download complete")
                
                return local_path
            except Exception as e:
                if callback:
                    callback(f"Download failed: {e}")
                return None
        
        def install_update(installer_path, silent=False, callback=None):
            """Start the installer"""
            try:
                if callback:
                    callback("Starting installer...")
                
                cmd = [installer_path]
                if silent:
                    cmd.append('/S')
                
                subprocess.Popen(cmd)
                
                if callback:
                    callback("Installer started")
                
                return True
            except Exception as e:
                if callback:
                    callback(f"Failed to start installer: {e}")
                return False
        
        def create_windows_shortcut(target_path, shortcut_path=None, description=None, icon_path=None):
            """Create a Windows shortcut"""
            try:
                if shortcut_path is None:
                    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
                    shortcut_path = os.path.join(desktop, "Quantum AI Assistant.lnk")
                
                # Create a batch file as a fallback
                bat_path = shortcut_path.replace('.lnk', '.bat')
                with open(bat_path, 'w') as f:
                    f.write(f'@echo off\nstart "" "{target_path}"\n')
                
                return bat_path
            except:
                return None
        
        def setup_update_task(installer_path, silent=True, delay=10):
            """Set up a delayed installer task"""
            try:
                batch_path = os.path.join(tempfile.gettempdir(), "run_installer.bat")
                with open(batch_path, 'w') as f:
                    f.write("@echo off\n")
                    f.write(f"timeout /t {delay} /nobreak\n")
                    install_cmd = f'"{installer_path}"'
                    if silent:
                        install_cmd += " /S"
                    f.write(f"start \"\" {install_cmd}\n")
                    f.write("del \"%~f0\"\n")
                
                subprocess.Popen(["cmd", "/c", batch_path])
                return True
            except:
                return False
        
        def terminate_running_instance(process_name="QuantumAIAssistant.exe"):
            """Terminate a process by name"""
            try:
                subprocess.call(["taskkill", "/F", "/IM", process_name], 
                              stdout=subprocess.DEVNULL, 
                              stderr=subprocess.DEVNULL)
                time.sleep(1)
                return True
            except:
                return False

# URLs are already defined above

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
        # For Windows executables
        print("Setting up Windows package...")
        
        # Try to use the specialized Windows updater module if available
        try:
            # Check if we're running as administrator
            if 'is_admin' in globals() and not is_admin():
                print("Administrator privileges required for installation.")
                print("Requesting elevation...")
                
                if 'elevate_privileges' in globals():
                    # Re-run with admin privileges
                    if elevate_privileges(sys.argv[0], "--destination", install_dir):
                        print("Installation will continue with administrator privileges.")
                        sys.exit(0)
            
            # Install the package
            if 'install_update' in globals():
                print("Installing package using Windows installer...")
                if install_update(package_path, silent=False, 
                                 callback=lambda msg: print(f"Install: {msg}")):
                    print("Installation process started successfully.")
                else:
                    # Fall back to manual instruction
                    print(f"Please run the installer manually: {package_path}")
            else:
                # Default method
                print(f"Installation complete. You can run the application by double-clicking {package_path}")
                
                # Try to run the installer
                try:
                    subprocess.Popen([package_path])
                    print("Installer started. Please follow on-screen instructions.")
                except Exception as e:
                    print(f"Could not start installer automatically: {e}")
                    print(f"Please run the installer manually: {package_path}")
        except Exception as e:
            print(f"Windows installation error: {e}")
            print(f"Please run the installer manually: {package_path}")
        
    elif platform_name == "macos" and extension == ".dmg":
        # For macOS DMG files
        print(f"Installation complete. Please open {package_path} and drag the application to your Applications folder.")
        
        # Try to mount the DMG file
        try:
            subprocess.run(["open", package_path])
            print("DMG file opened. Please follow the on-screen instructions.")
        except Exception as e:
            print(f"Could not open DMG file automatically: {e}")
        
    elif platform_name == "linux" and extension == ".AppImage":
        # For Linux AppImage files, make it executable
        print(f"Making {package_path} executable...")
        try:
            os.chmod(package_path, 0o755)
            print(f"Installation complete. You can run the application by double-clicking {package_path}")
            
            # Try to create XDG desktop entry
            try:
                app_name = "QuantumAIAssistant"
                desktop_dir = os.path.expanduser("~/.local/share/applications")
                os.makedirs(desktop_dir, exist_ok=True)
                desktop_path = os.path.join(desktop_dir, f"{app_name}.desktop")
                
                with open(desktop_path, 'w') as f:
                    f.write("[Desktop Entry]\n")
                    f.write("Type=Application\n")
                    f.write("Name=Quantum AI Assistant\n")
                    f.write(f"Exec={package_path}\n")
                    f.write("Icon=quantum\n")
                    f.write("Terminal=false\n")
                    f.write("Categories=Science;Education;\n")
                
                os.chmod(desktop_path, 0o755)
                print(f"Desktop entry created at {desktop_path}")
            except Exception as e:
                print(f"Could not create desktop entry: {e}")
                
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
        
        # Try to use the specialized Windows function if available
        try:
            if 'create_windows_shortcut' in globals():
                # Use the specialized function
                result = create_windows_shortcut(
                    target_path, 
                    shortcut_path, 
                    "Quantum AI Assistant - Quantum Computing with W&B Integration",
                    None  # No custom icon path
                )
                if result:
                    print(f"Shortcut created at {result}")
                    return
            
            # Fall back to default methods if specialized function failed or isn't available
            try:
                # Try COM automation first
                try:
                    import pythoncom
                    from win32com.client import Dispatch
                    
                    shell = Dispatch('WScript.Shell')
                    shortcut = shell.CreateShortCut(shortcut_path)
                    shortcut.Targetpath = target_path
                    shortcut.WorkingDirectory = os.path.dirname(target_path)
                    shortcut.Description = "Quantum AI Assistant"
                    shortcut.save()
                    print(f"Shortcut created at {shortcut_path}")
                except ImportError:
                    # Fall back to direct file creation if COM is not available
                    with open(shortcut_path + ".bat", 'w') as f:
                        f.write(f'@echo off\nstart "" "{target_path}"\n')
                    print(f"Batch file created at {shortcut_path}.bat")
            except Exception as e:
                print(f"Error creating Windows shortcut via standard methods: {e}")
                
                # Last resort - create a simple batch file
                try:
                    batch_path = os.path.join(desktop_path, "Quantum AI Assistant.bat")
                    with open(batch_path, 'w') as f:
                        f.write(f'@echo off\necho Starting Quantum AI Assistant...\n"{target_path}"\n')
                    print(f"Created batch file shortcut at {batch_path}")
                except Exception as batch_e:
                    print(f"All shortcut creation methods failed: {batch_e}")
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