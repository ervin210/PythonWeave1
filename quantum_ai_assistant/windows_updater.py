"""
Windows-specific update functionality for Quantum AI Assistant
Provides specialized Windows installation and update capabilities
"""

import os
import sys
import time
import tempfile
import subprocess
import shutil
import ctypes
from urllib.request import urlopen, Request
import json
import platform

def is_admin():
    """Check if the current process has administrator privileges"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def elevate_privileges(script_path, *args):
    """
    Restart the current script with administrator privileges
    
    Args:
        script_path: Path to the script to run with elevated privileges
        *args: Additional arguments to pass to the script
    """
    if not is_admin():
        # Prepare the arguments
        arguments = ' '.join(args)
        
        # The command to execute with elevated privileges
        ctypes.windll.shell32.ShellExecuteW(
            None,           # Parent window handle
            "runas",        # Operation - "runas" means run as administrator
            sys.executable, # Application to execute
            f'"{script_path}" {arguments}',  # Parameters
            None,           # Directory
            1               # Show window normally
        )
        return True
    return False

def download_update(url, callback=None):
    """
    Download an update from the specified URL
    
    Args:
        url: The URL to download from
        callback: Function to call with progress updates
    
    Returns:
        Path to the downloaded file, or None if download failed
    """
    try:
        # Create a temporary directory for the download
        temp_dir = tempfile.mkdtemp()
        filename = os.path.basename(url)
        download_path = os.path.join(temp_dir, filename)
        
        if callback:
            callback(f"Downloading {filename}...")
        
        # Set up the request
        headers = {
            'User-Agent': 'QuantumAIAssistant-WindowsUpdater/1.0'
        }
        request = Request(url, headers=headers)
        
        # Download the file with progress reporting
        with urlopen(request) as response, open(download_path, 'wb') as out_file:
            # Get the file size if available
            file_size = int(response.headers.get('content-length', 0))
            bytes_downloaded = 0
            block_size = 8192
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                bytes_downloaded += len(buffer)
                out_file.write(buffer)
                
                # Report progress
                if file_size > 0 and callback:
                    percent = min(100, int(bytes_downloaded * 100 / file_size))
                    callback(f"Downloaded {percent}% of {filename}")
        
        if callback:
            callback(f"Download complete: {filename}")
        
        return download_path
    except Exception as e:
        if callback:
            callback(f"Download failed: {e}")
        return None

def install_update(installer_path, silent=False, callback=None):
    """
    Install an update from the specified installer path
    
    Args:
        installer_path: Path to the installer file
        silent: Whether to install silently without user interaction
        callback: Function to call with progress updates
    
    Returns:
        True if installation started successfully, False otherwise
    """
    try:
        if callback:
            callback("Starting installation...")
        
        # Check if the file exists
        if not os.path.exists(installer_path):
            if callback:
                callback(f"Installer not found: {installer_path}")
            return False
        
        # Build the command line
        command = [installer_path]
        if silent:
            command.append('/S')  # Silent install flag for NSIS installers
        
        # Start the installer
        if callback:
            callback("Launching installer...")
        
        # Use Popen to avoid waiting for the process to complete
        subprocess.Popen(command)
        
        if callback:
            callback("Installation started. Please follow on-screen instructions if any.")
        
        return True
    except Exception as e:
        if callback:
            callback(f"Installation failed to start: {e}")
        return False

def create_windows_shortcut(target_path, shortcut_path=None, description=None, icon_path=None):
    """
    Create a Windows shortcut file (.lnk)
    
    Args:
        target_path: Path to the target file or program
        shortcut_path: Path where to create the shortcut (default: desktop)
        description: Description for the shortcut
        icon_path: Path to an icon file
    
    Returns:
        Path to the created shortcut, or None if creation failed
    """
    try:
        # Determine shortcut path if not provided
        if not shortcut_path:
            desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
            app_name = os.path.basename(target_path).split('.')[0]
            shortcut_path = os.path.join(desktop, f"{app_name}.lnk")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(shortcut_path), exist_ok=True)
        
        # Try to use the Windows Script Host COM object
        try:
            import pythoncom
            from win32com.client import Dispatch
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = target_path
            shortcut.WorkingDirectory = os.path.dirname(target_path)
            
            if description:
                shortcut.Description = description
            
            if icon_path:
                shortcut.IconLocation = icon_path
            
            shortcut.save()
        except (ImportError, Exception):
            # Fall back to a batch file if COM object fails
            batch_path = shortcut_path.replace('.lnk', '.bat')
            with open(batch_path, 'w') as f:
                f.write(f'@echo off\nstart "" "{target_path}"\n')
            shortcut_path = batch_path
        
        return shortcut_path
    except Exception as e:
        print(f"Error creating shortcut: {e}")
        return None

def setup_update_task(installer_path, silent=True, delay=10):
    """
    Set up a scheduled task to run the installer after a delay
    This is useful if the installer needs to replace files that are in use
    
    Args:
        installer_path: Path to the installer file
        silent: Whether to install silently
        delay: Delay in seconds before running the installer
    
    Returns:
        True if task was set up successfully, False otherwise
    """
    try:
        # Create a batch file to run the installer
        batch_dir = tempfile.mkdtemp()
        batch_path = os.path.join(batch_dir, "run_installer.bat")
        
        with open(batch_path, 'w') as f:
            f.write("@echo off\n")
            f.write(f"timeout /t {delay} /nobreak\n")
            
            # Build the command
            install_cmd = f'"{installer_path}"'
            if silent:
                install_cmd += " /S"
            
            f.write(f"start \"\" {install_cmd}\n")
            # Cleanup
            f.write("del \"%~f0\"\n")
        
        # Execute the batch file
        subprocess.Popen(["cmd", "/c", batch_path], 
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        close_fds=True)
        
        return True
    except Exception as e:
        print(f"Error setting up update task: {e}")
        return False

def check_running_instance(process_name="QuantumAIAssistant.exe"):
    """
    Check if another instance of the application is running
    
    Args:
        process_name: Name of the process to check for
    
    Returns:
        True if the process is running, False otherwise
    """
    try:
        # Use tasklist to check for the process
        output = subprocess.check_output(["tasklist", "/FI", f"IMAGENAME eq {process_name}"], 
                                        universal_newlines=True)
        
        # Check if the process is in the output
        return process_name.lower() in output.lower()
    except:
        # If tasklist fails, assume the process is not running
        return False

def terminate_running_instance(process_name="QuantumAIAssistant.exe"):
    """
    Terminate running instances of the application
    
    Args:
        process_name: Name of the process to terminate
    
    Returns:
        True if terminated successfully or not running, False if failed to terminate
    """
    try:
        # Check if the process is running
        if not check_running_instance(process_name):
            return True
        
        # Try to terminate the process
        subprocess.call(["taskkill", "/F", "/IM", process_name], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL)
        
        # Wait a moment and check again
        time.sleep(1)
        return not check_running_instance(process_name)
    except:
        return False

def is_windows():
    """Check if running on Windows"""
    return platform.system().lower() == "windows"