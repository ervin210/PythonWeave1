#!/usr/bin/env python3
"""
Auto-updater for Quantum AI Assistant
Checks for updates and helps users install the latest version
"""

import os
import sys
import json
import time
import platform
import tempfile
import shutil
import subprocess
from urllib.request import urlopen, Request
import threading

# Version information - will be used to check for updates
CURRENT_VERSION = "1.0.0"

# Update endpoints
UPDATE_CHECK_URL = "https://api.github.com/repos/quantum-ai-assistant/releases/latest"
DOWNLOAD_BASE_URL = "https://github.com/quantum-ai-assistant/releases/download"

# Production fallback endpoints if GitHub is unavailable
BACKUP_UPDATE_CHECK_URL = "https://quantum-ai-assistant.com/api/updates/check"
BACKUP_DOWNLOAD_BASE_URL = "https://quantum-ai-assistant.com/downloads"

# How often to check for updates (in seconds)
UPDATE_CHECK_INTERVAL = 86400  # 24 hours

class AutoUpdater:
    """Handles checking for updates and updating the application"""
    
    def __init__(self):
        """Initialize the auto-updater"""
        self.platform_name, self.arch, self.extension = self._get_platform_info()
        self.update_available = False
        self.latest_version = CURRENT_VERSION
        self.update_url = None
        self.update_notes = None
        self.last_checked = 0
        self.check_in_progress = False
    
    def _get_platform_info(self):
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
    
    def check_for_updates(self, force=False):
        """
        Check if updates are available
        
        Args:
            force: If True, check even if recently checked
            
        Returns:
            tuple: (update_available, version, url, notes)
        """
        # Don't check too frequently unless forced
        current_time = time.time()
        if not force and current_time - self.last_checked < UPDATE_CHECK_INTERVAL:
            return (self.update_available, self.latest_version, self.update_url, self.update_notes)
        
        # Don't run multiple checks at once
        if self.check_in_progress:
            return (self.update_available, self.latest_version, self.update_url, self.update_notes)
        
        self.check_in_progress = True
        self.last_checked = current_time
        
        try:
            # For development or testing environments
            if "example.com" in UPDATE_CHECK_URL:
                print("Development mode: Using mock update check")
                self.update_available = False
                self.latest_version = CURRENT_VERSION
                return (self.update_available, self.latest_version, self.update_url, self.update_notes)
            
            # Create a user agent for the request
            headers = {
                'User-Agent': f'QuantumAIAssistant/{CURRENT_VERSION}'
            }
            
            # Try primary GitHub API endpoint first
            data = None
            try:
                request = Request(UPDATE_CHECK_URL, headers=headers)
                with urlopen(request, timeout=5) as response:
                    response_text = response.read().decode('utf-8')
                    if not response_text.strip():
                        print("Warning: Empty response from primary update server")
                        raise ValueError("Empty response received")
                    data = json.loads(response_text)
            except Exception as e:
                print(f"Primary update check failed: {e}")
                # Try fallback endpoint
                try:
                    request = Request(BACKUP_UPDATE_CHECK_URL, headers=headers)
                    with urlopen(request, timeout=5) as response:
                        response_text = response.read().decode('utf-8')
                        if not response_text.strip():
                            print("Warning: Empty response from backup update server")
                            raise ValueError("Empty response received")
                        data = json.loads(response_text)
                except Exception as fallback_error:
                    print(f"Fallback update check also failed: {fallback_error}")
                    # For a better user experience, don't raise an exception - just report no updates
                    self.update_available = False
                    self.latest_version = CURRENT_VERSION
                    self.update_url = ""
                    self.update_notes = "Could not check for updates. Will try again later."
                    self.check_in_progress = False
                    return (False, CURRENT_VERSION, "", "Update check failed. Will try again later.")
            
            if not data:
                raise ValueError("Failed to retrieve update data from any endpoint")
                
            # Extract version (remove 'v' prefix if present)
            latest_version = data.get('tag_name', '').lstrip('v')
            
            # Convert versions to tuples for comparison (e.g., "1.2.3" -> (1, 2, 3))
            current_parts = tuple(map(int, CURRENT_VERSION.split('.')))
            try:
                latest_parts = tuple(map(int, latest_version.split('.')))
                
                # If we have a new version
                if latest_parts > current_parts:
                    self.update_available = True
                    self.latest_version = latest_version
                    
                    # Find the right asset for this platform
                    asset_name = f"QuantumAIAssistant-{self.platform_name}-{self.arch}{self.extension}"
                    for asset in data.get('assets', []):
                        if asset['name'] == asset_name:
                            self.update_url = asset['browser_download_url']
                            break
                    
                    # If no platform-specific asset, fall back to the Python package
                    if not self.update_url:
                        for asset in data.get('assets', []):
                            if asset['name'].endswith('.whl'):
                                self.update_url = asset['browser_download_url']
                                break
                    
                    # Get update notes
                    self.update_notes = data.get('body', 'New version available!')
                else:
                    self.update_available = False
            except (ValueError, TypeError):
                # Handle invalid version format
                print(f"Invalid version format received: {latest_version}")
                self.update_available = False
                
        except Exception as e:
            print(f"Error checking for updates: {e}")
            self.update_available = False
            # In development or testing environments, don't show error messages to users
            if "example.com" in UPDATE_CHECK_URL:
                print("Development mode: Ignoring update check error")
            else:
                # In production, log the error properly
                import traceback
                traceback.print_exc()
        
        self.check_in_progress = False
        return (self.update_available, self.latest_version, self.update_url, self.update_notes)
    
    def download_and_install_update(self, callback=None):
        """
        Download and install the latest update
        
        Args:
            callback: Function to call with progress updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.update_available or not self.update_url:
            return False
        
        # For development mode with example.com URLs
        if "example.com" in self.update_url:
            print("Development mode: Simulating update download and install")
            if callback:
                callback("Development mode: Simulating update download...")
                time.sleep(1)  # Simulate a small delay
                callback("Development mode: Simulating update installation...")
                time.sleep(1)  # Simulate a small delay
                callback("Update simulated successfully in development mode!")
            return True
            
        try:
            # Progress updates
            if callback:
                callback("Downloading update...")
            
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Determine the download filename
                filename = os.path.basename(self.update_url)
                download_path = os.path.join(temp_dir, filename)
                
                # Try to download the file, with fallback
                download_success = False
                download_errors = []
                
                # Try primary URL first
                try:
                    if callback:
                        callback(f"Downloading from primary source...")
                    with urlopen(self.update_url, timeout=30) as response, open(download_path, 'wb') as out_file:
                        data = response.read()
                        out_file.write(data)
                    download_success = True
                except Exception as e:
                    download_errors.append(f"Primary download failed: {e}")
                    
                    # Try backup URL if primary fails
                    if callback:
                        callback(f"Primary download failed, trying backup source...")
                    
                    try:
                        # Use BACKUP_DOWNLOAD_BASE_URL with the version and platform info
                        backup_url = f"{BACKUP_DOWNLOAD_BASE_URL}/v{self.latest_version}/{filename}"
                        with urlopen(backup_url, timeout=30) as response, open(download_path, 'wb') as out_file:
                            data = response.read()
                            out_file.write(data)
                        download_success = True
                    except Exception as backup_e:
                        download_errors.append(f"Backup download failed: {backup_e}")
                
                # If both download attempts failed
                if not download_success:
                    error_msg = "\n".join(download_errors)
                    if callback:
                        callback(f"Failed to download update: {error_msg}")
                    return False
                
                if callback:
                    callback("Download complete. Installing...")
                
                # Handle different file types
                if filename.endswith('.whl'):
                    # Install Python package
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--upgrade", download_path],
                        check=True
                    )
                    success = result.returncode == 0
                
                elif self.platform_name == "windows" and filename.endswith('.exe'):
                    # For Windows, try to use specialized Windows updater
                    try:
                        from quantum_ai_assistant.windows_updater import (
                            terminate_running_instance, setup_update_task, install_update
                        )
                        
                        # First try to terminate any running instances
                        if callback:
                            callback("Checking for running instances...")
                        
                        terminate_running_instance("QuantumAIAssistant.exe")
                        
                        # Then set up a delayed installation task
                        if callback:
                            callback("Setting up update installation...")
                            
                        # Try to use the Windows update task system for deferred installation
                        # This helps when the current executable is locked
                        if setup_update_task(download_path, silent=True, delay=3):
                            if callback:
                                callback("Update scheduled successfully.")
                            success = True
                        else:
                            # If task setup fails, try direct installation
                            if callback:
                                callback("Attempting direct installation...")
                            if install_update(download_path, silent=True, 
                                           callback=lambda msg: callback(f"Install: {msg}") if callback else None):
                                success = True
                            else:
                                # Last resort, just try to run the installer directly
                                if callback:
                                    callback("Starting Windows installer directly...")
                                subprocess.Popen([download_path], shell=True)
                                success = True
                        
                    except ImportError:
                        # If specialized Windows module is not available, fall back to basic approach
                        try:
                            if callback:
                                callback("Starting Windows installer...")
                            subprocess.Popen([download_path], shell=True)
                            success = True
                        except Exception as e:
                            if callback:
                                callback(f"Failed to start installer: {e}")
                            success = False
                    except Exception as e:
                        if callback:
                            callback(f"Windows update error: {e}. Trying basic installation...")
                        # Fall back to basic installation method
                        try:
                            subprocess.Popen([download_path], shell=True)
                            success = True
                        except Exception as basic_e:
                            if callback:
                                callback(f"Failed to start installer: {basic_e}")
                            success = False
                
                elif self.platform_name == "macos" and filename.endswith('.dmg'):
                    # Mount DMG and copy app to Applications
                    subprocess.run(["open", download_path], check=True)
                    if callback:
                        callback("DMG opened. Please follow the installation instructions.")
                    success = True
                
                elif self.platform_name == "linux" and filename.endswith('.AppImage'):
                    # Make the AppImage executable and move it to a permanent location
                    install_dir = os.path.expanduser("~/QuantumAIAssistant")
                    os.makedirs(install_dir, exist_ok=True)
                    
                    install_path = os.path.join(install_dir, "QuantumAIAssistant.AppImage")
                    shutil.copy2(download_path, install_path)
                    os.chmod(install_path, 0o755)
                    
                    # Create desktop shortcut
                    desktop_path = os.path.expanduser("~/Desktop/Quantum AI Assistant.desktop")
                    with open(desktop_path, 'w') as f:
                        f.write("[Desktop Entry]\n")
                        f.write("Type=Application\n")
                        f.write("Name=Quantum AI Assistant\n")
                        f.write(f"Exec={install_path}\n")
                        f.write("Icon=terminal\n")
                        f.write("Terminal=false\n")
                    os.chmod(desktop_path, 0o755)
                    
                    success = True
                
                else:
                    # Unknown file type
                    if callback:
                        callback(f"Unknown update file format: {filename}")
                    success = False
                
                if success and callback:
                    callback(f"Update to version {self.latest_version} complete!")
                
                return success
                
        except Exception as e:
            if callback:
                callback(f"Error installing update: {e}")
            return False
    
    def start_background_check(self):
        """Start background thread to check for updates periodically"""
        def check_thread():
            while True:
                self.check_for_updates()
                time.sleep(UPDATE_CHECK_INTERVAL)
        
        thread = threading.Thread(target=check_thread, daemon=True)
        thread.start()
        return thread

# Initialize the updater when the module is imported
updater = AutoUpdater()

def check_for_updates_now():
    """Convenience function to check for updates immediately"""
    return updater.check_for_updates(force=True)

def download_and_install_update(callback=None):
    """Convenience function to download and install updates"""
    return updater.download_and_install_update(callback)

def start_background_update_checks():
    """Start periodic background update checks"""
    return updater.start_background_check()

# Automatically start background update checks when imported
update_thread = start_background_update_checks()