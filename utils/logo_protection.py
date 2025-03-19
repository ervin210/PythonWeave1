"""
Logo Protection Utility

This module provides functions to protect the application logo from unauthorized changes.
It implements a verification system to ensure the logo file remains intact.
"""

import os
import hashlib
import shutil
import time
import threading

# Constants
LOGO_PATH = "assets/quantum_logo.jpg"
SECURE_LOGO_PATH = "secure_assets/quantum_logo.jpg"
LOGO_HASH_FILE = "secure_assets/.logo_hash"

def calculate_file_hash(file_path):
    """Calculate the SHA-256 hash of a file"""
    if not os.path.exists(file_path):
        return None
        
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

def store_original_hash():
    """Store the original hash of the logo file"""
    if not os.path.exists(SECURE_LOGO_PATH):
        # If secure copy doesn't exist yet, we can't store a hash
        return False
    
    # If hash file already exists, no need to recreate it
    if os.path.exists(LOGO_HASH_FILE):
        return True
        
    logo_hash = calculate_file_hash(SECURE_LOGO_PATH)
    
    if logo_hash:
        os.makedirs(os.path.dirname(LOGO_HASH_FILE), exist_ok=True)
        # Make sure the file is writable before trying to write to it
        try:
            with open(LOGO_HASH_FILE, "w") as f:
                f.write(logo_hash)
            # After writing, set read-only permissions
            os.chmod(LOGO_HASH_FILE, 0o444)
            return True
        except PermissionError:
            # File might already be read-only, which is fine
            return True
    
    return False

def verify_logo_integrity():
    """Verify the integrity of the logo file"""
    # Check if hash file exists
    if not os.path.exists(LOGO_HASH_FILE):
        # First run, create the hash file
        return store_original_hash()
    
    # Read the stored hash
    with open(LOGO_HASH_FILE, "r") as f:
        stored_hash = f.read().strip()
    
    # Calculate current hash of the logo
    current_hash = calculate_file_hash(LOGO_PATH)
    
    # If hash doesn't match or file is missing, restore from secure copy
    if current_hash != stored_hash or current_hash is None:
        restore_logo()
        return False
    
    return True

def restore_logo():
    """Restore the logo from the secure copy"""
    if os.path.exists(SECURE_LOGO_PATH):
        # Make sure the destination directory exists
        os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)
        # Copy the secure logo to the assets directory
        shutil.copy2(SECURE_LOGO_PATH, LOGO_PATH)
        return True
    
    return False

def setup_logo_protection():
    """Set up logo protection by creating a secure copy and storing its hash"""
    # Ensure secure assets directory exists
    os.makedirs(os.path.dirname(SECURE_LOGO_PATH), exist_ok=True)
    
    # If the logo exists in assets but not in secure_assets, copy it
    if os.path.exists(LOGO_PATH) and not os.path.exists(SECURE_LOGO_PATH):
        shutil.copy2(LOGO_PATH, SECURE_LOGO_PATH)
    
    # If logo exists in secure_assets but not in assets, restore it
    elif os.path.exists(SECURE_LOGO_PATH) and not os.path.exists(LOGO_PATH):
        restore_logo()
    
    # Store the hash of the original logo
    store_original_hash()
    
    # Start the integrity check thread
    start_integrity_check()

def logo_integrity_checker():
    """Background thread function to periodically check logo integrity"""
    while True:
        verify_logo_integrity()
        time.sleep(60)  # Check every 60 seconds

def start_integrity_check():
    """Start a background thread to periodically check logo integrity"""
    integrity_thread = threading.Thread(target=logo_integrity_checker, daemon=True)
    integrity_thread.start()