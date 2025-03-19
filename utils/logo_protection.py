"""
Logo Protection Utility

This module provides functions to protect the application logo from unauthorized changes.
It implements a robust verification system to ensure the logo file remains absolutely immutable.
The system uses multiple verification methods including hash checking, duplicate copies, and
real-time monitoring to guarantee logo integrity across all platforms and conditions.
"""

import os
import hashlib
import shutil
import time
import threading
import logging
import base64
import datetime
import json
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='secure_assets/logo_protection.log'
)
logger = logging.getLogger('LogoProtection')

# Constants
LOGO_PATH = "assets/quantum_logo.svg"  # Using SVG as primary logo
LOGO_PATH_JPG = "assets/quantum_logo.jpg"  # JPG as fallback
SECURE_LOGO_PATH = "secure_assets/quantum_logo.svg"
BACKUP_LOGO_PATH = "secure_assets/quantum_logo_backup.svg"
ENCODED_LOGO_PATH = "secure_assets/.encoded_logo_backup"
LOGO_HASH_FILE = "secure_assets/.logo_hash"
LOGO_METADATA_FILE = "secure_assets/.logo_metadata"
COPYRIGHT_OWNER = "Ervin Remus Radosavlevici"
COPYRIGHT_EMAIL = ["ervin210@icloud.com", "ervin210@sky.com"]
CHECK_INTERVAL = 15  # Check every 15 seconds for maximum protection

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
    """Set up comprehensive logo protection by creating secure copies and storing metadata"""
    logger.info("Setting up logo protection system")
    
    try:
        # Ensure secure assets directory exists
        os.makedirs(os.path.dirname(SECURE_LOGO_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)
        
        # If attached_assets/blob.jpg exists and we don't have an SVG, convert it
        if os.path.exists("attached_assets/blob.jpg") and not os.path.exists(SECURE_LOGO_PATH):
            logger.info("Using attached blob as initial logo")
            # Copy SVG directly if it exists
            if os.path.exists("assets/quantum_logo.svg"):
                shutil.copy2("assets/quantum_logo.svg", SECURE_LOGO_PATH)
            else:
                # Otherwise use the JPG version
                shutil.copy2("attached_assets/blob.jpg", SECURE_LOGO_PATH)
            
            if not os.path.exists(LOGO_PATH):
                if os.path.exists("assets/quantum_logo.svg"):
                    shutil.copy2("assets/quantum_logo.svg", LOGO_PATH)
                else:
                    shutil.copy2("attached_assets/blob.jpg", LOGO_PATH)
        
        # If the logo exists in assets but not in secure_assets, copy it
        elif os.path.exists(LOGO_PATH) and not os.path.exists(SECURE_LOGO_PATH):
            logger.info("Copying existing logo to secure location")
            shutil.copy2(LOGO_PATH, SECURE_LOGO_PATH)
        
        # If logo exists in secure_assets but not in assets, restore it
        elif os.path.exists(SECURE_LOGO_PATH) and not os.path.exists(LOGO_PATH):
            logger.info("Restoring logo from secure location")
            restore_logo()
            
        # Create backup copy
        if os.path.exists(SECURE_LOGO_PATH) and not os.path.exists(BACKUP_LOGO_PATH):
            logger.info("Creating logo backup")
            create_logo_backup()
        
        # Create base64 encoded backup
        if os.path.exists(SECURE_LOGO_PATH) and not os.path.exists(ENCODED_LOGO_PATH):
            logger.info("Creating base64 encoded backup")
            encode_logo_to_base64()
        
        # Store logo metadata
        if os.path.exists(SECURE_LOGO_PATH) and not os.path.exists(LOGO_METADATA_FILE):
            logger.info("Storing logo metadata")
            store_logo_metadata()
        
        # Store the hash of the original logo
        logger.info("Storing logo hash")
        store_original_hash()
        
        # Verify initial integrity
        logger.info("Verifying initial logo integrity")
        integrity_status = verify_logo_integrity()
        if not integrity_status:
            logger.warning("Initial integrity check failed, restoring logo")
            multi_level_restore()
        
        # Start the integrity check thread
        logger.info("Starting continuous integrity monitoring")
        start_integrity_check()
        
        logger.info("Logo protection system fully initialized")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up logo protection: {str(e)}")
        return False

def encode_logo_to_base64():
    """Encode the logo as base64 for an additional backup layer"""
    if not os.path.exists(SECURE_LOGO_PATH):
        logger.error("Cannot encode logo: Secure logo file not found")
        return False
    
    try:
        with open(SECURE_LOGO_PATH, "rb") as img_file:
            encoded_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        os.makedirs(os.path.dirname(ENCODED_LOGO_PATH), exist_ok=True)
        with open(ENCODED_LOGO_PATH, "w") as f:
            f.write(encoded_data)
        
        # Set read-only permissions
        os.chmod(ENCODED_LOGO_PATH, 0o444)
        logger.info("Logo encoded to base64 and stored")
        return True
    except Exception as e:
        logger.error(f"Error encoding logo to base64: {str(e)}")
        return False

def decode_and_restore_logo():
    """Restore the logo from the base64 encoded backup"""
    if not os.path.exists(ENCODED_LOGO_PATH):
        logger.error("Cannot decode logo: Encoded backup not found")
        return False
    
    try:
        with open(ENCODED_LOGO_PATH, "r") as f:
            encoded_data = f.read()
        
        # Decode the base64 data
        decoded_data = base64.b64decode(encoded_data)
        
        # Verify it's a valid image
        try:
            Image.open(BytesIO(decoded_data))
        except:
            logger.error("Encoded backup does not contain a valid image")
            return False
        
        # Save the decoded image to the secure location
        os.makedirs(os.path.dirname(SECURE_LOGO_PATH), exist_ok=True)
        with open(SECURE_LOGO_PATH, "wb") as img_file:
            img_file.write(decoded_data)
        
        # Also restore the main logo
        with open(LOGO_PATH, "wb") as img_file:
            img_file.write(decoded_data)
        
        logger.info("Logo successfully restored from encoded backup")
        return True
    except Exception as e:
        logger.error(f"Error decoding and restoring logo: {str(e)}")
        return False

def create_logo_backup():
    """Create an additional backup of the logo file"""
    if not os.path.exists(SECURE_LOGO_PATH):
        logger.error("Cannot create backup: Secure logo file not found")
        return False
    
    try:
        # Make sure the destination directory exists
        os.makedirs(os.path.dirname(BACKUP_LOGO_PATH), exist_ok=True)
        # Copy the secure logo to the backup location
        shutil.copy2(SECURE_LOGO_PATH, BACKUP_LOGO_PATH)
        logger.info("Logo backup created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating logo backup: {str(e)}")
        return False

def store_logo_metadata():
    """Store metadata about the logo for additional verification"""
    if not os.path.exists(SECURE_LOGO_PATH):
        logger.error("Cannot store metadata: Secure logo file not found")
        return False
    
    try:
        # For SVG files, we handle differently than raster images
        if SECURE_LOGO_PATH.lower().endswith('.svg'):
            # For SVG, store file info instead of image properties
            file_size = os.path.getsize(SECURE_LOGO_PATH)
            file_hash = calculate_file_hash(SECURE_LOGO_PATH)
            
            with open(SECURE_LOGO_PATH, 'r') as f:
                svg_content = f.read()
                
            # Try to extract width and height from SVG
            import re
            width_match = re.search(r'width="(\d+)"', svg_content)
            height_match = re.search(r'height="(\d+)"', svg_content)
            
            width = int(width_match.group(1)) if width_match else 0
            height = int(height_match.group(1)) if height_match else 0
            
            metadata = {
                "format": "SVG",
                "mode": "vector",
                "width": width,
                "height": height,
                "created": datetime.datetime.now().isoformat(),
                "hash": file_hash,
                "size": file_size
            }
        else:
            # For raster images, use PIL
            img = Image.open(SECURE_LOGO_PATH)
            
            # Get image properties
            metadata = {
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "created": datetime.datetime.now().isoformat(),
                "hash": calculate_file_hash(SECURE_LOGO_PATH),
                "size": os.path.getsize(SECURE_LOGO_PATH)
            }
        
        # Save metadata
        os.makedirs(os.path.dirname(LOGO_METADATA_FILE), exist_ok=True)
        with open(LOGO_METADATA_FILE, "w") as f:
            f.write(json.dumps(metadata, indent=2))
        
        # Set read-only permissions
        os.chmod(LOGO_METADATA_FILE, 0o444)
        logger.info("Logo metadata stored successfully")
        return True
    except Exception as e:
        logger.error(f"Error storing logo metadata: {str(e)}")
        return False

def verify_logo_with_metadata():
    """Verify logo integrity using stored metadata"""
    if not os.path.exists(LOGO_METADATA_FILE) or not os.path.exists(LOGO_PATH):
        logger.error("Cannot verify with metadata: Files missing")
        return False
    
    try:
        # Read metadata
        with open(LOGO_METADATA_FILE, "r") as f:
            metadata = json.loads(f.read())
        
        # Verify current logo
        img = Image.open(LOGO_PATH)
        current_hash = calculate_file_hash(LOGO_PATH)
        
        # Check core properties
        if (img.width != metadata["width"] or 
            img.height != metadata["height"] or 
            current_hash != metadata["hash"]):
            logger.warning("Logo integrity check failed using metadata")
            return False
        
        logger.info("Logo verified successfully with metadata")
        return True
    except Exception as e:
        logger.error(f"Error verifying logo with metadata: {str(e)}")
        return False

def multi_level_restore():
    """Implement multi-level logo restoration with fallbacks"""
    logger.info("Starting multi-level logo restoration")
    
    # Try primary restore
    if os.path.exists(SECURE_LOGO_PATH):
        if restore_logo():
            logger.info("Logo restored from primary secure copy")
            return True
    
    # Try backup restore
    if os.path.exists(BACKUP_LOGO_PATH):
        try:
            shutil.copy2(BACKUP_LOGO_PATH, LOGO_PATH)
            shutil.copy2(BACKUP_LOGO_PATH, SECURE_LOGO_PATH)
            logger.info("Logo restored from backup copy")
            return True
        except Exception as e:
            logger.error(f"Error restoring from backup: {str(e)}")
    
    # Try base64 encoded restore
    if decode_and_restore_logo():
        logger.info("Logo restored from base64 encoded backup")
        return True
    
    logger.error("All logo restoration attempts failed")
    return False

def logo_integrity_checker():
    """Background thread function to periodically check logo integrity"""
    logger.info("Starting logo integrity checker thread")
    
    # Perform initial setup
    if not os.path.exists(BACKUP_LOGO_PATH):
        create_logo_backup()
    
    if not os.path.exists(ENCODED_LOGO_PATH):
        encode_logo_to_base64()
    
    if not os.path.exists(LOGO_METADATA_FILE):
        store_logo_metadata()
        
    while True:
        # Primary hash check
        if not verify_logo_integrity():
            logger.warning("Logo integrity check failed, attempting restoration")
            multi_level_restore()
        
        # Secondary metadata check
        if not verify_logo_with_metadata():
            logger.warning("Logo metadata verification failed, attempting restoration")
            multi_level_restore()
        
        time.sleep(CHECK_INTERVAL)

def start_integrity_check():
    """Start a background thread to periodically check logo integrity"""
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(SECURE_LOGO_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(LOGO_PATH), exist_ok=True)
        
        # Start the integrity checker thread
        integrity_thread = threading.Thread(target=logo_integrity_checker, daemon=True)
        integrity_thread.start()
        logger.info("Logo integrity checker thread started")
    except Exception as e:
        logger.error(f"Error starting integrity check: {str(e)}")