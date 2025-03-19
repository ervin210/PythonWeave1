"""
Key Generator Utility

This module provides functions to generate guaranteed unique keys for Streamlit elements.
These functions help prevent StreamlitDuplicateElementKey errors by creating keys
that are guaranteed to be unique across application renders.
"""

import time
import random
import uuid

def generate_unique_key(prefix="key"):
    """
    Generate a guaranteed unique key for Streamlit elements
    
    Args:
        prefix: A prefix to make the key more identifiable (default: "key")
        
    Returns:
        A string containing a unique key that can be used for Streamlit elements
    """
    timestamp = int(time.time() * 1000)  # Millisecond precision
    random_component = random.randint(10000, 99999)
    unique_uuid = str(uuid.uuid4())[:8]  # First 8 chars of a UUID
    
    return f"{prefix}_{timestamp}_{random_component}_{unique_uuid}"

def generate_button_key(button_name):
    """
    Generate a unique key specifically for buttons
    
    Args:
        button_name: A name identifier for the button
        
    Returns:
        A string containing a unique key for the button
    """
    # Clean the button name to make it suitable for a key
    clean_name = button_name.lower().replace(" ", "_")
    return generate_unique_key(f"btn_{clean_name}")

def generate_widget_key(widget_type, identifier=""):
    """
    Generate a unique key for any Streamlit widget
    
    Args:
        widget_type: Type of the widget (e.g., "selectbox", "checkbox", etc.)
        identifier: Additional identifier to make the key more specific
        
    Returns:
        A string containing a unique key for the widget
    """
    if identifier:
        prefix = f"{widget_type}_{identifier}"
    else:
        prefix = widget_type
        
    return generate_unique_key(prefix)

def generate_section_key(section_name):
    """
    Generate a unique key for a specific section of the application
    
    Args:
        section_name: Name of the section or component
        
    Returns:
        A string containing a unique key for the section
    """
    clean_name = section_name.lower().replace(" ", "_")
    return generate_unique_key(f"section_{clean_name}")