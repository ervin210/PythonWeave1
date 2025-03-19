#!/usr/bin/env python3
"""
Build cross-platform packages for the Quantum AI Assistant
"""

import os
import shutil
import subprocess
import platform

def create_directory_structure():
    """Create the necessary directory structure for packaging"""
    os.makedirs("dist", exist_ok=True)
    os.makedirs("build", exist_ok=True)
    os.makedirs("quantum_ai_assistant", exist_ok=True)
    
    # Create necessary subdirectories
    dirs = ["assets", "components", "secure_assets", ".streamlit"]
    for directory in dirs:
        os.makedirs(f"quantum_ai_assistant/{directory}", exist_ok=True)

def copy_source_files():
    """Copy source files to the package directory"""
    # Copy Python files
    py_files = [
        "app.py", 
        "components.py", 
        "quantum_assistant.py", 
        "utils.py", 
        "visualizations.py",
        "push_to_wandb.py"
    ]
    
    for file in py_files:
        if os.path.exists(file):
            shutil.copy(file, f"quantum_ai_assistant/{file}")
    
    # Copy components directory
    if os.path.exists("components"):
        for file in os.listdir("components"):
            if file.endswith(".py"):
                shutil.copy(f"components/{file}", f"quantum_ai_assistant/components/{file}")
    
    # Copy .streamlit config
    if os.path.exists(".streamlit/config.toml"):
        shutil.copy(".streamlit/config.toml", "quantum_ai_assistant/.streamlit/config.toml")
    
    # Copy README
    if os.path.exists("README.md"):
        shutil.copy("README.md", "quantum_ai_assistant/README.md")
    
    # Create __init__.py files
    open("quantum_ai_assistant/__init__.py", "w").close()
    open("quantum_ai_assistant/components/__init__.py", "w").close()

def create_platform_package():
    """Create platform-specific packages"""
    system = platform.system().lower()
    
    # Build Python package
    subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"])
    
    # Create platform-specific executable package
    try:
        subprocess.run(["pip", "install", "pyinstaller"])
        if system == "windows":
            subprocess.run([
                "pyinstaller", 
                "--name=QuantumAIAssistant",
                "--onefile",
                "--windowed",
                "--add-data=quantum_ai_assistant;quantum_ai_assistant",
                "quantum_ai_assistant/app.py"
            ])
        elif system == "darwin":  # macOS
            subprocess.run([
                "pyinstaller", 
                "--name=QuantumAIAssistant",
                "--onefile",
                "--windowed",
                "--add-data=quantum_ai_assistant:quantum_ai_assistant",
                "quantum_ai_assistant/app.py"
            ])
        else:  # Linux
            subprocess.run([
                "pyinstaller", 
                "--name=QuantumAIAssistant",
                "--onefile",
                "--add-data=quantum_ai_assistant:quantum_ai_assistant",
                "quantum_ai_assistant/app.py"
            ])
        
        print(f"Successfully created package for {system}")
    except Exception as e:
        print(f"Error creating executable for {system}: {e}")
        print("You can still use the Python package with 'pip install dist/*.whl'")

if __name__ == "__main__":
    create_directory_structure()
    copy_source_files()
    create_platform_package()
    print("Build complete. Check the 'dist' directory for the installable package.")
    print("To run the application after installation, use the command: quantum-assistant")