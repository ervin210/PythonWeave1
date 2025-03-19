#!/usr/bin/env python3
"""
Launcher script for Quantum AI Assistant
This script provides an easy way to run the Quantum AI Assistant
on various devices and platforms.
"""

import os
import sys
import webbrowser
import subprocess
import time
import platform
import socket
import argparse

def get_local_ip():
    """Get the local IP address for accessing from other devices"""
    try:
        # Create a socket to determine the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"  # Fallback to localhost

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Quantum AI Assistant Launcher")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the application on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser window")
    parser.add_argument("--ibm-token", type=str, help="IBM Quantum API Token")
    parser.add_argument("--wandb-token", type=str, help="Weights & Biases API Token")
    return parser.parse_args()

def main():
    """Main function to run the Quantum AI Assistant"""
    args = parse_arguments()
    
    # Determine the path to app.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    # If the app.py is not found in the current directory, try to find it
    if not os.path.exists(app_path):
        possible_paths = [
            os.path.join(script_dir, "quantum_ai_assistant", "app.py"),
            os.path.join(os.path.dirname(script_dir), "app.py"),
            os.path.join(os.path.dirname(os.path.dirname(script_dir)), "app.py")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                app_path = path
                break
        else:
            print("Error: Could not find app.py. Please run this script from the Quantum AI Assistant directory.")
            sys.exit(1)
    
    # Create command to run the Streamlit app
    command = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(args.port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    # Set environment variables for API tokens if provided
    env = os.environ.copy()
    if args.ibm_token:
        env["IBM_QUANTUM_TOKEN"] = args.ibm_token
    if args.wandb_token:
        env["WANDB_API_KEY"] = args.wandb_token
    
    # Get the local IP address
    ip_address = get_local_ip()
    
    print("="*50)
    print("Quantum AI Assistant Launcher")
    print("="*50)
    print(f"Local URL:     http://localhost:{args.port}")
    print(f"Network URL:   http://{ip_address}:{args.port}")
    print("="*50)
    print("Use the Network URL to access from other devices on your network")
    print("Starting application...")
    
    # Start the Streamlit process
    process = subprocess.Popen(command, env=env)
    
    try:
        # Wait a bit for the server to start
        time.sleep(3)
        
        # Open a browser window (unless --no-browser is specified)
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{args.port}")
        
        print("="*50)
        print("Quantum AI Assistant is running!")
        print("Press Ctrl+C to stop the application")
        print("="*50)
        
        # Keep the script running until Ctrl+C
        process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping Quantum AI Assistant...")
        process.terminate()
        process.wait()
        print("Application stopped.")
    
if __name__ == "__main__":
    main()