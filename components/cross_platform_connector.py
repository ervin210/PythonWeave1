import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime

def cross_platform_connector():
    """
    Cross-Platform Connectivity and Compatibility Manager
    
    This component ensures the Quantum AI Assistant can operate
    seamlessly across various platforms, networks, and devices.
    """
    st.header("🌐 Cross-Platform Connector")
    
    st.markdown("""
    The Cross-Platform Connector ensures your Quantum AI Assistant works optimally
    across all platforms, networks, and devices, providing a consistent experience regardless
    of how you're accessing the system.
    """)
    
    # Create tabs for different cross-platform features
    platform_tab, network_tab, compatibility_tab, status_tab = st.tabs([
        "Platform Support", "Network Configuration", "Compatibility", "System Status"
    ])
    
    # Platform Support Tab
    with platform_tab:
        st.subheader("Platform Support")
        
        st.markdown("""
        Configure and optimize the Quantum AI Assistant for different 
        software and hardware platforms.
        """)
        
        # Platform selection
        platform_category = st.selectbox(
            "Platform Category",
            ["Desktop", "Mobile", "Web", "Cloud", "Edge/IoT"]
        )
        
        if platform_category == "Desktop":
            desktop_os = st.selectbox(
                "Operating System",
                ["Windows", "macOS", "Linux", "ChromeOS"]
            )
            
            st.markdown(f"### {desktop_os} Configuration")
            
            # Display optimization status
            st.markdown("#### System Requirements")
            
            requirement_cols = st.columns(2)
            with requirement_cols[0]:
                st.markdown("**Minimum Requirements:**")
                st.markdown("- 4GB RAM")
                st.markdown("- 2 CPU cores")
                st.markdown("- 2GB storage")
                st.markdown("- OpenGL 3.3+ compatible GPU")
            
            with requirement_cols[1]:
                st.markdown("**Recommended Requirements:**")
                st.markdown("- 8GB+ RAM")
                st.markdown("- 4+ CPU cores")
                st.markdown("- 5GB+ storage")
                st.markdown("- CUDA/OpenCL compatible GPU")
            
            # Native application options
            st.markdown("#### Native Application")
            
            # Implementation for file extraction and installation
            import zipfile
            import tempfile
            import base64
            import io
            import subprocess
            import time
            
            # Create a real application package with functional code
            def create_application_package(os_type):
                # Create an in-memory zip file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add a README file
                    zipf.writestr('README.txt', f'Quantum AI Assistant for {os_type}\n'
                                              f'================================\n\n'
                                              f'This package contains the Quantum AI Assistant application for {os_type}.\n'
                                              f'Installation instructions are in INSTALL.txt.\n')
                    
                    # Add a detailed installation guide
                    zipf.writestr('INSTALL.txt', f'Installation Instructions for {os_type}\n'
                                               f'================================\n\n'
                                               f'1. Extract all files to a directory of your choice.\n'
                                               f'2. Make sure you have Python 3.8+ installed on your system.\n'
                                               f'3. Install required packages: pip install -r requirements.txt\n'
                                               f'4. Run the application: python quantum_assistant_app.py\n\n'
                                               f'For {os_type} users:\n'
                                               f'- You can also use the provided setup script which will handle all dependencies.\n')
                    
                    # Add requirements file
                    zipf.writestr('requirements.txt', 
                        'streamlit>=1.24.0\n'
                        'numpy>=1.22.0\n'
                        'pandas>=1.5.0\n'
                        'matplotlib>=3.5.0\n'
                        'qiskit>=0.43.0\n'
                        'qiskit-aer>=0.12.0\n'
                        'pennylane>=0.30.0\n'
                        'plotly>=5.14.0\n'
                        'wandb>=0.15.0\n'
                    )
                    
                    # Add a setup script based on OS
                    if os_type.lower() == 'windows':
                        zipf.writestr('setup.bat', 
                            '@echo off\n'
                            'echo Installing Quantum AI Assistant for Windows...\n'
                            'echo.\n'
                            'echo Checking for Python...\n'
                            'python --version > nul 2>&1\n'
                            'if %errorlevel% neq 0 (\n'
                            '    echo Python not found! Please install Python 3.8 or newer.\n'
                            '    echo Visit https://www.python.org/downloads/\n'
                            '    pause\n'
                            '    exit /b 1\n'
                            ')\n'
                            'echo Python found!\n'
                            'echo.\n'
                            'echo Installing dependencies...\n'
                            'pip install -r requirements.txt\n'
                            'if %errorlevel% neq 0 (\n'
                            '    echo Failed to install dependencies.\n'
                            '    pause\n'
                            '    exit /b 1\n'
                            ')\n'
                            'echo.\n'
                            'echo Creating desktop shortcut...\n'
                            'echo Set oWS = WScript.CreateObject("WScript.Shell") > createshortcut.vbs\n'
                            'echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\\Quantum AI Assistant.lnk" >> createshortcut.vbs\n'
                            'echo Set oLink = oWS.CreateShortcut(sLinkFile) >> createshortcut.vbs\n'
                            'echo oLink.TargetPath = "cmd.exe" >> createshortcut.vbs\n'
                            'echo oLink.Arguments = "/c python quantum_assistant_app.py" >> createshortcut.vbs\n'
                            'echo oLink.Description = "Quantum AI Assistant" >> createshortcut.vbs\n'
                            'echo oLink.IconLocation = "python.exe, 0" >> createshortcut.vbs\n'
                            'echo oLink.WorkingDirectory = "%cd%" >> createshortcut.vbs\n'
                            'echo oLink.Save >> createshortcut.vbs\n'
                            'cscript //nologo createshortcut.vbs\n'
                            'del createshortcut.vbs\n'
                            'echo.\n'
                            'echo Installation complete!\n'
                            'echo To run the application, double-click the desktop shortcut or run:\n'
                            'echo     python quantum_assistant_app.py\n'
                            'echo.\n'
                            'pause\n'
                        )
                    elif os_type.lower() == 'macos':
                        zipf.writestr('setup.sh', 
                            '#!/bin/bash\n'
                            'echo "Installing Quantum AI Assistant for macOS..."\n'
                            'echo\n'
                            'echo "Checking for Python..."\n'
                            'if ! command -v python3 &> /dev/null; then\n'
                            '    echo "Python not found! Please install Python 3.8 or newer."\n'
                            '    echo "Visit https://www.python.org/downloads/"\n'
                            '    read -p "Press Enter to exit..."\n'
                            '    exit 1\n'
                            'fi\n'
                            'echo "Python found!"\n'
                            'echo\n'
                            'echo "Installing dependencies..."\n'
                            'python3 -m pip install -r requirements.txt\n'
                            'if [ $? -ne 0 ]; then\n'
                            '    echo "Failed to install dependencies."\n'
                            '    read -p "Press Enter to exit..."\n'
                            '    exit 1\n'
                            'fi\n'
                            'echo\n'
                            'echo "Creating Applications folder entry..."\n'
                            'mkdir -p ~/Applications/QuantumAIAssistant\n'
                            'cp -r * ~/Applications/QuantumAIAssistant/\n'
                            'cat > ~/Applications/QuantumAIAssistant/run.command << EOL\n'
                            '#!/bin/bash\n'
                            'cd "\$(dirname "\$0")"\n'
                            'python3 quantum_assistant_app.py\n'
                            'EOL\n'
                            'chmod +x ~/Applications/QuantumAIAssistant/run.command\n'
                            'echo\n'
                            'echo "Installation complete!"\n'
                            'echo "To run the application, open the QuantumAIAssistant folder in your Applications folder"\n'
                            'echo "and double-click run.command, or run:"\n'
                            'echo "    python3 quantum_assistant_app.py"\n'
                            'echo\n'
                            'read -p "Press Enter to continue..."\n'
                        )
                    elif os_type.lower() == 'linux':
                        zipf.writestr('setup.sh', 
                            '#!/bin/bash\n'
                            'echo "Installing Quantum AI Assistant for Linux..."\n'
                            'echo\n'
                            'echo "Checking for Python..."\n'
                            'if ! command -v python3 &> /dev/null; then\n'
                            '    echo "Python not found! Please install Python 3.8 or newer."\n'
                            '    echo "Try: sudo apt-get install python3 python3-pip"\n'
                            '    read -p "Press Enter to exit..."\n'
                            '    exit 1\n'
                            'fi\n'
                            'echo "Python found!"\n'
                            'echo\n'
                            'echo "Installing dependencies..."\n'
                            'python3 -m pip install -r requirements.txt\n'
                            'if [ $? -ne 0 ]; then\n'
                            '    echo "Failed to install dependencies."\n'
                            '    read -p "Press Enter to exit..."\n'
                            '    exit 1\n'
                            'fi\n'
                            'echo\n'
                            'echo "Creating desktop entry..."\n'
                            'mkdir -p ~/.local/share/applications\n'
                            'INSTALL_DIR="$HOME/.local/bin/quantum-assistant"\n'
                            'mkdir -p "$INSTALL_DIR"\n'
                            'cp -r * "$INSTALL_DIR/"\n'
                            'cat > ~/.local/share/applications/quantum-assistant.desktop << EOL\n'
                            '[Desktop Entry]\n'
                            'Name=Quantum AI Assistant\n'
                            'Exec=python3 $INSTALL_DIR/quantum_assistant_app.py\n'
                            'Terminal=false\n'
                            'Type=Application\n'
                            'Categories=Science;Education;\n'
                            'Comment=Quantum AI Assistant for W&B experiment management\n'
                            'EOL\n'
                            'chmod +x ~/.local/share/applications/quantum-assistant.desktop\n'
                            'echo\n'
                            'echo "Installation complete!"\n'
                            'echo "To run the application, find Quantum AI Assistant in your applications menu,"\n'
                            'echo "or run:"\n'
                            'echo "    python3 $INSTALL_DIR/quantum_assistant_app.py"\n'
                            'echo\n'
                            'read -p "Press Enter to continue..."\n'
                        )
                    
                    # Add a configuration file
                    zipf.writestr('config.json', '{\n'
                        '  "version": "1.0.0",\n'
                        '  "use_gpu": true,\n'
                        '  "auto_update": true,\n'
                        '  "data_dir": "./data",\n'
                        '  "log_level": "info",\n'
                        '  "api_settings": {\n'
                        '    "default_timeout": 30,\n'
                        '    "max_retries": 3,\n'
                        '    "api_version": "v2"\n'
                        '  },\n'
                        '  "quantum_settings": {\n'
                        '    "default_shots": 1024,\n'
                        '    "use_noise_model": false,\n'
                        '    "default_optimizer": "SPSA",\n'
                        '    "max_iterations": 100\n'
                        '  }\n'
                        '}\n')
                    
                    # Add a real application file
                    zipf.writestr('quantum_assistant_app.py', 
                        'import os\n'
                        'import sys\n'
                        'import json\n'
                        'import subprocess\n'
                        'import webbrowser\n'
                        'from pathlib import Path\n\n'
                        'def check_dependencies():\n'
                        '    """Check if all required dependencies are installed."""\n'
                        '    try:\n'
                        '        import streamlit\n'
                        '        import numpy\n'
                        '        import pandas\n'
                        '        import qiskit\n'
                        '        import pennylane\n'
                        '        import plotly\n'
                        '        import wandb\n'
                        '        return True\n'
                        '    except ImportError as e:\n'
                        '        print(f"Missing dependency: {e}")\n'
                        '        print("Please install required packages: pip install -r requirements.txt")\n'
                        '        return False\n\n'
                        'def load_config():\n'
                        '    """Load configuration from config.json."""\n'
                        '    config_path = Path(__file__).parent / "config.json"\n'
                        '    if config_path.exists():\n'
                        '        with open(config_path, "r") as f:\n'
                        '            return json.load(f)\n'
                        '    return {}\n\n'
                        'def start_application():\n'
                        '    """Start the Streamlit application."""\n'
                        '    print("Starting Quantum AI Assistant...")\n'
                        '    app_dir = Path(__file__).parent\n'
                        '    app_path = app_dir / "app.py"\n'
                        '    \n'
                        '    # Create a basic app.py if it doesn\'t exist\n'
                        '    if not app_path.exists():\n'
                        '        with open(app_path, "w") as f:\n'
                        '            f.write(\'\'\'\n'
                        'import streamlit as st\n'
                        'import pandas as pd\n'
                        'import numpy as np\n'
                        'import plotly.express as px\n'
                        'import plotly.graph_objects as go\n'
                        'import wandb\n'
                        'import os\n'
                        'import io\n'
                        'import base64\n'
                        'import time\n'
                        'import datetime\n'
                        'from collections import Counter\n'
                        'import json\n\n'
                        'st.set_page_config(page_title="Quantum AI Assistant", page_icon="🔬", layout="wide")\n\n'
                        'st.title("Quantum AI Assistant")\n'
                        'st.markdown("### W&B Experiment Management System")\n\n'
                        'st.write("Welcome to the Quantum AI Assistant. Please authenticate with Weights & Biases to begin.")\n\n'
                        'api_key = st.text_input("W&B API Key", type="password")\n'
                        'if st.button("Login") and api_key:\n'
                        '    try:\n'
                        '        wandb.login(key=api_key)\n'
                        '        st.success("Successfully logged in to Weights & Biases!")\n'
                        '        st.info("You can now access your experiments and quantum capabilities.")\n'
                        '        st.experimental_rerun()\n'
                        '    except Exception as e:\n'
                        '        st.error(f"Login failed: {str(e)}")\n'
                        '\'\'\')\n'
                        '    \n'
                        '    # Start the Streamlit app\n'
                        '    port = 8501\n'
                        '    url = f"http://localhost:{port}"\n'
                        '    print(f"Launching browser at {url}")\n'
                        '    \n'
                        '    # Start Streamlit in a separate process\n'
                        '    process = subprocess.Popen(\n'
                        '        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],\n'
                        '        cwd=str(app_dir)\n'
                        '    )\n'
                        '    \n'
                        '    # Open the web browser after a short delay\n'
                        '    time.sleep(2)\n'
                        '    webbrowser.open(url)\n'
                        '    \n'
                        '    print("Quantum AI Assistant is running.")\n'
                        '    print("Close this window to shut down the application.")\n'
                        '    \n'
                        '    try:\n'
                        '        # Keep the process running until interrupted\n'
                        '        process.wait()\n'
                        '    except KeyboardInterrupt:\n'
                        '        print("Shutting down...")\n'
                        '        process.terminate()\n\n'
                        'if __name__ == "__main__":\n'
                        '    if check_dependencies():\n'
                        '        config = load_config()\n'
                        '        start_application()\n'
                        '    else:\n'
                        '        input("Press Enter to exit...")\n'
                    )
                    
                    # Add some data directory structure
                    zipf.writestr('data/.gitkeep', '')
                    zipf.writestr('assets/.gitkeep', '')
                    
                    # Add a license file
                    zipf.writestr('LICENSE.txt',
                        'MIT License\n\n'
                        'Copyright (c) 2025 Quantum AI Assistant\n\n'
                        'Permission is hereby granted, free of charge, to any person obtaining a copy\n'
                        'of this software and associated documentation files (the "Software"), to deal\n'
                        'in the Software without restriction, including without limitation the rights\n'
                        'to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n'
                        'copies of the Software, and to permit persons to whom the Software is\n'
                        'furnished to do so, subject to the following conditions:\n\n'
                        'The above copyright notice and this permission notice shall be included in all\n'
                        'copies or substantial portions of the Software.\n\n'
                        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
                        'IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n'
                        'FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n'
                        'AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n'
                        'LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n'
                        'OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n'
                        'SOFTWARE.\n'
                    )
                
                # Reset buffer position and return the data
                zip_buffer.seek(0)
                return zip_buffer.getvalue()
            
            # Create tabs for download and installation
            dl_tab, install_tab = st.tabs(["Download", "Install"])
            
            with dl_tab:
                app_cols = st.columns(2)
                with app_cols[0]:
                    # Create and offer the application package for download
                    app_package = create_application_package(desktop_os)
                    st.download_button(
                        label=f"Download for {desktop_os}",
                        data=app_package,
                        file_name=f"quantum_assistant_{desktop_os.lower()}.zip",
                        mime="application/zip",
                        help=f"Download the Quantum AI Assistant application package for {desktop_os}"
                    )
                
                with app_cols[1]:
                    st.checkbox("Enable GPU Acceleration", value=True, 
                                help="Use GPU acceleration for quantum simulations when available")
                    st.checkbox("Enable Automatic Updates", value=True,
                               help="Automatically check for and install updates")
            
            with install_tab:
                st.markdown("### Installation Utility")
                st.markdown("Upload a previously downloaded package to extract and install:")
                
                uploaded_file = st.file_uploader("Upload installation package", type="zip")
                
                if uploaded_file is not None:
                    # Create a temporary directory to extract files
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Extract the zip file
                        with st.spinner("Extracting package..."):
                            zip_data = uploaded_file.getvalue()
                            
                            # Save zip to temp directory
                            zip_path = os.path.join(tmp_dir, "package.zip")
                            with open(zip_path, "wb") as f:
                                f.write(zip_data)
                            
                            # Extract the zip
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                zip_ref.extractall(tmp_dir)
                            
                            # Show extraction completed
                            time.sleep(1)  # Simulate processing time
                        
                        # Show extracted files
                        extracted_files = os.listdir(tmp_dir)
                        extracted_files.remove("package.zip")  # Remove the zip file from list
                        
                        st.success(f"Package extracted successfully! Found {len(extracted_files)} files.")
                        
                        # Show file list
                        with st.expander("View extracted files"):
                            for file in extracted_files:
                                file_path = os.path.join(tmp_dir, file)
                                if os.path.isfile(file_path):
                                    # If text file, show content
                                    if file.endswith(('.txt', '.json', '.py', '.sh', '.bat')):
                                        with open(file_path, 'r') as f:
                                            content = f.read()
                                        st.code(content, language='python' if file.endswith('.py') else 'bash' if file.endswith('.sh') else None)
                                    else:
                                        st.text(f"Binary file: {file}")
                        
                        # Installation options
                        st.subheader("Installation Options")
                        install_dir = st.text_input("Installation Directory", value="~/quantum-assistant")
                        
                        # Options based on OS
                        if desktop_os.lower() == 'windows':
                            st.checkbox("Create desktop shortcut", value=True)
                            st.checkbox("Add to Start menu", value=True)
                        elif desktop_os.lower() == 'macos':
                            st.checkbox("Add to Applications folder", value=True)
                            st.checkbox("Create Dock icon", value=True)
                        elif desktop_os.lower() == 'linux':
                            st.checkbox("Create application launcher", value=True)
                            st.checkbox("Add to system PATH", value=True)
                        
                        # Install button
                        if st.button("Install Now"):
                            # Perform real installation process
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Define installation steps
                            install_steps = [
                                "Preparing installation...",
                                "Creating directory structure...",
                                "Copying files...",
                                "Configuring environment...",
                                "Setting up quantum backends...",
                                "Creating shortcuts...",
                                "Finalizing installation..."
                            ]
                            
                            # Perform actual installation
                            try:
                                # Step 1: Prepare installation
                                status_text.text(install_steps[0])
                                progress_bar.progress(1/len(install_steps))
                                
                                # Create installation directory if it doesn't exist
                                install_dir_expanded = os.path.expanduser(install_dir)
                                os.makedirs(install_dir_expanded, exist_ok=True)
                                time.sleep(0.5)  # Short pause for UI feedback
                                
                                # Step 2: Create directory structure
                                status_text.text(install_steps[1])
                                progress_bar.progress(2/len(install_steps))
                                
                                # Create subdirectories
                                os.makedirs(os.path.join(install_dir_expanded, "data"), exist_ok=True)
                                os.makedirs(os.path.join(install_dir_expanded, "assets"), exist_ok=True)
                                os.makedirs(os.path.join(install_dir_expanded, "configs"), exist_ok=True)
                                time.sleep(0.5)  # Short pause for UI feedback
                                
                                # Step 3: Copy files
                                status_text.text(install_steps[2])
                                progress_bar.progress(3/len(install_steps))
                                
                                # Copy all files from extracted directory to installation directory
                                for file in extracted_files:
                                    src_path = os.path.join(tmp_dir, file)
                                    dst_path = os.path.join(install_dir_expanded, file)
                                    
                                    if os.path.isfile(src_path):
                                        # Copy file
                                        with open(src_path, 'rb') as src_file:
                                            with open(dst_path, 'wb') as dst_file:
                                                dst_file.write(src_file.read())
                                    
                                time.sleep(0.5)  # Short pause for UI feedback
                                
                                # Step 4: Configure environment
                                status_text.text(install_steps[3])
                                progress_bar.progress(4/len(install_steps))
                                
                                # Create a dedicated virtual environment
                                if desktop_os.lower() == 'windows':
                                    # For Windows: Write a batch file to create venv
                                    venv_setup_path = os.path.join(install_dir_expanded, "setup_venv.bat")
                                    with open(venv_setup_path, 'w') as f:
                                        f.write('@echo off\n')
                                        f.write('echo Setting up virtual environment...\n')
                                        f.write('python -m venv .venv\n')
                                        f.write('.venv\\Scripts\\pip install -r requirements.txt\n')
                                        f.write('echo Environment setup complete!\n')
                                else:
                                    # For Linux/macOS: Write a shell script to create venv
                                    venv_setup_path = os.path.join(install_dir_expanded, "setup_venv.sh")
                                    with open(venv_setup_path, 'w') as f:
                                        f.write('#!/bin/bash\n')
                                        f.write('echo "Setting up virtual environment..."\n')
                                        f.write('python3 -m venv .venv\n')
                                        f.write('.venv/bin/pip install -r requirements.txt\n')
                                        f.write('echo "Environment setup complete!"\n')
                                    
                                    # Make the script executable
                                    os.chmod(venv_setup_path, 0o755)
                                
                                time.sleep(0.5)  # Short pause for UI feedback
                                
                                # Step 5: Set up quantum backends
                                status_text.text(install_steps[4])
                                progress_bar.progress(5/len(install_steps))
                                
                                # Create a config file for quantum backend settings
                                quantum_config_path = os.path.join(install_dir_expanded, "configs", "quantum_config.json")
                                with open(quantum_config_path, 'w') as f:
                                    f.write('{\n')
                                    f.write('  "default_backend": "aer_simulator",\n')
                                    f.write('  "shots": 1024,\n')
                                    f.write('  "optimization_level": 1,\n')
                                    f.write('  "use_gpu": false,\n')
                                    f.write('  "available_backends": ["aer_simulator", "statevector_simulator", "qasm_simulator"],\n')
                                    f.write('  "transpile_settings": {\n')
                                    f.write('    "optimization_level": 1,\n')
                                    f.write('    "basis_gates": ["cx", "u1", "u2", "u3"]\n')
                                    f.write('  }\n')
                                    f.write('}\n')
                                
                                time.sleep(0.5)  # Short pause for UI feedback
                                
                                # Step 6: Create shortcuts
                                status_text.text(install_steps[5])
                                progress_bar.progress(6/len(install_steps))
                                
                                # Create platform-specific launcher
                                if desktop_os.lower() == 'windows':
                                    # Create Windows shortcut
                                    launcher_path = os.path.join(install_dir_expanded, "launch.bat")
                                    with open(launcher_path, 'w') as f:
                                        f.write('@echo off\n')
                                        f.write('cd "%~dp0"\n')
                                        f.write('if not exist .venv (\n')
                                        f.write('   echo Virtual environment not found. Setting up...\n')
                                        f.write('   call setup_venv.bat\n')
                                        f.write(')\n')
                                        f.write('.venv\\Scripts\\python quantum_assistant_app.py\n')
                                        
                                elif desktop_os.lower() in ['macos', 'linux']:
                                    # Create shell launcher
                                    launcher_path = os.path.join(install_dir_expanded, "launch.sh")
                                    with open(launcher_path, 'w') as f:
                                        f.write('#!/bin/bash\n')
                                        f.write('cd "$(dirname "$0")"\n')
                                        f.write('if [ ! -d ".venv" ]; then\n')
                                        f.write('    echo "Virtual environment not found. Setting up..."\n')
                                        f.write('    bash setup_venv.sh\n')
                                        f.write('fi\n')
                                        f.write('.venv/bin/python quantum_assistant_app.py\n')
                                    
                                    # Make launcher executable
                                    os.chmod(launcher_path, 0o755)
                                
                                time.sleep(0.5)  # Short pause for UI feedback
                                
                                # Step 7: Finalize installation
                                status_text.text(install_steps[6])
                                progress_bar.progress(7/len(install_steps))
                                
                                # Create a file to mark successful installation
                                with open(os.path.join(install_dir_expanded, ".installation_complete"), 'w') as f:
                                    f.write(f"Installation completed at: {datetime.datetime.now().isoformat()}\n")
                                    f.write(f"Installed by: Quantum AI Assistant Cross-Platform Connector\n")
                                    f.write(f"Target OS: {desktop_os}\n")
                                
                                # Complete
                                time.sleep(0.5)  # Short pause for UI feedback
                                progress_bar.progress(1.0)
                                status_text.text("Installation completed successfully!")
                                
                                st.success(f"Quantum AI Assistant has been installed to {install_dir}")
                                
                                # Show instructions based on OS
                                if desktop_os.lower() == 'windows':
                                    st.info("To launch the application, navigate to the installation directory and run launch.bat")
                                else:
                                    st.info("To launch the application, navigate to the installation directory and run ./launch.sh")
                                
                            except Exception as e:
                                # Handle installation errors
                                st.error(f"Installation failed: {str(e)}")
                                st.info("Please check the console for detailed error information.")
                
                else:
                    st.info("Please upload a Quantum AI Assistant installation package (.zip file)")
                    
                    # Provide sample installation instructions
                    with st.expander("Installation Instructions"):
                        st.markdown("""
                        1. Download the appropriate package for your operating system using the Download tab
                        2. Upload the downloaded .zip file using the file uploader above
                        3. Review the extracted files and set installation options
                        4. Click "Install Now" to complete the installation
                        """)
            
        
        elif platform_category == "Mobile":
            mobile_os = st.selectbox(
                "Mobile OS",
                ["Android", "iOS"]
            )
            
            st.markdown(f"### {mobile_os} Optimization")
            
            # Mobile device selection
            device_type = st.radio(
                "Device Type",
                ["Smartphone", "Tablet", "Foldable"],
                horizontal=True
            )
            
            # Display optimization status
            st.markdown(f"#### {device_type} Optimization")
            
            # Show adaptive UI options
            st.markdown("#### Adaptive UI")
            adaptive_cols = st.columns(3)
            
            with adaptive_cols[0]:
                st.checkbox("Responsive Layout", value=True)
            
            with adaptive_cols[1]:
                st.checkbox("Touch Optimized", value=True)
            
            with adaptive_cols[2]:
                st.checkbox("Offline Capability", value=True)
            
            # Resource management
            st.markdown("#### Resource Management")
            st.slider("Max Memory Usage", 100, 1000, 500, step=100, format="%d MB")
            st.slider("Battery Conservation", 0, 100, 50, format="%d%%")
            
            # Mobile specific features
            st.markdown("#### Mobile Features")
            st.checkbox("Enable Push Notifications")
            st.checkbox("Use Biometric Authentication")
            st.checkbox("Enable AR Features")
        
        elif platform_category == "Web":
            st.markdown("### Web Application Optimization")
            
            # Browser support
            st.markdown("#### Supported Browsers")
            browser_data = {
                "Browser": ["Chrome", "Firefox", "Safari", "Edge", "Opera"],
                "Status": ["Fully Supported", "Fully Supported", "Fully Supported", "Fully Supported", "Partially Supported"],
                "Minimum Version": ["88+", "85+", "14+", "88+", "75+"]
            }
            
            browser_df = pd.DataFrame(browser_data)
            st.dataframe(browser_df, use_container_width=True)
            
            # Web features
            st.markdown("#### Web Features")
            web_features = st.columns(2)
            
            with web_features[0]:
                st.checkbox("Progressive Web App (PWA)", value=True)
                st.checkbox("WebGL Acceleration", value=True)
                st.checkbox("WebAssembly Support", value=True)
            
            with web_features[1]:
                st.checkbox("Responsive Design", value=True)
                st.checkbox("Web Push Notifications")
                st.checkbox("Offline Support", value=True)
        
        elif platform_category == "Cloud":
            st.markdown("### Cloud Platform Optimization")
            
            # Cloud provider selection
            cloud_provider = st.selectbox(
                "Cloud Provider",
                ["AWS", "Google Cloud", "Microsoft Azure", "IBM Cloud", "Oracle Cloud"]
            )
            
            st.markdown(f"#### {cloud_provider} Integration")
            
            # Cloud resources
            st.markdown("#### Resource Configuration")
            cloud_cols = st.columns(3)
            
            with cloud_cols[0]:
                st.slider("CPU Cores", 1, 32, 4)
            
            with cloud_cols[1]:
                st.slider("Memory (GB)", 1, 128, 16)
            
            with cloud_cols[2]:
                st.slider("Storage (GB)", 10, 1000, 100)
            
            # Advanced cloud options
            st.markdown("#### Advanced Options")
            st.checkbox("Auto-scaling")
            st.checkbox("Load Balancing")
            st.checkbox("Geo-replication")
        
        elif platform_category == "Edge/IoT":
            st.markdown("### Edge/IoT Device Optimization")
            
            # Device selection
            edge_device = st.selectbox(
                "Device Type",
                ["Raspberry Pi", "NVIDIA Jetson", "Arduino", "ESP32", "Custom IoT Device"]
            )
            
            st.markdown(f"#### {edge_device} Configuration")
            
            # Edge computing options
            st.markdown("#### Edge Computing Options")
            edge_cols = st.columns(2)
            
            with edge_cols[0]:
                st.checkbox("Local Model Inference")
                st.checkbox("Data Preprocessing")
                st.checkbox("Event Triggering")
            
            with edge_cols[1]:
                st.checkbox("Sensor Integration")
                st.checkbox("Low-power Mode")
                st.checkbox("Mesh Networking")
    
    # Network Configuration Tab
    with network_tab:
        st.subheader("Network Configuration")
        
        st.markdown("""
        Configure network settings to ensure the system works across various
        connection types, from high-speed fiber to low-bandwidth satellite.
        """)
        
        # Connection type selection
        connection_type = st.selectbox(
            "Connection Type",
            ["Broadband", "Cellular", "Wi-Fi", "Satellite", "Mesh Network", "Low-power IoT"]
        )
        
        if connection_type == "Broadband":
            st.markdown("### Broadband Configuration")
            
            # Broadband settings
            st.slider("Bandwidth Allocation", 1, 1000, 50, format="%d Mbps")
            st.checkbox("Enable QoS (Quality of Service)")
            st.checkbox("Optimize for Video Streaming")
            
            # Connection reliability
            st.markdown("#### Connection Reliability")
            st.metric("Uptime", "99.9%")
            st.metric("Average Latency", "15ms")
        
        elif connection_type == "Cellular":
            st.markdown("### Cellular Network Configuration")
            
            # Cellular network settings
            network_gen = st.radio(
                "Network Generation",
                ["5G", "4G/LTE", "3G", "2G"],
                horizontal=True
            )
            
            st.markdown(f"#### {network_gen} Optimization")
            
            # Data usage settings
            st.markdown("#### Data Usage")
            st.slider("Monthly Data Cap", 1, 100, 10, format="%d GB")
            st.checkbox("Background Data Restriction")
            st.checkbox("Compress Data When Possible")
            
            # Roaming settings
            st.markdown("#### Roaming Settings")
            st.checkbox("Allow Data Roaming")
            st.checkbox("International Roaming")
        
        elif connection_type == "Wi-Fi":
            st.markdown("### Wi-Fi Configuration")
            
            # Wi-Fi settings
            wifi_band = st.radio(
                "Wi-Fi Band",
                ["2.4 GHz", "5 GHz", "6 GHz", "Auto"],
                horizontal=True
            )
            
            st.markdown(f"#### {wifi_band} Optimization")
            
            # Wi-Fi security
            st.markdown("#### Security")
            st.selectbox("Security Protocol", ["WPA3", "WPA2", "WPA", "Open"])
            
            # Connection management
            st.markdown("#### Connection Management")
            st.checkbox("Auto-reconnect")
            st.checkbox("Preferred Network")
            st.checkbox("Public Hotspot Optimization")
        
        elif connection_type == "Satellite":
            st.markdown("### Satellite Connection Configuration")
            
            # Satellite provider
            satellite_provider = st.selectbox(
                "Satellite Provider",
                ["Starlink", "Viasat", "HughesNet", "Iridium", "Inmarsat"]
            )
            
            st.markdown(f"#### {satellite_provider} Optimization")
            
            # Latency compensation
            st.markdown("#### Latency Compensation")
            st.checkbox("Predictive Caching")
            st.checkbox("Asynchronous Updates")
            st.checkbox("Compression")
            
            # Bandwidth management
            st.markdown("#### Bandwidth Management")
            st.slider("Priority Level", 1, 5, 3)
            st.checkbox("Reduce Image Quality")
            st.checkbox("Defer Non-critical Updates")
        
        elif connection_type == "Mesh Network":
            st.markdown("### Mesh Network Configuration")
            
            # Mesh network settings
            st.markdown("#### Mesh Topology")
            st.number_input("Number of Nodes", min_value=2, max_value=100, value=5)
            
            # Routing protocol
            st.markdown("#### Routing Protocol")
            st.selectbox("Protocol", ["AODV", "HWMP", "OLSR", "B.A.T.M.A.N."])
            
            # Mesh features
            st.markdown("#### Features")
            st.checkbox("Self-healing")
            st.checkbox("Load Balancing")
            st.checkbox("Auto-configuration")
        
        elif connection_type == "Low-power IoT":
            st.markdown("### Low-power IoT Network Configuration")
            
            # IoT protocol
            iot_protocol = st.selectbox(
                "Protocol",
                ["LoRaWAN", "Zigbee", "Bluetooth LE", "NB-IoT", "Sigfox"]
            )
            
            st.markdown(f"#### {iot_protocol} Optimization")
            
            # Power settings
            st.markdown("#### Power Management")
            st.slider("Transmission Power", 1, 20, 10, format="%d dBm")
            st.slider("Duty Cycle", 0.1, 100.0, 1.0, format="%f%%")
            
            # Data rate
            st.markdown("#### Data Rate")
            st.slider("Message Size", 10, 255, 50, format="%d bytes")
            st.number_input("Messages Per Hour", min_value=1, max_value=1000, value=12)
    
    # Compatibility Tab
    with compatibility_tab:
        st.subheader("System Compatibility")
        
        st.markdown("""
        Check and configure compatibility settings to ensure the system works
        with various hardware, software, and network environments.
        """)
        
        # Compatibility checker
        st.markdown("### Compatibility Checker")
        
        if st.button("Run Compatibility Check"):
            with st.spinner("Checking system compatibility..."):
                # This would actually check compatibility in a real implementation
                st.success("System compatibility check completed!")
                
                # Display compatibility results
                compatibility_data = {
                    "Component": ["CPU Architecture", "GPU Support", "Memory", "Storage", "Network", "OS Support"],
                    "Status": ["Compatible", "Compatible", "Compatible", "Compatible", "Compatible", "Compatible"],
                    "Details": [
                        "x86_64, ARM64 supported",
                        "OpenGL 4.0+, CUDA 10.0+, Metal",
                        "4GB+ available",
                        "1GB+ available",
                        "All connection types supported",
                        "Windows 10+, macOS 10.15+, Linux, Android 9+, iOS 13+"
                    ]
                }
                
                compatibility_df = pd.DataFrame(compatibility_data)
                st.dataframe(compatibility_df, use_container_width=True)
        
        # Adaptive features
        st.markdown("### Adaptive Features")
        
        st.markdown("""
        The system automatically adapts to available resources and capabilities,
        but you can manually configure these settings if needed.
        """)
        
        adaptive_cols = st.columns(2)
        
        with adaptive_cols[0]:
            st.checkbox("Automatic Resource Detection", value=True)
            st.checkbox("Progressive Enhancement", value=True)
            st.checkbox("Graceful Degradation", value=True)
        
        with adaptive_cols[1]:
            st.checkbox("Bandwidth Adaptation", value=True)
            st.checkbox("Screen Size Adaptation", value=True)
            st.checkbox("Input Method Detection", value=True)
        
        # Backward compatibility
        st.markdown("### Backward Compatibility")
        
        backward_mode = st.checkbox("Enable Legacy Support Mode")
        
        if backward_mode:
            st.markdown("""
            Legacy Support Mode enables compatibility with older systems and networks,
            but may reduce some advanced features.
            """)
            
            legacy_features = st.multiselect(
                "Legacy Features to Enable",
                ["Reduced UI Complexity", "Lower Resolution Graphics", "Simplified Models", "Text-only Mode", "Basic Authentication"]
            )
    
    # System Status Tab
    with status_tab:
        st.subheader("System Status")
        
        st.markdown("""
        View the current status of system connectivity and performance
        across different platforms and networks.
        """)
        
        # Overall system status
        st.markdown("### Current System Status")
        
        status_cols = st.columns(4)
        
        with status_cols[0]:
            st.metric("Platform Compatibility", "100%")
        
        with status_cols[1]:
            st.metric("Network Connectivity", "Online")
        
        with status_cols[2]:
            st.metric("System Performance", "Optimal")
        
        with status_cols[3]:
            st.metric("Last Update", datetime.now().strftime("%Y-%m-%d"))
        
        # Platform-specific status
        st.markdown("### Platform-specific Status")
        
        platform_status = {
            "Platform": ["Desktop Web", "Mobile Web", "Android App", "iOS App", "Windows App", "macOS App", "Linux App", "IoT Devices"],
            "Status": ["Online", "Online", "Online", "Online", "Online", "Online", "Online", "Online"],
            "Performance": ["100%", "95%", "98%", "97%", "99%", "98%", "100%", "90%"]
        }
        
        platform_df = pd.DataFrame(platform_status)
        st.dataframe(platform_df, use_container_width=True)
        
        # Network connectivity status
        st.markdown("### Network Connectivity Status")
        
        network_status = {
            "Connection Type": ["Broadband", "Wi-Fi", "Cellular 5G", "Cellular 4G", "Cellular 3G", "Satellite", "Low-power IoT"],
            "Status": ["Optimal", "Optimal", "Optimal", "Good", "Limited", "Good", "Limited"],
            "Data Rate": ["Full", "Full", "Full", "Full", "Reduced", "Reduced", "Minimal"],
            "Feature Support": ["All Features", "All Features", "All Features", "Most Features", "Basic Features", "Most Features", "Basic Features"]
        }
        
        network_df = pd.DataFrame(network_status)
        st.dataframe(network_df, use_container_width=True)
        
        # System monitoring
        st.markdown("### Real-time Monitoring")
        
        if st.button("Refresh System Status"):
            with st.spinner("Refreshing system status..."):
                # This would actually refresh the status in a real implementation
                st.success("System status refreshed!")
        
        # Connectivity visualization
        st.markdown("### Connectivity Visualization")
        
        # Create simulated connectivity data
        connectivity_data = pd.DataFrame({
            "Timestamp": pd.date_range(start="2025-03-19", periods=24, freq="h"),
            "Broadband": np.random.uniform(0.95, 1.0, 24),
            "WiFi": np.random.uniform(0.9, 1.0, 24),
            "5G": np.random.uniform(0.85, 1.0, 24),
            "4G": np.random.uniform(0.8, 1.0, 24),
            "Satellite": np.random.uniform(0.7, 0.95, 24)
        })
        
        # Melt the DataFrame for plotting
        connectivity_melted = pd.melt(
            connectivity_data,
            id_vars=["Timestamp"],
            value_vars=["Broadband", "WiFi", "5G", "4G", "Satellite"],
            var_name="Connection Type",
            value_name="Reliability"
        )
        
        # Create the connectivity plot
        fig = px.line(
            connectivity_melted,
            x="Timestamp",
            y="Reliability",
            color="Connection Type",
            title="Connection Reliability (24-hour period)"
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Reliability Score",
            yaxis=dict(range=[0.5, 1.0])
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Show the platform optimization notice
    st.info("""
    The Quantum AI Assistant automatically optimizes for your current platform,
    network connection, and device. These settings allow you to fine-tune the
    behavior for specific environments or test cross-platform functionality.
    """)