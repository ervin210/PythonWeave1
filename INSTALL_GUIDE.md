# Quantum AI Assistant Installation Guide

## System Requirements

- **Operating Systems**: Windows 10+, macOS 10.15+, Ubuntu 20.04+, or other Linux distributions
- **Processor**: Dual-core processor, 2.0 GHz or higher
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Disk Space**: 500MB free space
- **Python**: Version 3.8 or higher (if using the Python package)
- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **Internet Connection**: Required for IBM Quantum hardware access and W&B integration

## Installation Methods

Choose the installation method that works best for you:

### Method 1: Pre-built Executables (Recommended)

#### Windows Installation
1. Run the downloaded `download_script.py` file
2. Select the destination folder when prompted
3. Wait for the download to complete
4. A desktop shortcut will be created automatically
5. Double-click the shortcut to launch the application

#### macOS Installation
1. Run the downloaded `download_script.py` file using Terminal:
   ```
   python download_script.py
   ```
2. When the DMG file is downloaded, double-click to open it
3. Drag the Quantum AI Assistant icon to the Applications folder
4. Launch the app from your Applications folder
5. If you see a security warning, go to System Preferences > Security & Privacy, and click "Open Anyway"

#### Linux Installation
1. Run the downloaded `download_script.py` file using Terminal:
   ```
   python download_script.py
   ```
2. The AppImage file will be downloaded and made executable
3. You can launch it by double-clicking the AppImage file
4. A desktop shortcut will be created automatically

### Method 2: Python Package (For Developers)

1. Ensure you have Python 3.8+ and pip installed
2. Run the download script with the `--python-package` flag:
   ```
   python download_script.py --python-package
   ```
3. The script will install the package using pip
4. You can then run the assistant with:
   ```
   python -m quantum_assistant
   ```
   or simply:
   ```
   quantum-assistant
   ```

### Method 3: Source Installation (Advanced)

1. Clone the repository:
   ```
   git clone https://github.com/quantum-ai-assistant/quantum-ai-assistant.git
   ```
2. Navigate to the cloned directory:
   ```
   cd quantum-ai-assistant
   ```
3. Install dependencies:
   ```
   pip install -e .
   ```
4. Run the application:
   ```
   python run_quantum_assistant.py
   ```

## API Keys Setup

### IBM Quantum Account

To use real quantum hardware, you need an IBM Quantum account:

1. Sign up at [IBM Quantum](https://quantum-computing.ibm.com/)
2. After logging in, go to your account settings
3. Find your API token and copy it
4. Set it as an environment variable:
   ```
   export IBM_QUANTUM_TOKEN="your_token_here"
   ```
   Or provide it at runtime:
   ```
   python run_quantum_assistant.py --ibm-token your_token_here
   ```
   Or enter it in the application when prompted

### Weights & Biases (W&B) Account

To use W&B integration for experiment tracking:

1. Sign up at [Weights & Biases](https://wandb.ai/)
2. Get your API key from your account settings
3. Set it as an environment variable:
   ```
   export WANDB_API_KEY="your_api_key_here"
   ```
   Or provide it at runtime:
   ```
   python run_quantum_assistant.py --wandb-token your_api_key_here
   ```
   Or enter it in the application when prompted

## Troubleshooting

### Common Issues

**Application won't start**:
- Ensure you have the necessary permissions to run the application
- Check if any antivirus software is blocking the execution
- Verify that the port 5000 is not being used by another application

**Cannot connect to IBM Quantum**:
- Verify your API token is correct
- Check your internet connection
- Ensure your token has the necessary permissions

**Package installation fails**:
- Ensure you have the latest version of pip:
  ```
  python -m pip install --upgrade pip
  ```
- Try installing in a virtual environment:
  ```
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -e .
  ```

**Browser doesn't open automatically**:
- Open your browser manually and navigate to http://localhost:5000

### Getting Help

If you encounter any issues not covered here, please:

1. Check the [GitHub repository issues](https://github.com/quantum-ai-assistant/issues)
2. Join our [community forum](https://community.quantum-ai-assistant.com)
3. Contact support at support@quantum-ai-assistant.com