# Quantum AI Assistant Quick Start Guide

## Getting Started

The Quantum AI Assistant is now packaged for use on all devices. Follow these simple steps to get started:

### Option 1: Download and Install (Recommended for Most Users)

1. **Download** the `quantum_ai_assistant_complete.zip` package
2. **Extract** the ZIP file to a location on your computer
3. **Run** one of the following:
   - On Windows: Double-click `download_script.py`
   - On macOS/Linux: Open a terminal, navigate to the extracted folder, and run `python download_script.py`
4. Follow the on-screen instructions to complete the installation
5. Once installed, you'll have a desktop shortcut to launch the Quantum AI Assistant

### Option 2: Direct Run (For Advanced Users)

If you already have Python and the required packages installed:

1. **Download** the `quantum_ai_assistant_complete.zip` package
2. **Extract** the ZIP file to a location on your computer
3. **Run** one of the following:
   - On Windows: Double-click `run_quantum_assistant.py`
   - On macOS/Linux: Open a terminal, navigate to the extracted folder, and run `python run_quantum_assistant.py`
4. The application will start and open in your web browser

## Using on Mobile Devices

To use the Quantum AI Assistant on your mobile device:

1. **Install** the application on a computer on your local network using one of the methods above
2. When running the application, note the "Network URL" displayed in the terminal (looks like `http://192.168.x.x:5000`)
3. On your mobile device, open a web browser
4. Enter the Network URL in the address bar
5. You can now use the Quantum AI Assistant on your mobile device!

## Cloud Access (For Always-On Access)

For permanent access from any device:

1. Deploy the application to a cloud service (AWS, Google Cloud, etc.)
2. Access the application from any device using the provided URL

## IBM Quantum Integration

To use real quantum hardware:

1. Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)
2. Get your API token from your account settings
3. When running the application, you can provide your token using:
   ```
   python run_quantum_assistant.py --ibm-token YOUR_TOKEN_HERE
   ```
4. Or enter it in the application interface when prompted

## Need Help?

If you need assistance, please refer to the detailed `INSTALL_GUIDE.md` file for more information.