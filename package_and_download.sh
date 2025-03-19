#!/bin/bash
# This script packages the Quantum AI Assistant and prepares it for download

# Make the scripts executable
chmod +x build_packages.py download_script.py run_quantum_assistant.py

# Create a distribution directory
mkdir -p dist

# Run the build script to create packages
echo "Building packages..."
python build_packages.py

# Create a ZIP archive of everything needed for easy download
echo "Creating download archive..."
zip -r dist/quantum_ai_assistant_complete.zip \
    dist/*.whl \
    run_quantum_assistant.py \
    download_script.py \
    INSTALL_GUIDE.md \
    README.md

# Print instructions
echo "="
echo "Package creation complete!"
echo "="
echo "To distribute the Quantum AI Assistant:"
echo "1. Share the 'dist/quantum_ai_assistant_complete.zip' file"
echo "2. Users can extract it and run download_script.py"
echo "3. Advanced users can directly run run_quantum_assistant.py"
echo "="