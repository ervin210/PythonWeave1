from setuptools import setup, find_packages

setup(
    name="quantum_ai_assistant",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.22.0",
        "qiskit>=1.0.0",
        "qiskit-aer>=0.12.0",
        "qiskit-ibm-runtime>=0.11.0",
        "pennylane>=0.30.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "plotly>=5.14.0",
        "wandb>=0.15.0",
        "pylatexenc>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "quantum-assistant=quantum_ai_assistant.app:main",
        ],
    },
)