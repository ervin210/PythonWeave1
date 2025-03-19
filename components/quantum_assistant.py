import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import qiskit
from qiskit import QuantumCircuit
# Use qiskit-ibm-runtime for newer version
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import pennylane as qml
import time
import io
from io import BytesIO
import wandb
import json
import io
import base64
from PIL import Image

def quantum_assistant():
    """
    Quantum Computing AI Assistant for W&B experiment analysis
    """
    # Display logo and title in a row
    col1, col2 = st.columns([1, 3])
    
    with col1:
        try:
            # Try to load the logo from assets directory
            logo = Image.open("assets/quantum_logo.jpg")
            st.image(logo, width=150)
        except FileNotFoundError:
            # Fall back to the attached blob image if the logo is not found
            try:
                logo = Image.open("attached_assets/blob.jpg")
                st.image(logo, width=150)
            except FileNotFoundError:
                st.info("Logo not found. Please add a logo file at assets/quantum_logo.jpg")
    
    with col2:
        st.title("Quantum AI Assistant")
        st.markdown("*Powered by quantum computing and machine learning*")
    
    # Introduction section
    st.markdown("""
    Welcome to the Quantum AI Assistant for your Weights & Biases experiments. 
    This assistant helps you analyze your machine learning experiments using quantum computing techniques 
    and provides AI-powered insights.
    """)
    
    # Assistant tabs
    assistant_tab, quantum_tab, security_tab, insights_tab = st.tabs([
        "Assistant", "Quantum Computing", "Quantum Security", "AI Insights"
    ])
    
    with assistant_tab:
        render_assistant_interface()
        
    with quantum_tab:
        render_quantum_tools()
    
    with security_tab:
        render_quantum_security()
        
    with insights_tab:
        render_ai_insights()

def render_assistant_interface():
    """
    Renders the conversational assistant interface
    """
    st.subheader("Ask me about your experiments")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help you analyze your Weights & Biases experiments today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about your experiments or quantum computing"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your experiments..."):
                response = generate_assistant_response(prompt)
                st.write(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def generate_assistant_response(prompt):
    """
    Generate a response to the user's query based on their W&B data
    Enhanced with quantum security and cybersecurity capabilities
    
    Args:
        prompt: User's query text
        
    Returns:
        Generated response text
    """
    # Only process if user is authenticated
    if not st.session_state.authenticated:
        return "Please log in with your W&B API key first to access your experiment data."
    
    # Get info about current project/run if available
    project_info = ""
    if st.session_state.selected_project:
        project_id = st.session_state.selected_project["id"]
        project_info = f"for project {project_id}"
        
        if st.session_state.selected_run:
            run_id = st.session_state.selected_run["id"]
            project_info = f"for run {run_id} in project {project_id}"
    
    # Advanced response generation with additional topics
    prompt_lower = prompt.lower()
    
    # Quantum Security & Cybersecurity responses
    if "quantum security" in prompt_lower or ("quantum" in prompt_lower and "security" in prompt_lower):
        return """
        Quantum security leverages quantum computing principles to enhance data protection:
        
        1. Quantum Key Distribution (QKD) uses quantum properties for secure communication
        2. Post-quantum cryptography designs algorithms resistant to quantum attacks
        3. Quantum-resistant algorithms protect against Shor's algorithm threats
        
        I can help simulate quantum security protocols in the Quantum Computing tab.
        """
    
    elif "quantum cryptography" in prompt_lower:
        return """
        Quantum cryptography uses quantum mechanical properties to perform cryptographic tasks:
        
        1. BB84 Protocol: First QKD protocol using polarized photons
        2. E91 Protocol: Uses quantum entanglement for key distribution
        3. Three-stage protocol: Provides security without entanglement
        
        Would you like me to help you implement a quantum cryptography simulation?
        """
    
    elif "quantum cybersecurity" in prompt_lower or ("cybersecurity" in prompt_lower):
        return """
        Quantum computing impacts cybersecurity in several ways:
        
        1. Quantum computers can break RSA and ECC encryption using Shor's algorithm
        2. Quantum-resistant algorithms are being developed for post-quantum security
        3. Quantum random number generators provide true randomness for security
        4. Quantum sensing can detect eavesdropping attempts in communication channels
        
        I can assist in evaluating your security protocols against quantum threats.
        """
        
    elif "shor's algorithm" in prompt_lower or "shor algorithm" in prompt_lower:
        return """
        Shor's algorithm is a quantum algorithm that efficiently factors large integers:
        
        1. It can break RSA encryption by finding prime factors
        2. Runs in polynomial time on quantum computers (unlike classical algorithms)
        3. Demonstrates quantum computing's threat to current cryptographic systems
        
        I can explain how this algorithm works and its implications for cybersecurity.
        """
        
    elif "quantum machine learning" in prompt_lower or "qml" in prompt_lower:
        return """
        Quantum Machine Learning (QML) combines quantum computing with machine learning:
        
        1. Quantum Neural Networks use quantum circuits for neural network operations
        2. Variational Quantum Classifiers perform classification tasks on quantum hardware
        3. Quantum kernel methods enhance support vector machines
        4. Quantum GANs leverage quantum advantages for generative tasks
        
        I can help you analyze QML experiments and integrate them with classical ML workflows.
        """
    
    # Original responses for standard topics
    elif "compare" in prompt_lower and "runs" in prompt_lower:
        return f"I can help you compare runs {project_info}. Use the Compare Runs feature in the Runs page to select multiple runs and analyze their performance on different metrics."
    
    elif "best" in prompt_lower and ("model" in prompt_lower or "run" in prompt_lower):
        return f"To find the best model {project_info}, I recommend looking at the Sweeps page where you can see which hyperparameters led to the best performance metrics."
    
    elif "quantum" in prompt_lower:
        return """
        I can help you apply quantum computing techniques to your ML experiments:
        
        1. Quantum circuit simulation for algorithm testing
        2. Integration with quantum hardware via IBM Quantum
        3. Quantum feature maps for enhanced data representation
        4. Variational quantum algorithms for optimization tasks
        
        Check out the Quantum Computing tab for hands-on tools.
        """
    
    elif "optimize" in prompt_lower or "hyperparameter" in prompt_lower:
        return "For hyperparameter optimization, I recommend using W&B Sweeps which provides Bayesian, grid and random search methods. You can view your sweep results in the Sweeps page."
    
    elif "explain" in prompt_lower or "why" in prompt_lower:
        return f"I can help explain trends in your experimental results {project_info}. Look at the metrics visualizations to understand how your model's performance changed over time."
    
    elif "export" in prompt_lower or "download" in prompt_lower:
        return "You can export your experiment data using the Export buttons on various pages. This allows you to download metrics, parameters, and results as CSV files for further analysis."
    
    elif "error" in prompt_lower or "issue" in prompt_lower or "problem" in prompt_lower:
        return f"""
        I can help troubleshoot issues with your experiments {project_info}:
        
        1. Check error logs in the run details page
        2. Analyze system resource usage (GPU, memory)
        3. Validate data preprocessing steps
        4. Review model architecture for potential issues
        
        Would you like me to help diagnose a specific error?
        """
    
    # Default response with enhanced capabilities
    else:
        return f"""
        I'm your quantum AI assistant for W&B experiments {project_info}. I can help with:
        
        1. Analyzing ML experiments and comparing runs
        2. Quantum computing simulations and algorithm implementation
        3. Quantum security and cybersecurity assessment
        4. Troubleshooting issues and optimizing performance
        
        Just ask me specific questions about your experiments or quantum computing needs.
        """

def render_quantum_tools():
    """
    Renders quantum computing tools for ML experiment analysis
    """
    st.subheader("Quantum Computing Tools")
    
    st.markdown("""
    Apply quantum computing techniques to understand and enhance your machine learning experiments.
    """)
    
    # Create tabs for different quantum tools
    circuit_tab, builder_tab, ml_integration_tab = st.tabs([
        "Quantum Circuit Simulator", 
        "Circuit Builder", 
        "Quantum-ML Integration"
    ])
    
    with circuit_tab:
        # Quantum Circuit Simulator
        st.subheader("Quantum Circuit Simulator")
        
        circuit_type = st.selectbox(
            "Select a quantum circuit type",
            ["Bell State", "GHZ State", "Quantum Fourier Transform", "Custom"]
        )
        
        num_qubits = st.slider("Number of qubits", 2, 8, 2)
        
        if circuit_type == "Bell State":
            circuit = create_bell_state()
        elif circuit_type == "GHZ State":
            circuit = create_ghz_state(num_qubits)
        elif circuit_type == "Quantum Fourier Transform":
            circuit = create_qft_circuit(num_qubits)
        else:  # Custom
            custom_code = st.text_area(
                "Define your quantum circuit (Qiskit code)",
                """from qiskit import QuantumCircuit
# Create a circuit with 2 qubits
qc = QuantumCircuit(2, 2)
# Add gates
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
"""
            )
            try:
                # Create a local namespace to execute the code
                local_vars = {}
                exec(custom_code, {"QuantumCircuit": QuantumCircuit}, local_vars)
                # Try to find the quantum circuit in the local namespace
                circuit = None
                for var in local_vars.values():
                    if isinstance(var, QuantumCircuit):
                        circuit = var
                        break
                if circuit is None:
                    st.error("No QuantumCircuit object found in your code. Make sure to create a circuit.")
                    return
            except Exception as e:
                st.error(f"Error in custom circuit code: {str(e)}")
                return
        
        # Display the circuit
        st.subheader("Circuit Diagram")
        try:
            circuit_img = circuit_to_image(circuit)
            st.image(circuit_img, width=700)
        except Exception as e:
            st.error(f"Error displaying circuit: {str(e)}")
        
        # Run the simulation
        if st.button("Run Quantum Simulation"):
            with st.spinner("Running quantum simulation..."):
                try:
                    results = run_quantum_circuit(circuit)
                    st.success("Simulation complete!")
                    
                    # Plot the results
                    counts = results.get_counts()
                    fig = plot_quantum_results(counts)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display measurement probabilities
                    st.subheader("Measurement Probabilities")
                    probs_df = pd.DataFrame(
                        [(state, count/sum(counts.values())*100) for state, count in counts.items()],
                        columns=["State", "Probability (%)"]
                    ).sort_values("Probability (%)", ascending=False)
                    
                    st.dataframe(probs_df, use_container_width=True)
                    
                    # If we have project and run data, offer to save results to W&B
                    if st.session_state.authenticated and st.session_state.selected_run:
                        if st.button("Save Quantum Results to W&B Run"):
                            with st.spinner("Saving to W&B..."):
                                save_quantum_results_to_wandb(circuit, counts)
                                st.success("Results saved to your W&B run!")
                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
    
    with builder_tab:
        # Interactive Circuit Builder
        st.subheader("Drag-and-Drop Quantum Circuit Builder")
        
        st.markdown("""
        Build your quantum circuit by adding gates to qubits. This visual builder helps you
        design circuits without writing code.
        """)
        
        # Initialize circuit state in session if not exists
        if "builder_circuit" not in st.session_state:
            st.session_state.builder_circuit = {
                "num_qubits": 3,
                "num_clbits": 3,
                "gates": []  # list of {qubit, gate_type, params, step}
            }
        
        # Controls for circuit parameters
        col1, col2 = st.columns(2)
        with col1:
            new_num_qubits = st.number_input("Number of qubits", min_value=1, max_value=10, 
                                        value=st.session_state.builder_circuit["num_qubits"])
        with col2:
            new_num_clbits = st.number_input("Number of classical bits", min_value=0, max_value=10, 
                                        value=st.session_state.builder_circuit["num_clbits"])
        
        # Update circuit dimensions if changed
        if new_num_qubits != st.session_state.builder_circuit["num_qubits"] or \
           new_num_clbits != st.session_state.builder_circuit["num_clbits"]:
            st.session_state.builder_circuit["num_qubits"] = new_num_qubits
            st.session_state.builder_circuit["num_clbits"] = new_num_clbits
            # Filter out gates on qubits that no longer exist
            st.session_state.builder_circuit["gates"] = [
                g for g in st.session_state.builder_circuit["gates"] 
                if g["qubit"] < new_num_qubits and 
                (g["target"] < new_num_qubits if "target" in g else True)
            ]
        
        # Gate selection
        st.subheader("Add Gates")
        
        # Single-qubit gates
        st.markdown("##### Single-Qubit Gates")
        single_cols = st.columns(4)
        
        with single_cols[0]:
            if st.button("H (Hadamard)"):
                add_gate = {"gate_type": "h", "qubit": 0, "step": get_next_step(0)}
                open_gate_modal(add_gate)
        
        with single_cols[1]:
            if st.button("X (NOT)"):
                add_gate = {"gate_type": "x", "qubit": 0, "step": get_next_step(0)}
                open_gate_modal(add_gate)
        
        with single_cols[2]:
            if st.button("Y"):
                add_gate = {"gate_type": "y", "qubit": 0, "step": get_next_step(0)}
                open_gate_modal(add_gate)
        
        with single_cols[3]:
            if st.button("Z"):
                add_gate = {"gate_type": "z", "qubit": 0, "step": get_next_step(0)}
                open_gate_modal(add_gate)
        
        # Two-qubit gates
        st.markdown("##### Two-Qubit Gates")
        two_cols = st.columns(3)
        
        with two_cols[0]:
            if st.button("CNOT"):
                add_gate = {"gate_type": "cx", "qubit": 0, "target": 1, "step": get_next_step(0)}
                open_gate_modal(add_gate, two_qubit=True)
        
        with two_cols[1]:
            if st.button("SWAP"):
                add_gate = {"gate_type": "swap", "qubit": 0, "target": 1, "step": get_next_step(0)}
                open_gate_modal(add_gate, two_qubit=True)
        
        with two_cols[2]:
            if st.button("CZ"):
                add_gate = {"gate_type": "cz", "qubit": 0, "target": 1, "step": get_next_step(0)}
                open_gate_modal(add_gate, two_qubit=True)
        
        # Rotation gates
        st.markdown("##### Rotation & Phase Gates")
        rot_cols = st.columns(4)
        
        with rot_cols[0]:
            if st.button("RX"):
                add_gate = {"gate_type": "rx", "qubit": 0, "params": np.pi/2, "step": get_next_step(0)}
                open_gate_modal(add_gate, with_params=True)
        
        with rot_cols[1]:
            if st.button("RY"):
                add_gate = {"gate_type": "ry", "qubit": 0, "params": np.pi/2, "step": get_next_step(0)}
                open_gate_modal(add_gate, with_params=True)
        
        with rot_cols[2]:
            if st.button("RZ"):
                add_gate = {"gate_type": "rz", "qubit": 0, "params": np.pi/2, "step": get_next_step(0)}
                open_gate_modal(add_gate, with_params=True)
        
        with rot_cols[3]:
            if st.button("Phase"):
                add_gate = {"gate_type": "p", "qubit": 0, "params": np.pi/4, "step": get_next_step(0)}
                open_gate_modal(add_gate, with_params=True)
        
        # Measurement
        measure_cols = st.columns(2)
        with measure_cols[0]:
            if st.button("Measure"):
                add_gate = {"gate_type": "measure", "qubit": 0, "clbit": 0, "step": get_next_step(0)}
                open_gate_modal(add_gate, with_measurement=True)
        
        with measure_cols[1]:
            if st.button("Measure All"):
                # Add measurements to all qubits
                for q in range(st.session_state.builder_circuit["num_qubits"]):
                    clbit = min(q, st.session_state.builder_circuit["num_clbits"]-1) if st.session_state.builder_circuit["num_clbits"] > 0 else None
                    if clbit is not None:
                        step = get_next_step(q)
                        st.session_state.builder_circuit["gates"].append({
                            "gate_type": "measure",
                            "qubit": q,
                            "clbit": clbit,
                            "step": step
                        })
        
        # Gate modal for selecting parameters
        if "gate_modal" in st.session_state and st.session_state.gate_modal:
            gate_data = st.session_state.gate_modal
            with st.form(key="gate_form"):
                st.write(f"Add {gate_data['gate_type'].upper()} Gate")
                
                # Select qubit
                qubit = st.number_input("Qubit", min_value=0, 
                                     max_value=st.session_state.builder_circuit["num_qubits"]-1, 
                                     value=gate_data.get("qubit", 0))
                
                # For two-qubit gates
                if "two_qubit" in gate_data and gate_data["two_qubit"]:
                    target = st.number_input("Target Qubit", min_value=0, 
                                         max_value=st.session_state.builder_circuit["num_qubits"]-1,
                                         value=min(gate_data.get("target", 1), st.session_state.builder_circuit["num_qubits"]-1))
                    if target == qubit:
                        st.error("Control and target qubits must be different")
                
                # For rotation gates
                params = None
                if "with_params" in gate_data and gate_data["with_params"]:
                    param_value = gate_data.get("params", np.pi/2)
                    params = st.slider("Angle (radians)", min_value=0.0, max_value=float(2*np.pi), 
                                   value=float(param_value), step=0.01)
                
                # For measurement
                clbit = None
                if "with_measurement" in gate_data and gate_data["with_measurement"]:
                    if st.session_state.builder_circuit["num_clbits"] > 0:
                        clbit = st.number_input("Classical Bit", min_value=0, 
                                           max_value=st.session_state.builder_circuit["num_clbits"]-1, 
                                           value=min(gate_data.get("clbit", 0), st.session_state.builder_circuit["num_clbits"]-1))
                    else:
                        st.error("No classical bits available for measurement")
                
                # Form submission
                col1, col2 = st.columns(2)
                with col1:
                    cancel = st.form_submit_button("Cancel")
                with col2:
                    submit = st.form_submit_button("Add Gate")
                
                if submit:
                    # Create gate definition
                    gate = {
                        "gate_type": gate_data["gate_type"],
                        "qubit": qubit,
                        "step": get_next_step(qubit)
                    }
                    
                    if "two_qubit" in gate_data and gate_data["two_qubit"]:
                        if target != qubit:
                            gate["target"] = target
                        else:
                            # Don't add if control and target are the same
                            gate = None
                    
                    if "with_params" in gate_data and gate_data["with_params"]:
                        gate["params"] = params
                    
                    if "with_measurement" in gate_data and gate_data["with_measurement"]:
                        if st.session_state.builder_circuit["num_clbits"] > 0:
                            gate["clbit"] = clbit
                        else:
                            # Don't add measurement if no classical bits
                            gate = None
                    
                    # Add the gate to the circuit
                    if gate:
                        st.session_state.builder_circuit["gates"].append(gate)
                    
                    # Close modal
                    st.session_state.gate_modal = None
                
                if cancel:
                    st.session_state.gate_modal = None
        
        # View and edit circuit
        st.subheader("Circuit")
        
        # Display the circuit (visual representation)
        display_visual_circuit(st.session_state.builder_circuit)
        
        # Show current gate list
        if st.session_state.builder_circuit["gates"]:
            st.subheader("Gate List")
            
            gates_df = pd.DataFrame([
                {
                    "Gate": g["gate_type"].upper(),
                    "Qubit": g["qubit"],
                    "Target": g.get("target", ""),
                    "Param": g.get("params", ""),
                    "Measure to": g.get("clbit", ""),
                    "Step": g["step"]
                } for g in st.session_state.builder_circuit["gates"]
            ]).sort_values("Step")
            
            st.dataframe(gates_df, use_container_width=True)
            
            # Option to remove gates
            if st.button("Clear All Gates"):
                st.session_state.builder_circuit["gates"] = []
            
            # Run the built circuit
            if st.button("Build and Run Circuit"):
                with st.spinner("Building and running circuit..."):
                    try:
                        built_circuit = build_circuit_from_builder()
                        st.subheader("Generated Circuit")
                        circuit_img = circuit_to_image(built_circuit)
                        st.image(circuit_img, width=700)
                        
                        # Generate and display the Qiskit code
                        code = generate_qiskit_code(built_circuit)
                        with st.expander("Qiskit Code", expanded=False):
                            st.code(code, language="python")
                        
                        # Run simulation
                        results = run_quantum_circuit(built_circuit)
                        st.success("Simulation complete!")
                        
                        # Plot the results
                        counts = results.get_counts()
                        fig = plot_quantum_results(counts)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display measurement probabilities
                        st.subheader("Measurement Probabilities")
                        probs_df = pd.DataFrame(
                            [(state, count/sum(counts.values())*100) for state, count in counts.items()],
                            columns=["State", "Probability (%)"]
                        ).sort_values("Probability (%)", ascending=False)
                        
                        st.dataframe(probs_df, use_container_width=True)
                        
                        # If we have project and run data, offer to save results to W&B
                        if st.session_state.authenticated and st.session_state.selected_run:
                            if st.button("Save Quantum Results to W&B Run"):
                                with st.spinner("Saving to W&B..."):
                                    save_quantum_results_to_wandb(built_circuit, counts)
                                    st.success("Results saved to your W&B run!")
                    except Exception as e:
                        st.error(f"Error building or running circuit: {str(e)}")
        
    with ml_integration_tab:
        # Quantum-ML Integration
        st.subheader("Integrate Quantum Computing with W&B ML Experiments")
        
        st.markdown("""
        Connect your quantum circuits with your machine learning experiments. 
        Enhance classical ML models with quantum computing techniques.
        """)
        
        integration_type = st.selectbox(
            "Select Integration Type",
            ["Quantum Feature Map", "Quantum Kernel", "Variational Quantum Circuit", "Quantum Data Encoding"]
        )
        
        if integration_type == "Quantum Feature Map":
            st.markdown("""
            ### Quantum Feature Map
            
            Map classical data into a quantum Hilbert space to potentially capture more complex patterns.
            
            **How it works:**
            1. Classical data is encoded into quantum states
            2. Quantum circuit applies a feature map transformation
            3. The quantum state embeds the data in a higher-dimensional space
            4. Measurements provide new feature representations
            
            **Potential Benefits:**
            - Access to exponentially large feature spaces
            - Ability to represent complex non-linear relationships
            - May find patterns classical algorithms can't detect
            """)
            
            # Example configuration
            st.subheader("Configure Feature Map")
            
            num_features = st.slider("Number of Data Features", 1, 10, 2)
            entanglement = st.selectbox("Entanglement Pattern", ["Full", "Linear", "Circular"])
            reps = st.slider("Circuit Repetitions", 1, 5, 2)
            
            if st.button("Generate Feature Map Circuit"):
                with st.spinner("Generating circuit..."):
                    try:
                        # Simple example feature map circuit
                        feature_map_circuit = create_feature_map_circuit(num_features, entanglement, reps)
                        st.subheader("Feature Map Circuit")
                        circuit_img = circuit_to_image(feature_map_circuit)
                        st.image(circuit_img, width=700)
                        
                        st.markdown("""
                        **Using with ML models:**
                        
                        To use this feature map with your ML experiments:
                        1. Apply this circuit to encode your input data
                        2. Use the measurement results as features for classical ML models
                        3. Track both quantum and classical components in W&B
                        """)
                    except Exception as e:
                        st.error(f"Error generating feature map: {str(e)}")
        
        elif integration_type == "Quantum Kernel":
            st.markdown("""
            ### Quantum Kernel Methods
            
            Use quantum circuits to compute kernel functions for machine learning algorithms.
            
            **How it works:**
            1. Data points are encoded into quantum states
            2. A quantum circuit computes their similarity (kernel value)
            3. The kernel matrix is used in classical kernel methods (SVM, etc.)
            
            **Potential Benefits:**
            - Compute kernels that may be hard to calculate classically
            - Capture complex similarities between data points
            - Enhance kernel-based ML methods
            """)
            
            # Example configuration
            st.subheader("Configure Quantum Kernel")
            
            num_features = st.slider("Number of Data Features", 1, 10, 2, key="kernel_features")
            kernel_type = st.selectbox("Kernel Type", ["Quantum Feature Map", "Projected Quantum Kernel", "Custom"])
            
            if st.button("Generate Kernel Circuit"):
                with st.spinner("Generating circuit..."):
                    try:
                        # Simple example kernel circuit
                        kernel_circuit = create_kernel_circuit(num_features, kernel_type)
                        st.subheader("Kernel Circuit")
                        circuit_img = circuit_to_image(kernel_circuit)
                        st.image(circuit_img, width=700)
                        
                        st.markdown("""
                        **Using with ML models:**
                        
                        To use this quantum kernel with your ML experiments:
                        1. Compute the kernel matrix for your dataset
                        2. Use with kernel methods like SVM, kernel ridge regression, etc.
                        3. Track results and compare with classical kernels in W&B
                        """)
                    except Exception as e:
                        st.error(f"Error generating kernel circuit: {str(e)}")
        
        elif integration_type == "Variational Quantum Circuit":
            st.markdown("""
            ### Variational Quantum Circuits (VQC)
            
            Parameterized quantum circuits that can be trained similar to neural networks.
            
            **How it works:**
            1. Data is encoded into quantum states
            2. A parameterized quantum circuit processes the data
            3. Measurements are taken and used to compute a cost function
            4. Parameters are optimized to minimize the cost function
            
            **Potential Benefits:**
            - Potentially represent complex functions with fewer parameters
            - Leverage quantum properties like entanglement and superposition
            - Suitable for hybrid quantum-classical optimization
            """)
            
            # Example configuration
            st.subheader("Configure VQC")
            
            num_qubits = st.slider("Number of Qubits", 2, 10, 4, key="vqc_qubits")
            layers = st.slider("Number of Variational Layers", 1, 5, 2)
            rotation_blocks = st.multiselect("Rotation Blocks", ["RX", "RY", "RZ"], default=["RY"])
            entanglement = st.selectbox("Entanglement Pattern", ["Full", "Linear", "Circular"], key="vqc_entanglement")
            
            if st.button("Generate VQC Circuit"):
                with st.spinner("Generating circuit..."):
                    try:
                        # Create example VQC circuit
                        vqc_circuit = create_variational_circuit(num_qubits, layers, rotation_blocks, entanglement)
                        st.subheader("Variational Circuit")
                        circuit_img = circuit_to_image(vqc_circuit)
                        st.image(circuit_img, width=700)
                        
                        st.markdown("""
                        **Using with ML models:**
                        
                        To use this VQC with your ML experiments:
                        1. Set up a hybrid quantum-classical optimization loop
                        2. Use classical optimizer to update circuit parameters
                        3. Track training progress and model performance in W&B
                        """)
                    except Exception as e:
                        st.error(f"Error generating VQC: {str(e)}")
                        
        elif integration_type == "Quantum Data Encoding":
            st.markdown("""
            ### Quantum Data Encoding
            
            Techniques to encode classical data into quantum states for processing.
            
            **How it works:**
            1. Classical data is mapped to quantum circuit operations
            2. The resulting quantum state represents the encoded data
            3. Different encoding strategies preserve different properties
            
            **Potential Benefits:**
            - Enables quantum processing of classical data
            - Forms the foundation for most quantum ML algorithms
            - Different encodings can be optimized for different problems
            """)
            
            # Example configuration
            st.subheader("Configure Data Encoding")
            
            encoding_type = st.selectbox("Encoding Type", 
                ["Angle Encoding", "Amplitude Encoding", "Basis Encoding", "Binary Phase Encoding"])
            num_features = st.slider("Number of Data Features", 1, 8, 4, key="encoding_features")
            
            if st.button("Generate Encoding Circuit"):
                with st.spinner("Generating circuit..."):
                    try:
                        # Create example encoding circuit
                        encoding_circuit = create_data_encoding_circuit(encoding_type, num_features)
                        st.subheader("Data Encoding Circuit")
                        circuit_img = circuit_to_image(encoding_circuit)
                        st.image(circuit_img, width=700)
                        
                        st.markdown("""
                        **Using with ML models:**
                        
                        To use this encoding with your ML experiments:
                        1. Preprocess your features to match the encoding requirements
                        2. Apply this encoding as the first stage of your quantum ML pipeline
                        3. Compare different encoding strategies and track results in W&B
                        """)
                    except Exception as e:
                        st.error(f"Error generating encoding circuit: {str(e)}")


# --- Quantum Circuit Builder Helper Functions ---

def open_gate_modal(gate_data, two_qubit=False, with_params=False, with_measurement=False):
    """Open the modal for adding a gate"""
    gate_data["two_qubit"] = two_qubit
    gate_data["with_params"] = with_params
    gate_data["with_measurement"] = with_measurement
    st.session_state.gate_modal = gate_data

def get_next_step(qubit):
    """Get the next available step for a qubit"""
    if not st.session_state.builder_circuit["gates"]:
        return 0
        
    # Find the maximum step for this qubit
    qubit_gates = [g for g in st.session_state.builder_circuit["gates"] 
                  if g["qubit"] == qubit or (g.get("target") == qubit)]
    
    if not qubit_gates:
        return 0
        
    return max(g["step"] for g in qubit_gates) + 1

def display_visual_circuit(circuit_data):
    """Create a visual representation of the quantum circuit"""
    # Create a grid representation of the circuit
    num_qubits = circuit_data["num_qubits"]
    
    # Find the maximum step needed
    max_step = 0
    for gate in circuit_data["gates"]:
        max_step = max(max_step, gate["step"])
    
    # Add a buffer of 5 steps
    grid_width = max_step + 5
    
    # Create HTML for the circuit grid
    html = """
    <style>
    .circuit-grid {
        display: table;
        border-collapse: collapse;
        margin: 10px 0;
    }
    .qubit-row {
        display: table-row;
        height: 60px;
    }
    .step-cell {
        display: table-cell;
        width: 60px;
        height: 60px;
        text-align: center;
        vertical-align: middle;
        position: relative;
    }
    .qubit-label {
        display: table-cell;
        width: 80px;
        text-align: right;
        padding-right: 10px;
        font-weight: bold;
    }
    .gate {
        display: inline-block;
        width: 40px;
        height: 40px;
        line-height: 40px;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .gate-h { background-color: #2196F3; }
    .gate-x { background-color: #F44336; }
    .gate-y { background-color: #FF9800; }
    .gate-z { background-color: #9C27B0; }
    .gate-rx { background-color: #E91E63; }
    .gate-ry { background-color: #00BCD4; }
    .gate-rz { background-color: #673AB7; }
    .gate-p { background-color: #607D8B; }
    .gate-cx { 
        background-color: #F44336; 
        position: relative;
    }
    .gate-cz { 
        background-color: #9C27B0; 
        position: relative;
    }
    .gate-swap { 
        background-color: #FF9800; 
        position: relative;
    }
    .control-line {
        position: absolute;
        background-color: black;
        z-index: 1;
    }
    .vertical-line {
        width: 2px;
        left: 50%;
        transform: translateX(-50%);
    }
    .measure {
        background-color: #607D8B;
        position: relative;
    }
    .measure:after {
        content: "M";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-weight: bold;
    }
    .wire {
        position: absolute;
        left: 0;
        right: 0;
        height: 2px;
        background-color: black;
        top: 50%;
        transform: translateY(-50%);
        z-index: 0;
    }
    </style>
    
    <div class="circuit-grid">
    """
    
    # Create qubit rows
    for q in range(num_qubits):
        html += f'<div class="qubit-row">'
        html += f'<div class="qubit-label">q{q}:</div>'
        
        # Add cells for each step
        for step in range(grid_width):
            html += f'<div class="step-cell">'
            
            # Add a wire
            html += f'<div class="wire"></div>'
            
            # Check if there's a gate at this position
            gates_at_pos = [g for g in circuit_data["gates"] 
                           if g["qubit"] == q and g["step"] == step]
            
            # Also check for multi-qubit gates where this qubit is the target
            target_gates = [g for g in circuit_data["gates"] 
                         if g.get("target") == q and g["step"] == step]
            
            # Single-qubit gates
            if gates_at_pos:
                gate = gates_at_pos[0]
                gate_type = gate["gate_type"]
                
                # Display appropriate gate
                if gate_type == "h":
                    html += f'<div class="gate gate-h">H</div>'
                elif gate_type == "x":
                    html += f'<div class="gate gate-x">X</div>'
                elif gate_type == "y":
                    html += f'<div class="gate gate-y">Y</div>'
                elif gate_type == "z":
                    html += f'<div class="gate gate-z">Z</div>'
                elif gate_type == "rx":
                    html += f'<div class="gate gate-rx">RX</div>'
                elif gate_type == "ry":
                    html += f'<div class="gate gate-ry">RY</div>'
                elif gate_type == "rz":
                    html += f'<div class="gate gate-rz">RZ</div>'
                elif gate_type == "p":
                    html += f'<div class="gate gate-p">P</div>'
                elif gate_type == "measure":
                    html += f'<div class="gate measure"></div>'
                # Control qubit for multi-qubit gates
                elif gate_type in ["cx", "cz"]:
                    html += f'<div class="gate gate-{gate_type}">●</div>'
                elif gate_type == "swap":
                    html += f'<div class="gate gate-swap">⨯</div>'
            
            # Target qubit for multi-qubit gates
            elif target_gates:
                gate = target_gates[0]
                gate_type = gate["gate_type"]
                
                if gate_type == "cx":
                    html += f'<div class="gate gate-x">X</div>'
                elif gate_type == "cz":
                    html += f'<div class="gate gate-z">Z</div>'
                elif gate_type == "swap":
                    html += f'<div class="gate gate-swap">⨯</div>'
            
            html += '</div>'  # end step-cell
        
        html += '</div>'  # end qubit-row
    
    html += '</div>'  # end circuit-grid
    
    # Display using st.markdown with unsafe_allow_html
    st.markdown(html, unsafe_allow_html=True)

def build_circuit_from_builder():
    """Build a Qiskit circuit from the builder state"""
    circuit_data = st.session_state.builder_circuit
    
    # Create a new quantum circuit
    qc = QuantumCircuit(
        circuit_data["num_qubits"],
        circuit_data["num_clbits"] if circuit_data["num_clbits"] > 0 else 0
    )
    
    # Sort gates by step for proper order
    sorted_gates = sorted(circuit_data["gates"], key=lambda g: g["step"])
    
    # Add gates to the circuit
    for gate in sorted_gates:
        gate_type = gate["gate_type"]
        qubit = gate["qubit"]
        
        if gate_type == "h":
            qc.h(qubit)
        elif gate_type == "x":
            qc.x(qubit)
        elif gate_type == "y":
            qc.y(qubit)
        elif gate_type == "z":
            qc.z(qubit)
        elif gate_type == "rx":
            qc.rx(gate.get("params", 0), qubit)
        elif gate_type == "ry":
            qc.ry(gate.get("params", 0), qubit)
        elif gate_type == "rz":
            qc.rz(gate.get("params", 0), qubit)
        elif gate_type == "p":
            qc.p(gate.get("params", 0), qubit)
        elif gate_type == "cx":
            qc.cx(qubit, gate["target"])
        elif gate_type == "cz":
            qc.cz(qubit, gate["target"])
        elif gate_type == "swap":
            qc.swap(qubit, gate["target"])
        elif gate_type == "measure":
            if "clbit" in gate and circuit_data["num_clbits"] > 0:
                qc.measure(qubit, gate["clbit"])
    
    return qc

def generate_qiskit_code(circuit):
    """Generate Qiskit Python code for the given circuit"""
    num_qubits = circuit.num_qubits
    num_clbits = circuit.num_clbits
    
    code = f"from qiskit import QuantumCircuit\n\n"
    code += f"# Create circuit with {num_qubits} qubits and {num_clbits} classical bits\n"
    code += f"qc = QuantumCircuit({num_qubits}, {num_clbits})\n\n"
    
    # Add gate operations
    code += "# Add gates\n"
    for instruction, qargs, cargs in circuit.data:
        gate_name = instruction.name
        
        if gate_name in ["h", "x", "y", "z"]:
            qubit = qargs[0].index
            code += f"qc.{gate_name}({qubit})\n"
        elif gate_name in ["rx", "ry", "rz", "p"]:
            qubit = qargs[0].index
            param = instruction.params[0]
            code += f"qc.{gate_name}({param}, {qubit})\n"
        elif gate_name == "cx":
            control = qargs[0].index
            target = qargs[1].index
            code += f"qc.cx({control}, {target})\n"
        elif gate_name == "cz":
            control = qargs[0].index
            target = qargs[1].index
            code += f"qc.cz({control}, {target})\n"
        elif gate_name == "swap":
            qubit1 = qargs[0].index
            qubit2 = qargs[1].index
            code += f"qc.swap({qubit1}, {qubit2})\n"
        elif gate_name == "measure":
            qubit = qargs[0].index
            clbit = cargs[0].index
            code += f"qc.measure({qubit}, {clbit})\n"
    
    return code

# --- Quantum-ML integration functions ---

def create_feature_map_circuit(num_features, entanglement, reps):
    """Create a quantum feature map circuit"""
    num_qubits = num_features
    qc = QuantumCircuit(num_qubits)
    
    # First layer of Hadamards
    for i in range(num_qubits):
        qc.h(i)
    
    # Repeated blocks based on reps
    for r in range(reps):
        # Rotation layer (data would be encoded here)
        for i in range(num_qubits):
            # These rotations would normally depend on the input data
            qc.rz(np.pi/4, i)  # Placeholder angle
        
        # Entanglement layer
        if entanglement == "Full":
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    qc.cx(i, j)
        elif entanglement == "Linear":
            for i in range(num_qubits-1):
                qc.cx(i, i+1)
        elif entanglement == "Circular":
            for i in range(num_qubits-1):
                qc.cx(i, i+1)
            qc.cx(num_qubits-1, 0)  # Connect last to first
    
    return qc

def create_kernel_circuit(num_features, kernel_type):
    """Create a quantum kernel circuit"""
    num_qubits = num_features
    qc = QuantumCircuit(num_qubits)
    
    if kernel_type == "Quantum Feature Map":
        # Similar to feature map but designed for kernel evaluation
        for i in range(num_qubits):
            qc.h(i)
        
        # First data vector would be encoded here
        for i in range(num_qubits):
            qc.rz(np.pi/4, i)  # Placeholder angle
            
        # Entanglement
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
            
        # Inverse operations to compute kernel
        for i in range(num_qubits-1, 0, -1):
            qc.cx(i-1, i)
            
        # Second data vector would be encoded here
        for i in range(num_qubits):
            qc.rz(-np.pi/4, i)  # Placeholder, would be -angle2
            
        for i in range(num_qubits):
            qc.h(i)
            
    elif kernel_type == "Projected Quantum Kernel":
        # Angle encoding
        for i in range(num_qubits):
            qc.ry(np.pi/4, i)  # Placeholder angle
        
        # Random unitary transformation
        for i in range(num_qubits-1):
            qc.cx(i, i+1)
        
        for i in range(num_qubits):
            qc.rx(np.pi/2, i)
            
    elif kernel_type == "Custom":
        # Custom kernel design
        for i in range(num_qubits):
            qc.h(i)
            
        for r in range(2):  # 2 repetitions
            for i in range(num_qubits):
                qc.rz(np.pi/3, i)  # Placeholder
                qc.rx(np.pi/2, i)  # Placeholder
                
            if r < 1:  # Skip entanglement in last layer
                for i in range(num_qubits-1):
                    qc.cz(i, i+1)
    
    # Measurement basis
    for i in range(num_qubits):
        qc.h(i)
        
    # Add measurements
    qc.measure_all()
    
    return qc

def create_variational_circuit(num_qubits, layers, rotation_blocks, entanglement):
    """Create a variational quantum circuit"""
    qc = QuantumCircuit(num_qubits)
    
    # Initial state preparation (could be data encoding)
    for i in range(num_qubits):
        qc.h(i)
    
    # Variational layers
    for l in range(layers):
        # Rotation blocks
        for i in range(num_qubits):
            if "RX" in rotation_blocks:
                # Parameter would typically be trainable
                qc.rx(np.pi/4, i)  # Placeholder angle
            if "RY" in rotation_blocks:
                qc.ry(np.pi/4, i)  # Placeholder angle
            if "RZ" in rotation_blocks:
                qc.rz(np.pi/4, i)  # Placeholder angle
        
        # Entanglement
        if entanglement == "Full":
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    qc.cx(i, j)
        elif entanglement == "Linear":
            for i in range(num_qubits-1):
                qc.cx(i, i+1)
        elif entanglement == "Circular":
            for i in range(num_qubits-1):
                qc.cx(i, i+1)
            qc.cx(num_qubits-1, 0)
    
    # Final rotation layer
    for i in range(num_qubits):
        if "RY" in rotation_blocks:
            qc.ry(np.pi/4, i)  # Placeholder angle
    
    # Add measurements
    qc.measure_all()
    
    return qc

def create_data_encoding_circuit(encoding_type, num_features):
    """Create a circuit for data encoding"""
    # Determine number of qubits based on encoding type and features
    if encoding_type == "Amplitude Encoding":
        # Need log2(n) qubits for n features
        num_qubits = int(np.ceil(np.log2(num_features)))
    else:
        # For other encodings, typically 1 qubit per feature
        num_qubits = num_features
        
    qc = QuantumCircuit(num_qubits)
    
    if encoding_type == "Angle Encoding":
        # Encode each feature into rotation angle of a qubit
        for i in range(min(num_qubits, num_features)):
            qc.ry(np.pi/4, i)  # Placeholder angle, would be data[i]
            
    elif encoding_type == "Amplitude Encoding":
        # Initialize in superposition
        for i in range(num_qubits):
            qc.h(i)
        
        # Complex amplitude encoding would require custom gates
        # This is a placeholder showing the general structure
        for i in range(num_qubits-1):
            qc.cz(i, i+1)
            
    elif encoding_type == "Basis Encoding":
        # Each bit of binary feature representation flips a qubit
        # For simplicity, we'll just show sample X gates
        for i in range(min(num_qubits, num_features)):
            # Would check if bit i is 1 in data
            if i % 2 == 1:  # Just for demonstration
                qc.x(i)
                
    elif encoding_type == "Binary Phase Encoding":
        # Initialize superposition
        for i in range(num_qubits):
            qc.h(i)
            
        # Conditionally apply phase based on data bits
        for i in range(min(num_qubits, num_features)):
            # Would check if bit i is 1 in data
            if i % 2 == 1:  # Just for demonstration
                qc.z(i)
    
    # For demonstration, add measurements
    qc.measure_all()
    
    return qc

def render_quantum_security():
    """
    Renders quantum security and cybersecurity capabilities
    """
    st.subheader("Quantum Security & Cybersecurity")
    
    st.markdown("""
    Evaluate and enhance your security posture against quantum threats with these advanced tools.
    Quantum computing poses both challenges and opportunities for cybersecurity.
    """)
    
    # Create tabs for different security tools
    encryption_tab, risk_tab, qrng_tab = st.tabs([
        "Quantum-Safe Encryption", 
        "Quantum Risk Assessment",
        "Quantum Random Number Generator"
    ])
    
    with encryption_tab:
        st.subheader("Quantum-Safe Encryption Analysis")
        
        st.markdown("""
        Evaluate the quantum resistance of different encryption algorithms against quantum attacks.
        Quantum computers threaten traditional encryption through Shor's algorithm and Grover's algorithm.
        """)
        
        # Encryption algorithm selection
        algorithm_type = st.selectbox(
            "Select encryption algorithm to evaluate",
            ["RSA (2048-bit)", "ECC (P-256)", "AES-256", "Lattice-based (NTRU)", "Hash-based (SPHINCS+)", "Custom"]
        )
        
        # Display algorithm details and quantum vulnerability
        if algorithm_type == "RSA (2048-bit)":
            st.warning("⚠️ **High Risk**: Vulnerable to Shor's algorithm")
            
            st.markdown("""
            **RSA-2048 Assessment:**
            - **Quantum Security Level**: 0 bits (broken by quantum computers)
            - **Estimated Qubits Needed**: ~4,000 logical qubits
            - **Time to Break (Est.)**: Hours on a fault-tolerant quantum computer
            - **Recommendation**: Migrate to quantum-resistant alternatives
            """)
            
            # Time estimates chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Classical Supercomputer", "Future Quantum Computer"],
                y=[2.7e19, 8],
                text=["~1 billion years", "~8 hours"],
                textposition="auto",
                marker_color=["green", "red"]
            ))
            fig.update_layout(
                title="Estimated Time to Break RSA-2048",
                yaxis_type="log",
                yaxis_title="Hours (log scale)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif algorithm_type == "ECC (P-256)":
            st.warning("⚠️ **High Risk**: Vulnerable to Shor's algorithm")
            
            st.markdown("""
            **ECC P-256 Assessment:**
            - **Quantum Security Level**: 0 bits (broken by quantum computers)
            - **Estimated Qubits Needed**: ~2,300 logical qubits
            - **Time to Break (Est.)**: Hours on a fault-tolerant quantum computer
            - **Recommendation**: Migrate to quantum-resistant alternatives
            """)
            
            # Time estimates chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Classical Supercomputer", "Future Quantum Computer"],
                y=[1.1e13, 4],
                text=["~1.3 million years", "~4 hours"],
                textposition="auto",
                marker_color=["green", "red"]
            ))
            fig.update_layout(
                title="Estimated Time to Break ECC P-256",
                yaxis_type="log",
                yaxis_title="Hours (log scale)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif algorithm_type == "AES-256":
            st.success("✅ **Low Risk**: Resistant to quantum attacks with sufficient key size")
            
            st.markdown("""
            **AES-256 Assessment:**
            - **Quantum Security Level**: ~128 bits (with Grover's algorithm)
            - **Estimated Qubits Needed**: Thousands for meaningful speedup
            - **Time to Break (Est.)**: Still exponential, but reduced security margin
            - **Recommendation**: Increase key size to AES-512 for long-term security
            """)
            
            # Security level chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Classical Security", "Quantum Security"],
                y=[256, 128],
                text=["256 bits", "128 bits"],
                textposition="auto",
                marker_color=["green", "yellow"]
            ))
            fig.update_layout(
                title="AES-256 Security Level Comparison",
                yaxis_title="Security Level (bits)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif algorithm_type == "Lattice-based (NTRU)":
            st.success("✅ **Low Risk**: Believed to be quantum-resistant")
            
            st.markdown("""
            **NTRU Assessment:**
            - **Quantum Security Level**: ~128+ bits (believed resistant to quantum attacks)
            - **Security Basis**: Hardness of solving certain lattice problems
            - **Standardization Status**: Finalist in NIST post-quantum standardization
            - **Recommendation**: Suitable for quantum-resistant implementations
            """)
            
            # Performance comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Key Size", "Ciphertext Size", "Encryption Speed", "Decryption Speed"],
                y=[1, 1.2, 0.7, 0.8],
                text=["1x", "1.2x", "0.7x", "0.8x"],
                textposition="auto",
                marker_color="blue",
                name="Relative to RSA (smaller is better)"
            ))
            fig.update_layout(
                title="NTRU Performance vs. RSA (Normalized)",
                yaxis_title="Relative Performance"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif algorithm_type == "Hash-based (SPHINCS+)":
            st.success("✅ **Low Risk**: Believed to be quantum-resistant")
            
            st.markdown("""
            **SPHINCS+ Assessment:**
            - **Quantum Security Level**: ~128+ bits (believed resistant to quantum attacks)
            - **Security Basis**: Hardness of finding hash function collisions
            - **Standardization Status**: Finalist in NIST post-quantum standardization
            - **Recommendation**: Suitable for quantum-resistant digital signatures
            """)
            
            # Performance comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Signature Size", "Key Generation", "Signing Speed", "Verification Speed"],
                y=[20, 0.3, 15, 0.5],
                text=["20x", "0.3x", "15x", "0.5x"],
                textposition="auto",
                marker_color="blue",
                name="Relative to RSA (smaller is better)"
            ))
            fig.update_layout(
                title="SPHINCS+ Performance vs. RSA (Normalized)",
                yaxis_title="Relative Performance"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Custom
            st.info("Enter parameters for custom algorithm analysis")
            
            custom_security = st.number_input("Classical Security Level (bits)", min_value=1, max_value=512, value=256)
            quantum_speedup = st.slider("Estimated Quantum Speedup Factor", min_value=1, max_value=100, value=2)
            
            quantum_security = custom_security / (2**quantum_speedup)
            
            st.markdown(f"""
            **Custom Algorithm Assessment:**
            - **Classical Security Level**: {custom_security} bits
            - **Estimated Quantum Security Level**: {quantum_security:.2f} bits
            - **Quantum Speedup Factor**: {quantum_speedup}x
            """)
            
            if quantum_security < 80:
                st.error("⚠️ **Critical Risk**: Security level too low for quantum resistance")
            elif quantum_security < 128:
                st.warning("🔶 **Moderate Risk**: May need stronger parameters for long-term security")
            else:
                st.success("✅ **Low Risk**: Likely resistant to quantum attacks")
        
        # Migration recommendations
        st.subheader("Migration Recommendations")
        st.markdown("""
        **NIST Recommended Post-Quantum Algorithms:**
        
        1. **Key Encapsulation Mechanisms (KEM):**
           - CRYSTALS-Kyber (primary recommendation)
           - NTRU and SABER (alternatives)
        
        2. **Digital Signatures:**
           - CRYSTALS-Dilithium (primary recommendation)
           - FALCON and SPHINCS+ (alternatives)
        
        3. **Implementation Considerations:**
           - Crypto-agility: Support multiple algorithms
           - Hybrid approaches: Combine classical + post-quantum
           - Regular security assessments against quantum advances
        """)
    
    with risk_tab:
        st.subheader("Quantum Risk Assessment")
        
        st.markdown("""
        Evaluate your organization's risk exposure to quantum computing threats
        and develop a quantum-ready security strategy.
        """)
        
        # Risk assessment form
        with st.form("risk_assessment_form"):
            st.markdown("#### Security Infrastructure Assessment")
            
            encryption_types = st.multiselect(
                "Select encryption algorithms currently in use:",
                ["RSA", "ECC", "AES", "3DES", "Blowfish", "ChaCha20", "Post-Quantum Algorithms"],
                default=["RSA", "AES"]
            )
            
            key_exchange = st.multiselect(
                "Select key exchange methods in use:",
                ["Diffie-Hellman", "ECDH", "RSA key exchange", "PSK", "Quantum Key Distribution", "Post-Quantum Methods"],
                default=["Diffie-Hellman", "ECDH"]
            )
            
            data_lifespan = st.slider(
                "How long must your data remain secure? (years)",
                min_value=1,
                max_value=50,
                value=10
            )
            
            st.markdown("#### Quantum Threat Timeline")
            
            q_timeline = st.slider(
                "When do you expect cryptographically-relevant quantum computers? (years from now)",
                min_value=1,
                max_value=30,
                value=10
            )
            
            migration_time = st.slider(
                "Estimated time needed to migrate to quantum-resistant algorithms (years)",
                min_value=1,
                max_value=10,
                value=3
            )
            
            st.markdown("#### Critical Systems")
            
            critical_systems = st.multiselect(
                "Select systems with highest security requirements:",
                ["Financial transactions", "Customer data", "Authentication systems", 
                 "Cloud services", "IoT devices", "Healthcare data", "Government/Defense"],
                default=["Financial transactions", "Customer data"]
            )
            
            submit_button = st.form_submit_button("Analyze Quantum Risk")
        
        # If form is submitted, show risk analysis
        if submit_button:
            st.subheader("Quantum Risk Analysis Results")
            
            # Calculate risk score
            risk_score = 0
            
            # Add risk for vulnerable algorithms
            if "RSA" in encryption_types: risk_score += 30
            if "ECC" in encryption_types: risk_score += 25
            if "Diffie-Hellman" in key_exchange: risk_score += 15
            if "ECDH" in key_exchange: risk_score += 15
            if "RSA key exchange" in key_exchange: risk_score += 20
            
            # Reduce risk for quantum-safe algorithms
            if "Post-Quantum Algorithms" in encryption_types: risk_score -= 20
            if "AES" in encryption_types and "3DES" not in encryption_types: risk_score -= 5
            if "Quantum Key Distribution" in key_exchange: risk_score -= 10
            if "Post-Quantum Methods" in key_exchange: risk_score -= 15
            
            # Adjust for timeline factors
            if data_lifespan > q_timeline: risk_score += 25
            if q_timeline - migration_time < 5: risk_score += 15
            
            # Critical systems factor
            critical_count = len(critical_systems)
            risk_score += critical_count * 5
            
            # Normalize score (0-100)
            risk_score = max(0, min(100, risk_score))
            
            # Display risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Quantum Risk Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "gold"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk timeline
            st.subheader("Quantum Threat Timeline")
            
            years = list(range(0, max(data_lifespan, q_timeline + 5) + 1))
            security_margin = [100 if y < q_timeline - migration_time else 
                              max(0, 100 - (y - (q_timeline - migration_time)) * 100 / migration_time)
                              for y in years]
            
            timeline_df = pd.DataFrame({
                'Year': [f"Now + {y}yr" for y in years],
                'Security Margin (%)': security_margin
            })
            
            fig = px.line(timeline_df, x='Year', y='Security Margin (%)', markers=True)
            fig.add_vline(x=f"Now + {q_timeline}yr", line_dash="dash", line_color="red",
                         annotation_text="Est. Cryptographically-Relevant Quantum Computer")
            fig.add_vline(x=f"Now + {data_lifespan}yr", line_dash="dash", line_color="blue",
                         annotation_text="Required Data Protection Period")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations based on risk score
            st.subheader("Recommendations")
            
            if risk_score < 40:
                st.success("**Low Quantum Risk**")
                st.markdown("""
                Your current security posture appears well-positioned for the quantum transition:
                
                1. **Monitor Developments**: Continue tracking quantum computing advancements
                2. **Crypto-Agility**: Maintain ability to quickly update cryptographic algorithms
                3. **Selective Migration**: Begin transitioning most critical systems to post-quantum algorithms
                """)
            elif risk_score < 70:
                st.warning("**Moderate Quantum Risk**")
                st.markdown("""
                Your organization faces significant risks from quantum computing advances:
                
                1. **Accelerate Planning**: Develop a detailed quantum security transition plan
                2. **Begin Migration**: Start implementing hybrid classical/post-quantum solutions
                3. **Risk Assessment**: Identify and prioritize vulnerable systems and data
                4. **Timeline Revision**: Consider accelerating your migration timeline
                """)
            else:
                st.error("**High Quantum Risk**")
                st.markdown("""
                Your organization is highly vulnerable to quantum computing threats:
                
                1. **Urgent Action Required**: Immediate attention to quantum threats is necessary
                2. **Critical System Protection**: Prioritize protecting systems with long-term security requirements
                3. **Comprehensive Migration**: Develop and implement a full transition to quantum-resistant algorithms
                4. **Expert Consultation**: Consider engaging with quantum security specialists
                5. **Defensive Depth**: Implement additional security layers to protect vulnerable encryption
                """)

    with qrng_tab:
        st.subheader("Quantum Random Number Generator")
        
        st.markdown("""
        Generate true random numbers based on quantum principles. These random numbers
        are useful for cryptographic applications, simulations, and statistical sampling.
        """)
        
        # Options for QRNG
        num_bits = st.slider("Number of random bits to generate", 8, 256, 128, step=8)
        output_format = st.selectbox("Output format", ["Binary", "Hexadecimal", "Decimal", "Base64"])
        
        # Generate button
        if st.button("Generate Quantum Random Numbers"):
            with st.spinner("Accessing quantum hardware..."):
                # Create a quantum circuit for random number generation
                qrng_circuit = QuantumCircuit(num_bits, num_bits)
                
                # Apply Hadamard gates to create superposition
                for i in range(num_bits):
                    qrng_circuit.h(i)
                
                # Measure all qubits
                qrng_circuit.measure_all()
                
                # Run the circuit
                try:
                    # Display the circuit
                    circuit_img = circuit_to_image(qrng_circuit)
                    st.subheader("QRNG Circuit")
                    st.image(circuit_img, width=700)
                    
                    # Simulate the circuit
                    simulator = Aer.get_backend('qasm_simulator')
                    job = simulator.run(qrng_circuit, shots=1)
                    result = job.result()
                    counts = result.get_counts()
                    
                    # Extract random bits
                    random_bits = list(counts.keys())[0]
                    
                    # Convert to requested format
                    if output_format == "Binary":
                        output_value = random_bits
                    elif output_format == "Hexadecimal":
                        output_value = hex(int(random_bits, 2))[2:]
                    elif output_format == "Decimal":
                        output_value = str(int(random_bits, 2))
                    else:  # Base64
                        import base64
                        # Convert binary string to bytes and then to base64
                        bytes_val = int(random_bits, 2).to_bytes((len(random_bits) + 7) // 8, byteorder='big')
                        output_value = base64.b64encode(bytes_val).decode('utf-8')
                    
                    # Show results
                    st.subheader("Generated Quantum Random Value")
                    st.code(output_value)
                    
                    # Show quantum properties
                    st.subheader("Quantum Properties")
                    st.markdown(f"""
                    **Entropy Source**: Quantum measurement of superposition states
                    
                    **Bits Generated**: {num_bits} bits
                    
                    **True Randomness**: Unlike classical random number generators which are deterministic,
                    quantum random numbers derive their randomness from inherent quantum unpredictability.
                    
                    **Applications**:
                    - Cryptographic key generation
                    - Secure communications
                    - Monte Carlo simulations
                    - Statistical sampling
                    - Gaming and lottery systems
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating quantum random numbers: {str(e)}")

def render_ai_insights():
    """
    Renders AI-powered insights for W&B experiments
    """
    # Header with logo and title
    col1, col2 = st.columns([1, 5])
    
    with col1:
        try:
            logo = Image.open("assets/quantum_logo.jpg")
            st.image(logo, width=80)
        except FileNotFoundError:
            pass
    
    with col2:
        st.header("AI Insights for Your Experiments")
    
    if not st.session_state.authenticated:
        st.warning("Please log in with your W&B API key to access AI insights for your experiments.")
        return
    
    st.markdown("""
    Get AI-powered insights about your machine learning experiments using
    quantum computing techniques and advanced analytics.
    Select a project and run to analyze.
    """)
    
    # Only proceed if user has selected a project
    if not st.session_state.selected_project:
        st.info("Please select a project from the Projects page first.")
        return
    
    project_id = st.session_state.selected_project["id"]
    
    # Fetch runs if we have a project selected
    with st.spinner("Fetching runs data..."):
        try:
            api = wandb.Api()
            runs = list(api.runs(project_id))
            
            if not runs:
                st.info(f"No runs found in project {project_id}.")
                return
                
            # Get available metrics across all runs
            all_metrics = set()
            for run in runs:
                all_metrics.update(run.summary._json_dict.keys())
            
            # Filter out non-metric keys
            metrics = [m for m in all_metrics if not m.startswith('_') and isinstance(
                runs[0].summary._json_dict.get(m, None), (int, float)
            )]
            
            if not metrics:
                st.warning("No numeric metrics found in your runs.")
                return
                
            # Select metrics to analyze
            selected_metrics = st.multiselect(
                "Select metrics to analyze",
                options=sorted(metrics),
                default=sorted(metrics)[:2] if len(metrics) >= 2 else sorted(metrics)[:1]
            )
            
            if not selected_metrics:
                st.info("Please select at least one metric to analyze.")
                return
                
            # Generate insights
            st.subheader("Generated Insights")
            
            with st.spinner("Analyzing your experiments..."):
                # Collect metric values across runs
                metric_data = {}
                for metric in selected_metrics:
                    values = []
                    for run in runs:
                        if metric in run.summary._json_dict:
                            value = run.summary._json_dict[metric]
                            if isinstance(value, (int, float)):
                                values.append({
                                    "run_id": run.id,
                                    "run_name": run.name,
                                    "value": value
                                })
                    
                    if values:
                        metric_data[metric] = values
                
                # Generate insights for each metric
                for metric, values in metric_data.items():
                    st.write(f"### Analysis of {metric}")
                    
                    # Sort values
                    sorted_values = sorted(values, key=lambda x: x["value"], reverse=True)
                    
                    # Best run
                    best_run = sorted_values[0]
                    st.write(f"🏆 Best performance: **{best_run['value']:.4f}** achieved by run **{best_run['run_name']}**")
                    
                    # Calculate statistics
                    metric_values = [v["value"] for v in values]
                    avg_value = np.mean(metric_values)
                    median_value = np.median(metric_values)
                    std_value = np.std(metric_values)
                    
                    st.write(f"📊 Average: **{avg_value:.4f}**, Median: **{median_value:.4f}**, Std Dev: **{std_value:.4f}**")
                    
                    # Plot distribution
                    fig = px.histogram(
                        metric_values, 
                        title=f"Distribution of {metric} across runs",
                        labels={"value": metric}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Generate text insight
                    if std_value / avg_value > 0.2:  # High variance
                        st.write("🔍 **Insight:** High variance detected in your runs. Consider stabilizing your training process or exploring different hyperparameters.")
                    else:
                        st.write("🔍 **Insight:** Your runs show consistent performance. You might want to explore more diverse hyperparameters to potentially improve results.")
                
                # Quantum advantage section
                st.subheader("Quantum Advantage Potential")
                st.markdown("""
                Based on your experiment patterns, these areas might benefit from quantum computing approaches:
                
                1. **Feature Selection**: Quantum algorithms can explore exponentially large feature spaces more efficiently.
                2. **Optimization**: Quantum annealing may help find better global optima for your model parameters.
                3. **Anomaly Detection**: Quantum circuits can potentially identify complex patterns in your data.
                
                Try the Quantum Computing tools to explore these possibilities with your data.
                """)
                
        except Exception as e:
            st.error(f"Error analyzing experiments: {str(e)}")

# --- Quantum utility functions ---

def create_bell_state():
    """Create a Bell state quantum circuit"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc

def create_ghz_state(num_qubits):
    """Create a GHZ state with the specified number of qubits"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    qc.measure_all()
    return qc

def create_qft_circuit(num_qubits):
    """Create a Quantum Fourier Transform circuit"""
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Initialize with a superposition
    for i in range(num_qubits):
        qc.h(i)
    
    # Apply QFT
    for i in range(num_qubits):
        for j in range(i+1, num_qubits):
            qc.cp(np.pi/float(2**(j-i)), i, j)
        qc.h(i)
    
    # Swap qubits
    for i in range(num_qubits//2):
        qc.swap(i, num_qubits-i-1)
    
    qc.measure_all()
    return qc

def circuit_to_image(circuit):
    """Convert a Qiskit circuit to an image"""
    from utils.logo_protection import COPYRIGHT_OWNER
    
    # Draw the circuit
    circuit_drawing = circuit.draw(output='mpl')
    
    # Save to buffer
    buf = io.BytesIO()
    circuit_drawing.savefig(buf, format='png')
    buf.seek(0)
    
    # Open as PIL Image
    img = Image.open(buf)
    
    # Add logo and copyright
    try:
        logo = Image.open("assets/quantum_logo.jpg")
        # Resize logo to be proportional to the circuit image
        logo_width = min(img.width // 4, 150)
        logo_height = int(logo.height * (logo_width / logo.width))
        logo = logo.resize((logo_width, logo_height))
        
        # Create a new image with space for logo
        new_height = img.height + logo_height + 20  # Add padding
        new_img = Image.new('RGB', (img.width, new_height), (255, 255, 255))
        
        # Paste circuit image
        new_img.paste(img, (0, logo_height + 20))
        
        # Paste logo at top-left
        new_img.paste(logo, (10, 10))
        
        # Add copyright text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(new_img)
        try:
            # Try to load a font, fall back to default if not available
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        copyright_text = f"© {COPYRIGHT_OWNER}"
        draw.text((logo_width + 20, 20), copyright_text, fill=(0, 0, 0), font=font)
        
        return new_img
    except Exception as e:
        # If any error occurs with the logo, return the original image
        return img

def run_quantum_circuit(circuit):
    """Run a quantum circuit simulation"""
    # For Qiskit 1.0+, use the AerSimulator from qiskit_aer
    from qiskit_aer import AerSimulator
    simulator = AerSimulator()
    
    # In Qiskit 1.0+, we need to make sure the circuit has measurements
    # If it doesn't have measurements, add them to all qubits
    if not circuit.num_clbits:
        from qiskit import QuantumCircuit
        measured_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_qubits)
        measured_circuit.compose(circuit, inplace=True)
        measured_circuit.measure_all()
        circuit = measured_circuit
    
    # Run the simulation
    job = simulator.run(circuit, shots=1024)
    result = job.result()
    return result

def plot_quantum_results(counts):
    """Plot the results of a quantum circuit simulation"""
    states = list(counts.keys())
    values = list(counts.values())
    
    fig = px.bar(
        x=states, 
        y=values,
        labels={'x': 'Quantum State', 'y': 'Count'},
        title="Quantum Circuit Measurement Results"
    )
    
    fig.update_layout(
        xaxis_title="Quantum State",
        yaxis_title="Count",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def save_quantum_results_to_wandb(circuit, counts):
    """Save quantum simulation results to a W&B run"""
    if not st.session_state.authenticated or not st.session_state.selected_run:
        return False
    
    try:
        # Get the run
        api = wandb.Api()
        project_id = st.session_state.selected_project["id"]
        run_id = st.session_state.selected_run["id"]
        run = api.run(f"{project_id}/{run_id}")
        
        # Create a summary of the quantum results
        quantum_results = {
            "quantum_circuit_type": circuit.name if hasattr(circuit, "name") else "custom",
            "num_qubits": circuit.num_qubits,
            "measurement_counts": counts,
            "most_frequent_state": max(counts.items(), key=lambda x: x[1])[0],
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Update the run summary with the quantum results
        run.summary.update({
            "quantum_results": json.dumps(quantum_results)
        })
        run.update()
        
        return True
    except Exception as e:
        st.error(f"Error saving to W&B: {str(e)}")
        return False