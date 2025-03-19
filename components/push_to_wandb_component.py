import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wandb
from datetime import datetime
import json
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import pennylane as qml

def push_to_wandb_component():
    """Push quantum circuits and experiments to W&B"""
    st.subheader("Push to W&B")
    st.markdown("""
    Create and push new quantum experiments directly to your W&B account.
    Track circuits, metrics, and results automatically.
    """)
    
    # Circuit type selection
    circuit_type = st.selectbox(
        "Select quantum circuit type to create:",
        ["Bell State", "GHZ State", "Quantum Fourier Transform", "Random Circuit", "Custom Circuit"]
    )
    
    # Configuration parameters
    with st.expander("Circuit Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            n_qubits = st.number_input("Number of qubits:", min_value=1, max_value=10, value=2)
        with col2:
            shots = st.number_input("Number of shots:", min_value=1, max_value=10000, value=1024)
            
        if circuit_type == "Random Circuit":
            circuit_depth = st.slider("Circuit depth:", min_value=1, max_value=10, value=3)
        
        # Experiment naming
        experiment_name = st.text_input(
            "Experiment name:", 
            value=f"{circuit_type.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        # Tags
        tags = st.text_input("Tags (comma separated):", value="quantum")
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # Project selection
    st.subheader("Select W&B Project")
    try:
        api = wandb.Api()
        
        # Option to use existing project or create new
        use_existing = st.radio(
            "Project options:",
            ["Use existing project", "Create new project"]
        )
        
        if use_existing == "Use existing project":
            # Load projects
            with st.spinner("Loading projects..."):
                projects = []
                for project in api.projects():
                    projects.append({
                        "name": project.name,
                        "entity": project.entity,
                        "id": f"{project.entity}/{project.name}",
                    })
                
                if projects:
                    project_options = [f"{p['entity']}/{p['name']}" for p in projects]
                    selected_project = st.selectbox("Select project:", project_options)
                    
                    if selected_project:
                        entity, project_name = selected_project.split("/")
                else:
                    st.warning("No projects found. Please create a new project.")
                    use_existing = "Create new project"
        
        if use_existing == "Create new project":
            # Create new project form
            with st.form("new_project_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    entity = st.text_input("Entity name:", value=api.viewer()['entity'])
                
                with col2:
                    project_name = st.text_input("Project name:", value="quantum-assistant-project")
                
                project_desc = st.text_area("Project description:", value="Quantum computing experiments from Quantum AI Assistant")
                
                submit_new_project = st.form_submit_button("Create Project")
                
                if submit_new_project:
                    try:
                        # Create project via API
                        new_project = api.create_project(name=project_name, entity=entity, description=project_desc)
                        st.success(f"Project {entity}/{project_name} created successfully!")
                    except Exception as e:
                        st.error(f"Error creating project: {str(e)}")
                        return
    except Exception as e:
        st.error(f"Error connecting to W&B API: {str(e)}")
        st.info("Please make sure you are properly authenticated with W&B.")
        return
    
    # Run experiment button
    if st.button("Create and Push Experiment to W&B"):
        with st.spinner("Creating and pushing experiment..."):
            try:
                # Create circuit based on type
                if circuit_type == "Bell State":
                    circuit = create_bell_state(n_qubits)
                elif circuit_type == "GHZ State":
                    circuit = create_ghz_state(n_qubits)
                elif circuit_type == "Quantum Fourier Transform":
                    circuit = create_qft_circuit(n_qubits)
                elif circuit_type == "Random Circuit":
                    circuit = create_random_circuit(n_qubits, circuit_depth)
                else:  # Custom Circuit
                    circuit = QuantumCircuit(n_qubits)
                    circuit.h(range(n_qubits))
                    for i in range(n_qubits-1):
                        circuit.cx(i, i+1)
                    circuit.measure_all()
                
                # Run the circuit
                from push_to_wandb import run_quantum_circuit, circuit_to_image, plot_quantum_results
                counts = run_quantum_circuit(circuit, shots=shots)
                
                # Generate images
                circuit_img = circuit_to_image(circuit)
                results_img = plot_quantum_results(counts)
                
                # Initialize wandb run
                config = {
                    "circuit_type": circuit_type,
                    "n_qubits": n_qubits,
                    "shots": shots,
                    "circuit_depth": circuit_depth if circuit_type == "Random Circuit" else circuit.depth(),
                    "backend": "qasm_simulator"
                }
                
                run = wandb.init(
                    project=project_name,
                    entity=entity,
                    name=experiment_name,
                    config=config,
                    tags=tag_list,
                    reinit=True
                )
                
                # Log to wandb
                wandb.log({
                    "circuit_diagram": wandb.Image(circuit_img),
                    "measurement_results": wandb.Image(results_img),
                    "circuit_depth": circuit.depth(),
                    "circuit_width": circuit.width(),
                    "circuit_size": circuit.size(),
                    "gate_counts": str(circuit.count_ops()),
                })
                
                # Convert counts to probabilities
                probs = {}
                total = sum(counts.values())
                for state, count in counts.items():
                    probs[state] = count / total
                
                # Log probabilities as a table
                prob_data = [[state, prob] for state, prob in probs.items()]
                prob_table = wandb.Table(columns=["State", "Probability"], data=prob_data)
                wandb.log({"probability_table": prob_table})
                
                # Finish the run
                wandb.finish()
                
                # Show success message
                st.success(f"Experiment successfully pushed to W&B project {entity}/{project_name}!")
                
                # Show results
                st.subheader("Experiment Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(circuit_img, caption="Circuit Diagram")
                with col2:
                    st.image(results_img, caption="Measurement Results")
                
                # Add link to the experiment
                st.markdown(f"View experiment details: [W&B Dashboard](https://wandb.ai/{entity}/{project_name}/runs/{run.id})")
            
            except Exception as e:
                st.error(f"Error running experiment: {str(e)}")

def create_bell_state(n_qubits=2):
    """Create a Bell state quantum circuit"""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for i in range(1, n_qubits):
        circuit.cx(0, i)
    circuit.measure_all()
    return circuit

def create_ghz_state(n_qubits):
    """Create a GHZ state with the specified number of qubits"""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for i in range(n_qubits-1):
        circuit.cx(i, i+1)
    circuit.measure_all()
    return circuit

def create_qft_circuit(n_qubits):
    """Create a Quantum Fourier Transform circuit"""
    from qiskit.circuit.library import QFT
    circuit = QuantumCircuit(n_qubits)
    # Apply Hadamard to all qubits to create superposition
    circuit.h(range(n_qubits))
    # Apply QFT
    circuit.append(QFT(n_qubits), range(n_qubits))
    circuit.measure_all()
    return circuit

def create_random_circuit(n_qubits, depth):
    """Create a random quantum circuit"""
    circuit = QuantumCircuit(n_qubits)
    
    # Add random gates
    from numpy.random import randint
    
    for _ in range(depth):
        # Apply random single-qubit gates
        for i in range(n_qubits):
            gate = randint(0, 3)
            if gate == 0:
                circuit.h(i)
            elif gate == 1:
                circuit.x(i)
            elif gate == 2:
                circuit.z(i)
            else:
                circuit.t(i)
        
        # Apply random CNOT gates
        for _ in range(max(1, n_qubits//2)):
            ctrl = randint(0, n_qubits)
            targ = randint(0, n_qubits)
            if ctrl != targ and ctrl < n_qubits and targ < n_qubits:
                circuit.cx(ctrl, targ)
    
    circuit.measure_all()
    return circuit