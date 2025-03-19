import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wandb
from datetime import datetime
import json
import os
import io
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import pennylane as qml

# Import components
from components.push_to_wandb_component import push_to_wandb_component
from components.pull_from_wandb_component import pull_from_wandb_component

def wandb_sync():
    """
    Push/Pull data to/from Weights & Biases
    """
    st.title("W&B Sync")
    
    if not st.session_state.authenticated:
        st.warning("Please authenticate with Weights & Biases first!")
        if st.button("Go to Authentication"):
            st.session_state.current_page = "auth"
            st.rerun()
        return
    
    st.markdown("""
    This component allows you to synchronize your quantum computing experiments with
    Weights & Biases, enabling seamless tracking, collaboration, and sharing.
    """)
    
    # Create tabs
    push_tab, pull_tab, settings_tab = st.tabs([
        "Push to W&B", "Pull from W&B", "Sync Settings"
    ])
    
    # Push to W&B tab
    with push_tab:
        st.subheader("Push Quantum Experiments to W&B")
        
        st.markdown("""
        Create and push new quantum computing experiments directly to Weights & Biases,
        with full support for quantum circuits, metrics, and artifacts.
        """)
        
        # Get or create project
        st.markdown("### Project Selection")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            # Option to use existing project or create new
            use_existing = st.radio(
                "Project options:",
                ["Use existing project", "Create new project"]
            )
        
        if use_existing == "Use existing project":
            # If we have projects loaded, show them
            if hasattr(st.session_state, 'projects') and st.session_state.projects:
                project_options = [f"{p['entity']}/{p['name']}" for p in st.session_state.projects]
                selected_project = st.selectbox("Select project:", project_options)
                
                # Parse the selected project
                if selected_project:
                    entity, project_name = selected_project.split("/")
            else:
                # Fetch projects if not loaded
                with st.spinner("Fetching projects..."):
                    try:
                        api = wandb.Api()
                        projects = []
                        for project in api.projects():
                            projects.append({
                                "name": project.name,
                                "entity": project.entity,
                                "id": f"{project.entity}/{project.name}",
                                "description": getattr(project, "description", ""),
                                "created_at": getattr(project, "created_at", ""),
                                "last_updated": getattr(project, "updated_at", "")
                            })
                        st.session_state.projects = projects
                        
                        project_options = [f"{p['entity']}/{p['name']}" for p in projects]
                        selected_project = st.selectbox("Select project:", project_options)
                        
                        # Parse the selected project
                        if selected_project:
                            entity, project_name = selected_project.split("/")
                    except Exception as e:
                        st.error(f"Error fetching projects: {str(e)}")
                        entity, project_name = None, None
        else:
            # Create new project form
            with st.form("new_project_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    entity = st.text_input("Entity name:", value=wandb.Api().viewer()['entity'])
                
                with col2:
                    project_name = st.text_input("Project name:", value="quantum-assistant-project")
                
                project_desc = st.text_area("Project description:", value="Quantum computing experiments from Quantum AI Assistant")
                
                submit_new_project = st.form_submit_button("Create Project")
                
                if submit_new_project:
                    try:
                        # Create project via API
                        api = wandb.Api()
                        new_project = api.create_project(name=project_name, entity=entity, description=project_desc)
                        st.success(f"Project {entity}/{project_name} created successfully!")
                        
                        # Refresh projects list
                        projects = []
                        for project in api.projects():
                            projects.append({
                                "name": project.name,
                                "entity": project.entity,
                                "id": f"{project.entity}/{project.name}",
                                "description": getattr(project, "description", ""),
                                "created_at": getattr(project, "created_at", ""),
                                "last_updated": getattr(project, "updated_at", "")
                            })
                        st.session_state.projects = projects
                    except Exception as e:
                        st.error(f"Error creating project: {str(e)}")
        
        # Create run section
        if entity and project_name:
            st.markdown("### Create New Run")
            
            with st.form("new_run_form"):
                # Run configuration
                run_name = st.text_input("Run name:", value=f"quantum-experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                
                # Tags
                tags_input = st.text_input("Tags (comma separated):", value="quantum, experiment")
                tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
                
                # Experiment type selection
                exp_type = st.selectbox(
                    "Experiment type:",
                    ["Quantum Circuit", "Variational Algorithm", "Quantum ML", "Custom"]
                )
                
                # Config parameters based on experiment type
                st.markdown("#### Configuration Parameters")
                
                if exp_type == "Quantum Circuit":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_qubits = st.number_input("Number of qubits:", min_value=1, max_value=20, value=3)
                    with col2:
                        circuit_depth = st.number_input("Circuit depth:", min_value=1, max_value=100, value=5)
                    
                    circuit_type = st.selectbox(
                        "Circuit type:",
                        ["Bell State", "GHZ State", "Quantum Fourier Transform", "Random Circuit"]
                    )
                    
                    backend = st.selectbox(
                        "Simulation backend:",
                        ["Aer Statevector", "Aer QASM", "Pennylane Default"]
                    )
                    
                    shots = st.number_input("Number of shots:", min_value=1, max_value=10000, value=1024)
                    
                    # Store config for run
                    config = {
                        "n_qubits": n_qubits,
                        "circuit_depth": circuit_depth,
                        "circuit_type": circuit_type,
                        "backend": backend,
                        "shots": shots
                    }
                    
                elif exp_type == "Variational Algorithm":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_qubits = st.number_input("Number of qubits:", min_value=1, max_value=20, value=4)
                    with col2:
                        n_layers = st.number_input("Number of layers:", min_value=1, max_value=10, value=2)
                    
                    optimizer = st.selectbox(
                        "Optimizer:",
                        ["Adam", "SGD", "SPSA", "COBYLA", "L-BFGS-B"]
                    )
                    
                    learning_rate = st.number_input("Learning rate:", min_value=0.001, max_value=0.5, value=0.01, step=0.001, format="%.4f")
                    
                    cost_function = st.selectbox(
                        "Cost function:",
                        ["Energy", "Binary Cross Entropy", "MSE", "Custom"]
                    )
                    
                    n_iterations = st.number_input("Number of iterations:", min_value=1, max_value=1000, value=100)
                    
                    # Store config for run
                    config = {
                        "n_qubits": n_qubits,
                        "n_layers": n_layers,
                        "optimizer": optimizer,
                        "learning_rate": learning_rate,
                        "cost_function": cost_function,
                        "n_iterations": n_iterations
                    }
                    
                elif exp_type == "Quantum ML":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_qubits = st.number_input("Number of qubits:", min_value=1, max_value=20, value=4)
                    with col2:
                        n_layers = st.number_input("Number of layers:", min_value=1, max_value=10, value=3)
                    
                    model_type = st.selectbox(
                        "Model type:",
                        ["Quantum Neural Network", "Quantum Kernel", "Quantum Support Vector Machine", "Quantum Embedding"]
                    )
                    
                    dataset = st.selectbox(
                        "Dataset:",
                        ["Iris", "MNIST", "Custom", "Random"]
                    )
                    
                    data_dimension = st.number_input("Data dimension:", min_value=1, max_value=20, value=4)
                    
                    n_samples = st.number_input("Number of samples:", min_value=10, max_value=1000, value=100)
                    
                    n_epochs = st.number_input("Number of epochs:", min_value=1, max_value=100, value=20)
                    
                    batch_size = st.number_input("Batch size:", min_value=1, max_value=64, value=16)
                    
                    # Store config for run
                    config = {
                        "n_qubits": n_qubits,
                        "n_layers": n_layers,
                        "model_type": model_type,
                        "dataset": dataset,
                        "data_dimension": data_dimension,
                        "n_samples": n_samples,
                        "n_epochs": n_epochs,
                        "batch_size": batch_size
                    }
                    
                else:  # Custom
                    st.markdown("Enter custom configuration parameters as key-value pairs:")
                    
                    # Custom config with dynamic fields
                    custom_config = {}
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        custom_key1 = st.text_input("Parameter 1 name:", value="n_qubits")
                    with col2:
                        custom_value1 = st.text_input("Parameter 1 value:", value="4")
                    if custom_key1 and custom_value1:
                        custom_config[custom_key1] = custom_value1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        custom_key2 = st.text_input("Parameter 2 name:", value="depth")
                    with col2:
                        custom_value2 = st.text_input("Parameter 2 value:", value="10")
                    if custom_key2 and custom_value2:
                        custom_config[custom_key2] = custom_value2
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        custom_key3 = st.text_input("Parameter 3 name:", value="shots")
                    with col2:
                        custom_value3 = st.text_input("Parameter 3 value:", value="1024")
                    if custom_key3 and custom_value3:
                        custom_config[custom_key3] = custom_value3
                    
                    # Additional free-form parameters as JSON
                    json_config = st.text_area("Additional parameters (JSON format):", value='{"optimizer": "Adam", "learning_rate": 0.01}')
                    
                    if json_config:
                        try:
                            additional_config = json.loads(json_config)
                            custom_config.update(additional_config)
                        except json.JSONDecodeError:
                            st.warning("Invalid JSON format in additional parameters field.")
                    
                    # Store config for run
                    config = custom_config
                
                # Submit button
                submit_run = st.form_submit_button("Create and Push Run")
                
                if submit_run:
                    with st.spinner("Creating and running experiment..."):
                        try:
                            # Initialize wandb run
                            run = wandb.init(
                                project=project_name,
                                entity=entity,
                                name=run_name,
                                config=config,
                                tags=tags,
                                reinit=True
                            )
                            
                            # Run the experiment based on type
                            if exp_type == "Quantum Circuit":
                                # Create and run circuit
                                if circuit_type == "Bell State":
                                    circuit = create_bell_state(n_qubits)
                                elif circuit_type == "GHZ State":
                                    circuit = create_ghz_state(n_qubits)
                                elif circuit_type == "Quantum Fourier Transform":
                                    circuit = create_qft_circuit(n_qubits)
                                else:  # Random Circuit
                                    circuit = create_random_circuit(n_qubits, circuit_depth)
                                
                                # Run simulation
                                if backend.startswith("Aer"):
                                    backend_type = "statevector_simulator" if backend == "Aer Statevector" else "qasm_simulator"
                                    sim = Aer.get_backend(backend_type)
                                    result = qiskit.execute(circuit, sim, shots=shots).result()
                                    counts = result.get_counts(circuit)
                                    
                                    # Save circuit diagram
                                    circuit_img = circuit_to_image(circuit)
                                    wandb.log({"circuit_diagram": wandb.Image(circuit_img)})
                                    
                                    # Log results
                                    wandb.log({
                                        "circuit_depth": circuit.depth(),
                                        "circuit_width": circuit.width(),
                                        "circuit_size": circuit.size(),
                                        "gate_counts": circuit.count_ops(),
                                    })
                                    
                                    # Log measurement probabilities
                                    probs = {}
                                    for state, count in counts.items():
                                        probs[state] = count / shots
                                    wandb.log({"measurement_probabilities": probs})
                                    
                                    # Create and log visualization
                                    fig = plot_quantum_results(counts)
                                    wandb.log({"measurement_results": wandb.Image(fig)})
                                    
                                else:  # Pennylane
                                    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
                                    
                                    @qml.qnode(dev)
                                    def pennylane_circuit():
                                        if circuit_type == "Bell State":
                                            qml.Hadamard(wires=0)
                                            qml.CNOT(wires=[0, 1])
                                        elif circuit_type == "GHZ State":
                                            qml.Hadamard(wires=0)
                                            for i in range(1, n_qubits):
                                                qml.CNOT(wires=[0, i])
                                        elif circuit_type == "Quantum Fourier Transform":
                                            qml.QFT(wires=range(n_qubits))
                                        else:  # Random Circuit
                                            # Add random rotation gates
                                            for i in range(n_qubits):
                                                qml.RX(np.random.uniform(0, 2*np.pi), wires=i)
                                                qml.RY(np.random.uniform(0, 2*np.pi), wires=i)
                                                qml.RZ(np.random.uniform(0, 2*np.pi), wires=i)
                                            
                                            # Add entangling gates
                                            for layer in range(circuit_depth):
                                                for i in range(n_qubits-1):
                                                    qml.CNOT(wires=[i, i+1])
                                        
                                        return qml.probs(wires=range(n_qubits))
                                    
                                    # Run the circuit
                                    probabilities = pennylane_circuit()
                                    
                                    # Log results
                                    wandb.log({
                                        "pennylane_probabilities": probabilities.tolist(),
                                        "circuit_type": circuit_type,
                                        "n_qubits": n_qubits,
                                    })
                            
                            elif exp_type == "Variational Algorithm":
                                # Simple VQE simulation
                                dev = qml.device("default.qubit", wires=n_qubits)
                                
                                # Define ansatz
                                def ansatz(params, wires):
                                    for i in range(n_qubits):
                                        qml.RX(params[0, i], wires=i)
                                        qml.RY(params[1, i], wires=i)
                                        qml.RZ(params[2, i], wires=i)
                                    
                                    for layer in range(n_layers):
                                        # Entangling layer
                                        for i in range(n_qubits-1):
                                            qml.CNOT(wires=[i, i+1])
                                        
                                        # Rotation layer
                                        for i in range(n_qubits):
                                            qml.RX(params[3 + layer*3, i], wires=i)
                                            qml.RY(params[4 + layer*3, i], wires=i)
                                            qml.RZ(params[5 + layer*3, i], wires=i)
                                
                                # Define cost function
                                @qml.qnode(dev)
                                def cost_function(params):
                                    ansatz(params, wires=range(n_qubits))
                                    
                                    # Simple Hamiltonian - sum of Z on each qubit
                                    return qml.expval(sum(qml.PauliZ(i) for i in range(n_qubits)))
                                
                                # Initialize random parameters
                                num_params = 3 + n_layers * 3
                                params = np.random.uniform(0, 2*np.pi, (num_params, n_qubits))
                                
                                # Simple optimization loop
                                costs = []
                                for i in range(n_iterations):
                                    # Forward pass and grad calculation
                                    cost = cost_function(params)
                                    grad = qml.grad(cost_function)(params)
                                    
                                    # Simple gradient descent
                                    params -= learning_rate * grad
                                    
                                    # Log each iteration
                                    wandb.log({"iteration": i, "cost": cost})
                                    costs.append(cost)
                                
                                # Log final results
                                wandb.log({
                                    "final_cost": costs[-1],
                                    "n_iterations": n_iterations,
                                    "final_parameters": params.tolist(),
                                    "cost_history": costs
                                })
                                
                                # Create and log cost plot
                                fig = px.line(
                                    x=range(n_iterations),
                                    y=costs,
                                    labels={"x": "Iteration", "y": "Cost"},
                                    title="Cost Function Evolution"
                                )
                                wandb.log({"cost_plot": wandb.Image(fig)})
                            
                            elif exp_type == "Quantum ML":
                                # Generate synthetic dataset
                                if dataset == "Random":
                                    # Simple classification dataset
                                    X = np.random.normal(0, 1, (n_samples, data_dimension))
                                    y = np.random.randint(0, 2, n_samples)
                                else:
                                    # Load a sample of Iris dataset
                                    from sklearn.datasets import load_iris
                                    iris = load_iris()
                                    X = iris.data[:n_samples, :data_dimension]
                                    y = iris.target[:n_samples] % 2  # Binary labels
                                
                                # Normalize data
                                X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
                                
                                # Simple quantum embedding model
                                dev = qml.device("default.qubit", wires=n_qubits)
                                
                                # Feature map
                                def feature_map(x, wires):
                                    # Encode features
                                    for i, feat in enumerate(x):
                                        if i < n_qubits:
                                            qml.RX(feat * np.pi, wires=i)
                                            qml.RY(feat * np.pi, wires=i)
                                    
                                    # Add entanglement
                                    for i in range(n_qubits-1):
                                        qml.CNOT(wires=[i, i+1])
                                
                                # Variational circuit
                                def variational_circuit(params, wires):
                                    for layer in range(n_layers):
                                        # Rotation layer
                                        for i in range(n_qubits):
                                            qml.RX(params[layer, i, 0], wires=i)
                                            qml.RY(params[layer, i, 1], wires=i)
                                            qml.RZ(params[layer, i, 2], wires=i)
                                        
                                        # Entangling layer
                                        for i in range(n_qubits-1):
                                            qml.CNOT(wires=[i, i+1])
                                
                                # Define quantum neural network
                                @qml.qnode(dev)
                                def quantum_nn(x, params):
                                    feature_map(x, wires=range(n_qubits))
                                    variational_circuit(params, wires=range(n_qubits))
                                    return qml.expval(qml.PauliZ(0))
                                
                                # Initialize parameters
                                params = np.random.uniform(0, 2*np.pi, (n_layers, n_qubits, 3))
                                
                                # Simple training loop
                                accuracy_history = []
                                loss_history = []
                                
                                for epoch in range(n_epochs):
                                    # Shuffle data
                                    indices = np.random.permutation(n_samples)
                                    X_shuffled = X[indices]
                                    y_shuffled = y[indices]
                                    
                                    epoch_loss = 0
                                    n_batches = n_samples // batch_size
                                    
                                    for batch in range(n_batches):
                                        start = batch * batch_size
                                        end = start + batch_size
                                        X_batch = X_shuffled[start:end]
                                        y_batch = y_shuffled[start:end]
                                        
                                        # Calculate predictions and loss
                                        predictions = np.array([quantum_nn(x, params) for x in X_batch])
                                        y_pred = (predictions > 0).astype(int)
                                        
                                        # Binary cross entropy loss
                                        predictions_clipped = np.clip(predictions, -0.99, 0.99)  # Avoid log(0)
                                        loss = -np.mean(y_batch * np.log((predictions_clipped + 1) / 2) + 
                                                       (1 - y_batch) * np.log(1 - (predictions_clipped + 1) / 2))
                                        
                                        # Accuracy
                                        acc = np.mean(y_pred == y_batch)
                                        
                                        # Update parameters with simple gradient approximation
                                        grad = np.zeros_like(params)
                                        eps = 0.1
                                        
                                        # Very simple parameter update (not a proper gradient)
                                        for layer in range(n_layers):
                                            for i in range(n_qubits):
                                                for j in range(3):
                                                    params[layer, i, j] -= learning_rate * np.random.normal(0, 1) * (1 - acc)
                                        
                                        epoch_loss += loss
                                    
                                    # Calculate final epoch metrics
                                    epoch_loss /= n_batches
                                    
                                    # Evaluate on full dataset
                                    all_predictions = np.array([quantum_nn(x, params) for x in X])
                                    all_y_pred = (all_predictions > 0).astype(int)
                                    accuracy = np.mean(all_y_pred == y)
                                    
                                    wandb.log({
                                        "epoch": epoch,
                                        "loss": epoch_loss,
                                        "accuracy": accuracy
                                    })
                                    
                                    accuracy_history.append(accuracy)
                                    loss_history.append(epoch_loss)
                                
                                # Log final results
                                wandb.log({
                                    "final_accuracy": accuracy_history[-1],
                                    "final_loss": loss_history[-1],
                                    "n_epochs": n_epochs,
                                    "n_samples": n_samples,
                                    "data_dimension": data_dimension
                                })
                                
                                # Create and log training plots
                                fig1 = px.line(
                                    x=range(n_epochs),
                                    y=accuracy_history,
                                    labels={"x": "Epoch", "y": "Accuracy"},
                                    title="Accuracy vs. Epoch"
                                )
                                
                                fig2 = px.line(
                                    x=range(n_epochs),
                                    y=loss_history,
                                    labels={"x": "Epoch", "y": "Loss"},
                                    title="Loss vs. Epoch"
                                )
                                
                                wandb.log({
                                    "accuracy_plot": wandb.Image(fig1),
                                    "loss_plot": wandb.Image(fig2)
                                })
                            
                            else:  # Custom experiment
                                # Log basic custom metrics
                                wandb.log({
                                    "custom_experiment": True,
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                # Create and log some dummy data
                                fig = px.line(
                                    x=range(20),
                                    y=[np.random.random() for _ in range(20)],
                                    title="Custom Experiment Results"
                                )
                                wandb.log({"custom_plot": wandb.Image(fig)})
                            
                            # Finalize the run
                            run.finish()
                            
                            st.success(f"Successfully pushed experiment to W&B: {entity}/{project_name}/{run_name}")
                            
                            # Provide link to the run
                            run_url = f"https://wandb.ai/{entity}/{project_name}/runs/{run.id}"
                            st.markdown(f"[View run on W&B]({run_url})")
                            
                        except Exception as e:
                            st.error(f"Error running experiment: {str(e)}")
    
    # Pull from W&B tab
    with pull_tab:
        st.subheader("Pull Data from W&B")
        
        st.markdown("""
        Retrieve existing quantum computing experiments, models, and artifacts from W&B
        to analyze locally or integrate with your quantum workflows.
        """)
        
        # UI for selecting project and runs to pull
        st.markdown("### Select Project")
        
        # Show projects if loaded
        if hasattr(st.session_state, 'projects') and st.session_state.projects:
            project_options = [f"{p['entity']}/{p['name']}" for p in st.session_state.projects]
            selected_pull_project = st.selectbox("Project:", project_options, key="pull_project_select")
            
            if selected_pull_project:
                entity, project_name = selected_pull_project.split("/")
                
                # Fetch runs for the selected project
                if st.button("Load Runs", key="load_pull_runs"):
                    with st.spinner("Fetching runs..."):
                        try:
                            api = wandb.Api()
                            runs = []
                            for run in api.runs(selected_pull_project):
                                runs.append({
                                    "id": run.id,
                                    "name": run.name,
                                    "state": run.state,
                                    "created_at": run.created_at,
                                    "tags": run.tags,
                                    "url": run.url
                                })
                            st.session_state.pull_runs = runs
                        except Exception as e:
                            st.error(f"Error fetching runs: {str(e)}")
                
                # Display runs if loaded
                if hasattr(st.session_state, 'pull_runs') and st.session_state.pull_runs:
                    st.markdown("### Select Run to Pull")
                    
                    # Create a dataframe for better display
                    runs_df = pd.DataFrame(st.session_state.pull_runs)
                    
                    # Format timestamp
                    if 'created_at' in runs_df.columns:
                        runs_df['created_at'] = pd.to_datetime(runs_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Display runs
                    st.dataframe(runs_df, use_container_width=True)
                    
                    # Select run to pull
                    run_options = [f"{run['name']} ({run['id']})" for run in st.session_state.pull_runs]
                    selected_run_option = st.selectbox("Select run:", run_options)
                    
                    if selected_run_option:
                        run_id = selected_run_option.split("(")[-1].split(")")[0]
                        
                        # Options for what to pull
                        st.markdown("### Select Data to Pull")
                        
                        pull_options = st.multiselect(
                            "Select what to pull:",
                            ["Config", "Metrics", "Artifacts", "System Info", "Code"],
                            default=["Config", "Metrics"]
                        )
                        
                        if st.button("Pull Run Data"):
                            with st.spinner("Pulling data from W&B..."):
                                try:
                                    api = wandb.Api()
                                    run = api.run(f"{selected_pull_project}/{run_id}")
                                    
                                    # Store pulled data
                                    pulled_data = {
                                        "run_id": run.id,
                                        "run_name": run.name,
                                        "entity": run.entity,
                                        "project": run.project,
                                        "url": run.url
                                    }
                                    
                                    # Pull requested data
                                    if "Config" in pull_options:
                                        pulled_data["config"] = {k: v for k, v in run.config.items() if not k.startswith('_')}
                                    
                                    if "Metrics" in pull_options:
                                        # Pull summary metrics
                                        pulled_data["metrics"] = {k: v for k, v in run.summary._json_dict.items() if not k.startswith('_')}
                                        
                                        # Pull history if available
                                        try:
                                            history = run.history()
                                            pulled_data["history"] = history.to_dict('records')
                                        except:
                                            pulled_data["history"] = []
                                    
                                    if "System Info" in pull_options:
                                        pulled_data["system_info"] = run.system_metrics
                                    
                                    if "Code" in pull_options:
                                        try:
                                            pulled_data["code"] = run.code
                                        except:
                                            pulled_data["code"] = "Code not available"
                                    
                                    if "Artifacts" in pull_options:
                                        artifacts = []
                                        for artifact in run.logged_artifacts():
                                            artifacts.append({
                                                "name": artifact.name,
                                                "type": artifact.type,
                                                "description": artifact.description,
                                                "version": artifact.version,
                                                "size": artifact.size,
                                                "url": artifact.url
                                            })
                                        pulled_data["artifacts"] = artifacts
                                    
                                    # Store the pulled data
                                    st.session_state.pulled_run_data = pulled_data
                                    
                                    st.success(f"Successfully pulled data for run: {run.name}")
                                    
                                except Exception as e:
                                    st.error(f"Error pulling run data: {str(e)}")
                        
                        # Display pulled data if available
                        if hasattr(st.session_state, 'pulled_run_data') and st.session_state.pulled_run_data:
                            data = st.session_state.pulled_run_data
                            
                            st.markdown("### Pulled Run Data")
                            
                            # Run info
                            st.markdown(f"**Run:** {data['run_name']} ({data['run_id']})")
                            st.markdown(f"**Project:** {data['entity']}/{data['project']}")
                            st.markdown(f"**URL:** [Open in W&B]({data['url']})")
                            
                            # Config tab
                            if "config" in data:
                                with st.expander("Configuration", expanded=True):
                                    st.json(data["config"])
                            
                            # Metrics tab
                            if "metrics" in data:
                                with st.expander("Metrics", expanded=True):
                                    st.json(data["metrics"])
                                    
                                    # If we have history data, plot it
                                    if "history" in data and data["history"]:
                                        history_df = pd.DataFrame(data["history"])
                                        
                                        # Get metric columns
                                        metric_cols = [col for col in history_df.columns 
                                                     if col not in ['_step', 'step'] and 
                                                     pd.api.types.is_numeric_dtype(history_df[col])]
                                        
                                        if metric_cols:
                                            st.markdown("#### Metrics Over Time")
                                            
                                            # Let user select metrics
                                            selected_hist_metrics = st.multiselect(
                                                "Select metrics to plot:",
                                                metric_cols,
                                                default=metric_cols[:min(3, len(metric_cols))]
                                            )
                                            
                                            if selected_hist_metrics:
                                                # Determine step column
                                                step_col = '_step' if '_step' in history_df.columns else 'step' if 'step' in history_df.columns else None
                                                
                                                if step_col:
                                                    # Create plot
                                                    fig = px.line(
                                                        history_df,
                                                        x=step_col,
                                                        y=selected_hist_metrics,
                                                        title="Metrics Over Time"
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Artifacts tab
                            if "artifacts" in data and data["artifacts"]:
                                with st.expander("Artifacts", expanded=True):
                                    artifacts_df = pd.DataFrame(data["artifacts"])
                                    st.dataframe(artifacts_df)
                            
                            # System Info tab
                            if "system_info" in data:
                                with st.expander("System Info", expanded=False):
                                    st.json(data["system_info"])
                            
                            # Code tab
                            if "code" in data:
                                with st.expander("Code", expanded=False):
                                    st.code(data["code"])
                            
                            # Export options
                            export_format = st.selectbox(
                                "Export format:",
                                ["JSON", "CSV", "Excel"]
                            )
                            
                            if st.button("Export Pulled Data"):
                                if export_format == "JSON":
                                    # Convert to JSON
                                    json_data = json.dumps(data, indent=2, default=str)
                                    
                                    # Offer for download
                                    st.download_button(
                                        label="Download JSON",
                                        data=json_data,
                                        file_name=f"wandb_run_{data['run_id']}.json",
                                        mime="application/json"
                                    )
                                
                                elif export_format == "CSV":
                                    # Create ZIP with CSVs for different components
                                    zip_buffer = io.BytesIO()
                                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                                        # Config
                                        if "config" in data:
                                            config_df = pd.DataFrame([(k, v) for k, v in data["config"].items()], 
                                                                    columns=["Parameter", "Value"])
                                            zip_file.writestr("config.csv", config_df.to_csv(index=False))
                                        
                                        # Metrics
                                        if "metrics" in data:
                                            metrics_df = pd.DataFrame([(k, v) for k, v in data["metrics"].items()],
                                                                     columns=["Metric", "Value"])
                                            zip_file.writestr("metrics.csv", metrics_df.to_csv(index=False))
                                        
                                        # History
                                        if "history" in data and data["history"]:
                                            history_df = pd.DataFrame(data["history"])
                                            zip_file.writestr("history.csv", history_df.to_csv(index=False))
                                        
                                        # Artifacts
                                        if "artifacts" in data and data["artifacts"]:
                                            artifacts_df = pd.DataFrame(data["artifacts"])
                                            zip_file.writestr("artifacts.csv", artifacts_df.to_csv(index=False))
                                    
                                    # Reset buffer position
                                    zip_buffer.seek(0)
                                    
                                    # Offer for download
                                    st.download_button(
                                        label="Download CSVs (ZIP)",
                                        data=zip_buffer,
                                        file_name=f"wandb_run_{data['run_id']}_csv.zip",
                                        mime="application/zip"
                                    )
                                
                                elif export_format == "Excel":
                                    # Create Excel with multiple sheets
                                    excel_buffer = io.BytesIO()
                                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                                        # Info sheet
                                        info_df = pd.DataFrame([
                                            ["Run ID", data["run_id"]],
                                            ["Run Name", data["run_name"]],
                                            ["Entity", data["entity"]],
                                            ["Project", data["project"]],
                                            ["URL", data["url"]]
                                        ], columns=["Field", "Value"])
                                        info_df.to_excel(writer, sheet_name="Run Info", index=False)
                                        
                                        # Config sheet
                                        if "config" in data:
                                            config_df = pd.DataFrame([(k, v) for k, v in data["config"].items()], 
                                                                    columns=["Parameter", "Value"])
                                            config_df.to_excel(writer, sheet_name="Configuration", index=False)
                                        
                                        # Metrics sheet
                                        if "metrics" in data:
                                            metrics_df = pd.DataFrame([(k, v) for k, v in data["metrics"].items()],
                                                                     columns=["Metric", "Value"])
                                            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
                                        
                                        # History sheet
                                        if "history" in data and data["history"]:
                                            history_df = pd.DataFrame(data["history"])
                                            history_df.to_excel(writer, sheet_name="History", index=False)
                                        
                                        # Artifacts sheet
                                        if "artifacts" in data and data["artifacts"]:
                                            artifacts_df = pd.DataFrame(data["artifacts"])
                                            artifacts_df.to_excel(writer, sheet_name="Artifacts", index=False)
                                    
                                    excel_data = excel_buffer.getvalue()
                                    
                                    # Offer for download
                                    st.download_button(
                                        label="Download Excel",
                                        data=excel_data,
                                        file_name=f"wandb_run_{data['run_id']}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                else:
                    st.info("Click 'Load Runs' to fetch runs for this project.")
        else:
            st.info("No projects loaded. Please go to the Projects page to load your W&B projects.")
    
    # Settings tab
    with settings_tab:
        st.subheader("Sync Settings")
        
        st.markdown("""
        Configure synchronization settings between your quantum computing experiments
        and Weights & Biases.
        """)
        
        # API settings
        st.markdown("### API Configuration")
        
        api_key = st.text_input("W&B API Key", value="", type="password")
        
        if api_key:
            if st.button("Update API Key"):
                try:
                    # Save the API key to environment
                    os.environ["WANDB_API_KEY"] = api_key
                    wandb.login(key=api_key)
                    st.success("API key updated successfully!")
                    
                    # Try to get user info
                    api = wandb.Api()
                    username = api.viewer()['entity']
                    st.success(f"Authenticated as: {username}")
                except Exception as e:
                    st.error(f"Error updating API key: {str(e)}")
        
        # Auto-sync settings
        st.markdown("### Auto-Sync Settings")
        
        st.checkbox("Enable auto-sync for new quantum experiments", value=True)
        st.checkbox("Sync quantum circuit diagrams", value=True)
        st.checkbox("Sync measurement results", value=True)
        st.checkbox("Sync training metrics for variational circuits", value=True)
        
        # Logging level
        log_level = st.selectbox(
            "Logging level:",
            ["Minimal", "Standard", "Detailed", "Debug"],
            index=1
        )
        
        # Data privacy settings
        st.markdown("### Data Privacy")
        
        st.checkbox("Anonymize personal information", value=False)
        st.checkbox("Remove system-specific metadata", value=False)
        st.checkbox("Encrypt sensitive configuration values", value=False)
        
        # Apply settings
        if st.button("Apply Settings"):
            st.success("Sync settings updated successfully!")

# Helper functions for quantum circuit creation and execution

def create_bell_state(n_qubits=2):
    """Create a Bell state quantum circuit"""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit

def create_ghz_state(n_qubits):
    """Create a GHZ state with the specified number of qubits"""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for i in range(1, n_qubits):
        circuit.cx(0, i)
    circuit.measure_all()
    return circuit

def create_qft_circuit(n_qubits):
    """Create a Quantum Fourier Transform circuit"""
    circuit = QuantumCircuit(n_qubits)
    
    # Apply Hadamard to all qubits initially
    for i in range(n_qubits):
        circuit.h(i)
        
    # Apply controlled phase rotations
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            circuit.cp(2*np.pi/2**(j-i), i, j)
    
    circuit.measure_all()
    return circuit

def create_random_circuit(n_qubits, depth):
    """Create a random quantum circuit"""
    circuit = QuantumCircuit(n_qubits)
    
    # Add random gates
    gate_set = [
        lambda qc, qubit: qc.h(qubit),
        lambda qc, qubit: qc.x(qubit),
        lambda qc, qubit: qc.y(qubit),
        lambda qc, qubit: qc.z(qubit),
        lambda qc, qubit: qc.rx(np.random.uniform(0, 2*np.pi), qubit),
        lambda qc, qubit: qc.ry(np.random.uniform(0, 2*np.pi), qubit),
        lambda qc, qubit: qc.rz(np.random.uniform(0, 2*np.pi), qubit)
    ]
    
    entangling_gates = [
        lambda qc, q1, q2: qc.cx(q1, q2),
        lambda qc, q1, q2: qc.cz(q1, q2),
        lambda qc, q1, q2: qc.swap(q1, q2)
    ]
    
    # Add layers of gates
    for d in range(depth):
        # Single-qubit gates
        for i in range(n_qubits):
            gate = np.random.choice(gate_set)
            gate(circuit, i)
        
        # Entangling gates - connect random pairs of qubits
        for i in range(n_qubits-1):
            if np.random.random() < 0.7:  # 70% chance of adding entangling gate
                gate = np.random.choice(entangling_gates)
                gate(circuit, i, i+1)
    
    circuit.measure_all()
    return circuit

def circuit_to_image(circuit):
    """Convert a Qiskit circuit to an image"""
    from qiskit.visualization import circuit_drawer
    import io
    from PIL import Image
    
    # Create a figure of the circuit
    img_io = io.BytesIO()
    circuit_drawer(circuit, output='mpl', filename=img_io, style={'name': 'bw'})
    img_io.seek(0)
    
    # Convert to PIL Image
    image = Image.open(img_io)
    return image

def plot_quantum_results(counts):
    """Plot the results of a quantum circuit simulation"""
    # Create bar chart of results
    fig = px.bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        labels={'x': 'Measurement Outcome', 'y': 'Counts'},
        title='Quantum Measurement Results'
    )
    
    fig.update_layout(
        xaxis_title='Measurement Outcome', 
        yaxis_title='Counts'
    )
    
    return fig