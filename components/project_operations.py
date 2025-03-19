import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wandb
from datetime import datetime
import json

def project_operations_component():
    """
    Component for managing W&B projects and posting quantum experiments to projects
    """
    st.subheader("Project Operations")
    
    if "wandb_authenticated" not in st.session_state or not st.session_state.wandb_authenticated:
        st.warning("Please authenticate with Weights & Biases first!")
        st.button("Go to Authentication", on_click=lambda: setattr(st.session_state, "page", "auth"))
        return
    
    # Create tabs for different project operations
    project_tabs = st.tabs(["Post to Project", "Manage Projects", "Project Stats"])
    
    with project_tabs[0]:
        post_to_project()
    
    with project_tabs[1]:
        manage_projects()
    
    with project_tabs[2]:
        project_stats()

def post_to_project():
    """Post quantum experiments directly to W&B projects"""
    st.markdown("### Post to Project")
    st.markdown("Create and post quantum experiments directly to your W&B projects")
    
    # Project selection
    try:
        api = wandb.Api()
        projects = []
        
        with st.spinner("Loading projects..."):
            for project in api.projects():
                projects.append({
                    "name": project.name,
                    "entity": project.entity,
                    "id": f"{project.entity}/{project.name}",
                    "description": getattr(project, "description", ""),
                })
        
        if not projects:
            st.info("No projects found. Create a new project to get started.")
            return
        
        # Project selector
        project_options = [f"{p['entity']}/{p['name']}" for p in projects]
        selected_project = st.selectbox("Select project:", project_options)
        
        if selected_project:
            entity, project_name = selected_project.split("/")
            
            # Experiment configuration
            with st.form("quantum_experiment_form"):
                st.markdown("### Experiment Configuration")
                
                # Basic experiment details
                exp_name = st.text_input("Experiment name:", value=f"quantum-experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
                
                # Experiment type
                exp_type = st.selectbox(
                    "Experiment type:",
                    ["Bell State", "GHZ State", "QFT", "Custom Circuit"]
                )
                
                # Parameters
                col1, col2 = st.columns(2)
                with col1:
                    n_qubits = st.number_input("Number of qubits:", min_value=1, max_value=10, value=2)
                with col2:
                    shots = st.number_input("Number of shots:", min_value=1, max_value=10000, value=1024)
                
                tags = st.text_input("Tags (comma separated):", value="quantum")
                tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
                
                # Submit button
                submit_exp = st.form_submit_button("Create and Post Experiment")
                
                if submit_exp:
                    with st.spinner("Creating and posting experiment..."):
                        from push_to_wandb import create_bell_state, create_ghz_state, create_qft_circuit
                        from push_to_wandb import circuit_to_image, run_quantum_circuit, plot_quantum_results
                        import qiskit
                        from qiskit import QuantumCircuit
                        
                        # Create circuit based on experiment type
                        if exp_type == "Bell State":
                            circuit = create_bell_state()
                            circuit_type = "Bell State"
                        elif exp_type == "GHZ State":
                            circuit = create_ghz_state(n_qubits)
                            circuit_type = "GHZ State"
                        elif exp_type == "QFT":
                            circuit = create_qft_circuit(n_qubits)
                            circuit_type = "Quantum Fourier Transform"
                        else:
                            # Custom circuit
                            circuit = QuantumCircuit(n_qubits)
                            circuit.h(range(n_qubits))
                            for i in range(n_qubits-1):
                                circuit.cx(i, i+1)
                            circuit.measure_all()
                            circuit_type = "Custom Circuit"
                        
                        # Run the circuit
                        counts = run_quantum_circuit(circuit, shots=shots)
                        
                        # Generate images
                        circuit_img = circuit_to_image(circuit)
                        results_img = plot_quantum_results(counts)
                        
                        # Initialize wandb run in the selected project
                        run = wandb.init(
                            project=project_name,
                            entity=entity,
                            name=exp_name,
                            config={
                                "circuit_type": circuit_type,
                                "n_qubits": n_qubits,
                                "shots": shots,
                                "backend": "qasm_simulator"
                            },
                            tags=tag_list,
                            reinit=True
                        )
                        
                        # Log data to wandb
                        wandb.log({
                            "circuit_diagram": wandb.Image(circuit_img),
                            "measurement_results": wandb.Image(results_img),
                            "circuit_depth": circuit.depth(),
                            "circuit_width": circuit.width(),
                            "circuit_size": circuit.size(),
                            "gate_counts": str(circuit.count_ops()),
                        })
                        
                        # Convert counts to probabilities and log as table
                        probs = {}
                        total = sum(counts.values())
                        for state, count in counts.items():
                            probs[state] = count / total
                        
                        prob_data = [[state, prob] for state, prob in probs.items()]
                        prob_table = wandb.Table(columns=["State", "Probability"], data=prob_data)
                        wandb.log({"probability_table": prob_table})
                        
                        # Finish the run
                        wandb.finish()
                        
                        st.success(f"Experiment '{exp_name}' successfully posted to {selected_project}!")
                        
                        # Show results
                        st.markdown("### Experiment Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(circuit_img, caption="Circuit Diagram")
                        with col2:
                            st.image(results_img, caption="Measurement Results")
                        
                        # Add link to the experiment
                        st.markdown(f"View experiment details: [W&B Dashboard](https://wandb.ai/{entity}/{project_name}/runs/{run.id})")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure you are properly authenticated with W&B.")

def manage_projects():
    """Manage W&B projects"""
    st.markdown("### Manage Projects")
    
    try:
        api = wandb.Api()
        
        # Create new project section
        st.subheader("Create New Project")
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
        
        # Existing projects
        st.subheader("Existing Projects")
        with st.spinner("Loading projects..."):
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
            
            if projects:
                project_df = pd.DataFrame(projects)
                st.dataframe(project_df)
            else:
                st.info("No projects found.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure you are properly authenticated with W&B.")

def project_stats():
    """Display project statistics"""
    st.markdown("### Project Statistics")
    
    try:
        api = wandb.Api()
        
        # Project selection
        projects = []
        for project in api.projects():
            projects.append({
                "name": project.name,
                "entity": project.entity,
                "id": f"{project.entity}/{project.name}"
            })
        
        if not projects:
            st.info("No projects found. Create a new project to get started.")
            return
        
        project_options = [f"{p['entity']}/{p['name']}" for p in projects]
        selected_project = st.selectbox("Select project for statistics:", project_options)
        
        if selected_project:
            entity, project_name = selected_project.split("/")
            
            # Fetch project stats
            with st.spinner("Loading project statistics..."):
                runs = api.runs(path=f"{entity}/{project_name}")
                run_data = []
                
                for run in runs:
                    run_data.append({
                        "id": run.id,
                        "name": run.name,
                        "state": run.state,
                        "created_at": run.created_at,
                        "tags": ", ".join(run.tags) if run.tags else "",
                        "qubits": run.config.get("n_qubits", 0) if hasattr(run, "config") else 0,
                        "circuit_type": run.config.get("circuit_type", "") if hasattr(run, "config") else ""
                    })
                
                if run_data:
                    # Display runs table
                    st.subheader("Project Runs")
                    runs_df = pd.DataFrame(run_data)
                    st.dataframe(runs_df)
                    
                    # Generate statistics
                    st.subheader("Project Visualizations")
                    
                    # Circuit type distribution
                    if "circuit_type" in runs_df.columns and not runs_df["circuit_type"].isnull().all():
                        circuit_counts = runs_df["circuit_type"].value_counts().reset_index()
                        circuit_counts.columns = ["Circuit Type", "Count"]
                        
                        fig = px.pie(circuit_counts, values="Count", names="Circuit Type", 
                                     title="Circuit Type Distribution")
                        st.plotly_chart(fig)
                    
                    # Qubit distribution
                    if "qubits" in runs_df.columns and not runs_df["qubits"].isnull().all():
                        qubit_counts = runs_df["qubits"].value_counts().reset_index()
                        qubit_counts.columns = ["Number of Qubits", "Count"]
                        qubit_counts = qubit_counts.sort_values("Number of Qubits")
                        
                        fig = px.bar(qubit_counts, x="Number of Qubits", y="Count",
                                    title="Qubit Distribution")
                        st.plotly_chart(fig)
                    
                    # Runs over time
                    if "created_at" in runs_df.columns:
                        runs_df["created_at"] = pd.to_datetime(runs_df["created_at"])
                        runs_df["date"] = runs_df["created_at"].dt.date
                        runs_by_date = runs_df.groupby("date").size().reset_index()
                        runs_by_date.columns = ["Date", "Count"]
                        
                        fig = px.line(runs_by_date, x="Date", y="Count", 
                                      title="Runs Over Time")
                        st.plotly_chart(fig)
                else:
                    st.info("No runs found in this project.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure you are properly authenticated with W&B.")