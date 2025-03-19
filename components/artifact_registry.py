"""
W&B Artifact Registry integration component for Quantum AI Assistant
Allows users to interact with the W&B Artifact Registry, upload quantum circuits,
and manage quantum models in the registry.
"""

import streamlit as st
import wandb
import os
import time
import tempfile
import json
import pandas as pd
from utils.key_generator import generate_unique_key

def artifact_registry():
    """
    W&B Artifact Registry component for managing quantum models and artifacts
    """
    st.title("W&B Artifact Registry")
    
    # Check if authenticated
    if not st.session_state.get("authenticated", False):
        st.warning("Please authenticate with W&B API first")
        return
    
    # Set the default organization
    if "registry_org" not in st.session_state:
        st.session_state.registry_org = "radosavlevici-ervin-quantum-org"
    
    org_input = st.text_input(
        "Organization", 
        value=st.session_state.registry_org,
        key=generate_unique_key("org_input")
    )
    
    if org_input != st.session_state.registry_org:
        st.session_state.registry_org = org_input
    
    # Create tabs for different registry operations
    tabs = st.tabs([
        "Browse Registry", 
        "Upload to Registry", 
        "Use from Registry", 
        "Manage Registry"
    ])
    
    # Browse Registry tab
    with tabs[0]:
        browse_artifact_registry()
    
    # Upload to Registry tab
    with tabs[1]:
        upload_to_registry()
    
    # Use from Registry tab
    with tabs[2]:
        use_from_registry()
    
    # Manage Registry tab
    with tabs[3]:
        manage_registry()

def browse_artifact_registry():
    """Browse the W&B Artifact Registry"""
    st.subheader("Browse Artifact Registry")
    
    org = st.session_state.registry_org
    
    # Add a refresh button
    if st.button("Refresh Registry", key=generate_unique_key("refresh_registry")):
        if "registry_artifacts" in st.session_state:
            del st.session_state.registry_artifacts
    
    # Fetch artifacts if not already in session state
    if "registry_artifacts" not in st.session_state:
        with st.spinner("Fetching artifacts from registry..."):
            try:
                api = wandb.Api()
                
                # Define collection types to look for
                collection_types = ["dataset", "model", "algorithm", "quantum-circuit"]
                
                artifacts = []
                for collection_type in collection_types:
                    try:
                        collection_artifacts = api.artifacts(
                            type=collection_type,
                            per_page=100,
                            entity=org
                        )
                        
                        for artifact in collection_artifacts:
                            artifacts.append({
                                "name": artifact.name,
                                "type": artifact.type,
                                "description": getattr(artifact, "description", ""),
                                "version": artifact.version,
                                "created_at": artifact.created_at,
                                "updated_at": getattr(artifact, "updated_at", artifact.created_at),
                                "size": getattr(artifact, "size", 0),
                                "metadata": getattr(artifact, "metadata", {}),
                                "id": artifact.id
                            })
                    except Exception as e:
                        st.error(f"Error fetching {collection_type} artifacts: {str(e)}")
                
                st.session_state.registry_artifacts = artifacts
                
            except Exception as e:
                st.error(f"Error connecting to registry: {str(e)}")
                return
    
    # Display the artifacts
    if st.session_state.get("registry_artifacts"):
        artifacts = st.session_state.registry_artifacts
        
        # Add filter by type
        artifact_types = list(set(a["type"] for a in artifacts))
        selected_type = st.selectbox(
            "Filter by Type", 
            ["All"] + artifact_types,
            key=generate_unique_key("artifact_type_filter")
        )
        
        # Add search box
        search_query = st.text_input(
            "Search Artifacts", 
            key=generate_unique_key("artifact_search")
        )
        
        # Filter artifacts
        filtered_artifacts = artifacts
        if selected_type != "All":
            filtered_artifacts = [a for a in filtered_artifacts if a["type"] == selected_type]
        
        if search_query:
            filtered_artifacts = [
                a for a in filtered_artifacts 
                if search_query.lower() in a["name"].lower() or 
                   search_query.lower() in a.get("description", "").lower()
            ]
        
        # Display as a table
        if filtered_artifacts:
            # Create a DataFrame for better display
            artifact_data = []
            for artifact in filtered_artifacts:
                artifact_data.append({
                    "Name": artifact["name"],
                    "Type": artifact["type"],
                    "Version": artifact["version"],
                    "Created": pd.to_datetime(artifact["created_at"]).strftime('%Y-%m-%d %H:%M') if artifact["created_at"] else "",
                    "Size": format_size(artifact["size"]) if artifact["size"] else "Unknown"
                })
            
            # Display as a table
            st.dataframe(pd.DataFrame(artifact_data), use_container_width=True)
            
            # Select artifact for detailed view
            artifact_options = [f"{a['name']}:{a['version']} ({a['type']})" for a in filtered_artifacts]
            selected_artifact_option = st.selectbox(
                "Select Artifact for Details", 
                artifact_options,
                key=generate_unique_key("artifact_select")
            )
            
            if selected_artifact_option:
                # Find the selected artifact
                artifact_name, artifact_version = selected_artifact_option.split(":", 1)[0], selected_artifact_option.split(":", 1)[1].split(" ")[0]
                selected_artifact = next((a for a in filtered_artifacts if a["name"] == artifact_name and a["version"] == artifact_version), None)
                
                if selected_artifact:
                    display_artifact_details(selected_artifact)
        else:
            st.info("No artifacts found matching the criteria.")
    else:
        st.info("No artifacts found in the registry.")

def upload_to_registry():
    """Upload artifacts to the W&B Registry"""
    st.subheader("Upload to Registry")
    
    org = st.session_state.registry_org
    
    with st.form("upload_form", clear_on_submit=False):
        # Artifact type selection
        artifact_type = st.selectbox(
            "Artifact Type",
            ["model", "dataset", "algorithm", "quantum-circuit"],
            key=generate_unique_key("upload_type")
        )
        
        # Artifact name
        artifact_name = st.text_input(
            "Artifact Name",
            key=generate_unique_key("upload_name")
        )
        
        # Artifact description
        artifact_description = st.text_area(
            "Description (optional)",
            key=generate_unique_key("upload_description")
        )
        
        # Artifact content
        content_type = st.radio(
            "Content Type",
            ["Quantum Circuit", "File Upload", "Text/JSON"],
            key=generate_unique_key("content_type")
        )
        
        # Show appropriate input method based on content type
        if content_type == "Quantum Circuit":
            # Add circuit creation interface here
            num_qubits = st.slider("Number of Qubits", 1, 10, 2)
            circuit_type = st.selectbox(
                "Circuit Type",
                ["Bell State", "GHZ State", "QFT", "Custom"],
                key=generate_unique_key("circuit_type")
            )
            
            # If custom, allow defining gates
            if circuit_type == "Custom":
                custom_circuit = st.text_area(
                    "Define Custom Circuit (Qiskit syntax)",
                    """from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()""",
                    height=200,
                    key=generate_unique_key("custom_circuit")
                )
        
        elif content_type == "File Upload":
            uploaded_file = st.file_uploader(
                "Upload File",
                type=["py", "json", "qasm", "txt", "ipynb", "pkl"],
                key=generate_unique_key("file_upload")
            )
        
        elif content_type == "Text/JSON":
            json_content = st.text_area(
                "Enter JSON or Text Content",
                "{\n    \"name\": \"quantum_model\",\n    \"parameters\": {\n        \"theta\": 0.5,\n        \"phi\": 1.2\n    }\n}",
                height=200,
                key=generate_unique_key("json_content")
            )
        
        # Metadata
        metadata_json = st.text_area(
            "Metadata (JSON, optional)",
            "{\n    \"framework\": \"qiskit\",\n    \"version\": \"0.30.0\"\n}",
            key=generate_unique_key("metadata")
        )
        
        # Submit button
        submit_button = st.form_submit_button("Upload to Registry")
        
        if submit_button:
            if not artifact_name:
                st.error("Please provide an artifact name")
                return
            
            with st.spinner("Uploading to artifact registry..."):
                try:
                    # Initialize metadata
                    try:
                        metadata = json.loads(metadata_json)
                    except:
                        metadata = {}
                    
                    # Set up wandb run
                    run = wandb.init(
                        project="quantum-artifact-registry",
                        entity=org,
                        job_type="artifact-upload",
                        config=metadata
                    )
                    
                    # Create a temporary directory for artifact files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        if content_type == "Quantum Circuit":
                            # Create the circuit based on selection
                            if circuit_type == "Custom":
                                # Save custom circuit code to a file
                                with open(os.path.join(temp_dir, "circuit.py"), "w") as f:
                                    f.write(custom_circuit)
                                
                                # Add execution info
                                with open(os.path.join(temp_dir, "README.md"), "w") as f:
                                    f.write(f"# Custom Quantum Circuit\n\n")
                                    f.write(f"Number of qubits: {num_qubits}\n\n")
                                    f.write("```python\n")
                                    f.write(custom_circuit)
                                    f.write("\n```\n")
                            else:
                                # Create predefined circuit
                                from qiskit import QuantumCircuit
                                
                                if circuit_type == "Bell State":
                                    qc = QuantumCircuit(2)
                                    qc.h(0)
                                    qc.cx(0, 1)
                                    qc.measure_all()
                                    circuit_name = "bell_state"
                                
                                elif circuit_type == "GHZ State":
                                    qc = QuantumCircuit(num_qubits)
                                    qc.h(0)
                                    for i in range(1, num_qubits):
                                        qc.cx(0, i)
                                    qc.measure_all()
                                    circuit_name = "ghz_state"
                                
                                elif circuit_type == "QFT":
                                    from qiskit.circuit.library import QFT
                                    qc = QFT(num_qubits)
                                    qc.measure_all()
                                    circuit_name = "qft"
                                
                                # Save circuit to QASM file
                                qasm_path = os.path.join(temp_dir, f"{circuit_name}.qasm")
                                with open(qasm_path, "w") as f:
                                    f.write(qc.qasm())
                                
                                # Save visualization of the circuit
                                circuit_png_path = os.path.join(temp_dir, f"{circuit_name}.png")
                                # This would require matplotlib in a real environment
                                # qc.draw(output='mpl').savefig(circuit_png_path)
                                
                                # Add description markdown
                                with open(os.path.join(temp_dir, "README.md"), "w") as f:
                                    f.write(f"# {circuit_type} Circuit\n\n")
                                    f.write(f"Number of qubits: {num_qubits}\n\n")
                                    f.write("## Circuit Description\n\n")
                                    if circuit_type == "Bell State":
                                        f.write("A Bell state is a maximally entangled quantum state of two qubits.\n")
                                    elif circuit_type == "GHZ State":
                                        f.write(f"A GHZ state is a maximally entangled state of {num_qubits} qubits.\n")
                                    elif circuit_type == "QFT":
                                        f.write(f"The Quantum Fourier Transform is a quantum implementation of the discrete Fourier transform over the amplitudes of a quantum state.\n")
                        
                        elif content_type == "File Upload":
                            if not uploaded_file:
                                st.error("Please upload a file")
                                return
                            
                            # Save uploaded file
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                        
                        elif content_type == "Text/JSON":
                            # Save JSON/text content
                            try:
                                # Check if it's valid JSON
                                content = json.loads(json_content)
                                with open(os.path.join(temp_dir, "content.json"), "w") as f:
                                    json.dump(content, f, indent=2)
                            except:
                                # If not valid JSON, save as text
                                with open(os.path.join(temp_dir, "content.txt"), "w") as f:
                                    f.write(json_content)
                        
                        # Create the artifact
                        artifact = wandb.Artifact(
                            name=artifact_name,
                            type=artifact_type,
                            description=artifact_description,
                            metadata=metadata
                        )
                        
                        # Add all files from the temp directory
                        artifact.add_dir(temp_dir)
                        
                        # Log the artifact
                        run.log_artifact(artifact)
                        run.finish()
                    
                    st.success(f"Successfully uploaded {artifact_name} to the registry")
                    
                    # Clear artifacts list to refresh
                    if "registry_artifacts" in st.session_state:
                        del st.session_state.registry_artifacts
                
                except Exception as e:
                    st.error(f"Error uploading to registry: {str(e)}")

def use_from_registry():
    """Use artifacts from the W&B Registry"""
    st.subheader("Use from Registry")
    
    org = st.session_state.registry_org
    
    # Check if we have artifacts
    if "registry_artifacts" not in st.session_state or not st.session_state.registry_artifacts:
        st.info("Please browse the registry first to see available artifacts")
        return
    
    # Filter for quantum circuits and models
    usable_artifacts = [
        a for a in st.session_state.registry_artifacts
        if a["type"] in ["quantum-circuit", "model", "algorithm"]
    ]
    
    if not usable_artifacts:
        st.info("No usable quantum circuits or models found in the registry")
        return
    
    # Select artifact to use
    artifact_options = [f"{a['name']}:{a['version']} ({a['type']})" for a in usable_artifacts]
    selected_artifact_option = st.selectbox(
        "Select Artifact to Use", 
        artifact_options,
        key=generate_unique_key("use_artifact_select")
    )
    
    if selected_artifact_option:
        # Find the selected artifact
        artifact_name, artifact_version = selected_artifact_option.split(":", 1)[0], selected_artifact_option.split(":", 1)[1].split(" ")[0]
        selected_artifact = next((a for a in usable_artifacts if a["name"] == artifact_name and a["version"] == artifact_version), None)
        
        if selected_artifact:
            st.subheader(f"Using {selected_artifact['name']} ({selected_artifact['type']})")
            
            # Download the artifact
            if st.button("Download and Use", key=generate_unique_key("download_button")):
                with st.spinner("Downloading artifact..."):
                    try:
                        api = wandb.Api()
                        artifact = api.artifact(f"{org}/{selected_artifact['name']}:{selected_artifact['version']}")
                        
                        # Create a temporary directory for artifact files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Download the artifact
                            artifact_dir = artifact.download(root=temp_dir)
                            
                            # Check what's in the artifact
                            files = os.listdir(artifact_dir)
                            
                            st.success(f"Downloaded {len(files)} files from artifact")
                            
                            # Display files
                            st.write("Files in artifact:")
                            for file in files:
                                st.write(f"- {file}")
                            
                            # If it's a quantum circuit, try to load and display it
                            if selected_artifact["type"] == "quantum-circuit":
                                try:
                                    # Look for QASM file
                                    qasm_files = [f for f in files if f.endswith(".qasm")]
                                    if qasm_files:
                                        qasm_file = os.path.join(artifact_dir, qasm_files[0])
                                        
                                        # Read QASM content
                                        with open(qasm_file, "r") as f:
                                            qasm_content = f.read()
                                        
                                        st.subheader("Quantum Circuit (QASM)")
                                        st.code(qasm_content, language="qasm")
                                        
                                        # Load circuit with Qiskit
                                        try:
                                            from qiskit import QuantumCircuit
                                            circuit = QuantumCircuit.from_qasm_str(qasm_content)
                                            
                                            # Display circuit properties
                                            st.write(f"Number of qubits: {circuit.num_qubits}")
                                            st.write(f"Number of classical bits: {circuit.num_clbits}")
                                            st.write(f"Gate operations: {circuit.size()}")
                                            
                                            # Option to run simulation
                                            if st.button("Run Simulation", key=generate_unique_key("run_sim")):
                                                with st.spinner("Running quantum simulation..."):
                                                    from qiskit import Aer, execute
                                                    
                                                    # Run simulation
                                                    simulator = Aer.get_backend('qasm_simulator')
                                                    job = execute(circuit, simulator, shots=1024)
                                                    result = job.result()
                                                    counts = result.get_counts(circuit)
                                                    
                                                    # Display results
                                                    st.subheader("Simulation Results")
                                                    
                                                    # Convert to sorted list of (state, count) tuples
                                                    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                                                    
                                                    # Create a bar chart
                                                    import matplotlib.pyplot as plt
                                                    import numpy as np
                                                    
                                                    states = [state for state, _ in sorted_counts]
                                                    probabilities = [count/1024 for _, count in sorted_counts]
                                                    
                                                    fig, ax = plt.subplots(figsize=(10, 6))
                                                    ax.bar(states, probabilities)
                                                    ax.set_xlabel('State')
                                                    ax.set_ylabel('Probability')
                                                    ax.set_title('Quantum State Probabilities')
                                                    st.pyplot(fig)
                                                    
                                                    # Show numerical results
                                                    results_data = [{"State": state, "Count": count, "Probability": count/1024} 
                                                                   for state, count in sorted_counts]
                                                    st.dataframe(pd.DataFrame(results_data))
                                        except Exception as e:
                                            st.error(f"Error running simulation: {str(e)}")
                                    
                                    # Look for Python files
                                    py_files = [f for f in files if f.endswith(".py")]
                                    if py_files:
                                        py_file = os.path.join(artifact_dir, py_files[0])
                                        
                                        # Read Python content
                                        with open(py_file, "r") as f:
                                            py_content = f.read()
                                        
                                        st.subheader("Python Circuit Code")
                                        st.code(py_content, language="python")
                                        
                                        # Option to execute Python code (could be dangerous)
                                        st.warning("Running external code can be risky. Review carefully before executing.")
                                except Exception as e:
                                    st.error(f"Error loading quantum circuit: {str(e)}")
                            
                            # For models, check for model files
                            elif selected_artifact["type"] == "model":
                                # Look for common model files
                                model_files = [f for f in files if f.endswith((".pkl", ".json", ".h5", ".pt", ".pth"))]
                                if model_files:
                                    st.subheader("Model Files")
                                    for model_file in model_files:
                                        st.write(f"- {model_file}")
                                    
                                    st.info("Models can be loaded and used in your quantum circuits or ML pipelines")
                                
                                # Look for README or documentation
                                readme_files = [f for f in files if f.lower() in ["readme.md", "readme.txt", "documentation.md"]]
                                if readme_files:
                                    readme_file = os.path.join(artifact_dir, readme_files[0])
                                    
                                    # Read README content
                                    with open(readme_file, "r") as f:
                                        readme_content = f.read()
                                    
                                    st.subheader("Model Documentation")
                                    st.markdown(readme_content)
                    
                    except Exception as e:
                        st.error(f"Error using artifact: {str(e)}")

def manage_registry():
    """Manage W&B Artifact Registry"""
    st.subheader("Manage Registry")
    
    org = st.session_state.registry_org
    
    # Check if admin
    if not st.session_state.get("is_admin", False):
        st.warning("Registry management requires admin permissions")
        
        # Option to verify admin status
        if st.button("Verify Admin Status", key=generate_unique_key("verify_admin")):
            with st.spinner("Verifying admin status..."):
                try:
                    api = wandb.Api()
                    entity = api.entity(org)
                    
                    # Check if current user is admin of the entity
                    # In a real implementation, this would check current user's role in the organization
                    st.session_state.is_admin = True
                    st.success("Admin status verified")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error verifying admin status: {str(e)}")
        
        return
    
    # Admin management interface
    st.success("Organization Admin: " + org)
    
    # Organization artifact collections
    st.subheader("Artifact Collections")
    
    try:
        api = wandb.Api()
        collections = api.artifact_types(entity=org)
        
        if collections:
            collection_data = []
            for collection in collections:
                collection_data.append({
                    "Name": collection.name,
                    "Count": collection.collection_size,
                    "Description": getattr(collection, "description", "")
                })
            
            st.dataframe(pd.DataFrame(collection_data), use_container_width=True)
        else:
            st.info("No artifact collections found")
    except Exception as e:
        st.error(f"Error fetching collections: {str(e)}")
    
    # Add a collection
    with st.expander("Add New Collection"):
        with st.form("add_collection_form"):
            collection_name = st.text_input("Collection Name", key=generate_unique_key("collection_name"))
            collection_description = st.text_area("Description", key=generate_unique_key("collection_description"))
            
            submit_button = st.form_submit_button("Create Collection")
            
            if submit_button:
                with st.spinner("Creating collection..."):
                    try:
                        # Initialize a new run to create the collection type
                        run = wandb.init(
                            project="registry-management",
                            entity=org,
                            job_type="collection-creation"
                        )
                        
                        # Create an artifact of the new type
                        artifact = wandb.Artifact(
                            name=f"{collection_name}-init",
                            type=collection_name,
                            description=collection_description,
                            metadata={"registry_managed": True}
                        )
                        
                        # Add a simple README
                        with tempfile.TemporaryDirectory() as temp_dir:
                            readme_path = os.path.join(temp_dir, "README.md")
                            with open(readme_path, "w") as f:
                                f.write(f"# {collection_name}\n\n")
                                f.write(f"{collection_description}\n\n")
                                f.write(f"Collection created: {time.strftime('%Y-%m-%d')}\n")
                            
                            artifact.add_file(readme_path, "README.md")
                            
                            # Log the artifact
                            run.log_artifact(artifact)
                            run.finish()
                        
                        st.success(f"Collection '{collection_name}' created successfully")
                    except Exception as e:
                        st.error(f"Error creating collection: {str(e)}")
    
    # Delete or modify artifacts
    with st.expander("Delete or Modify Artifacts"):
        # Only show if we have artifacts
        if "registry_artifacts" in st.session_state and st.session_state.registry_artifacts:
            artifacts = st.session_state.registry_artifacts
            
            # Select artifact to manage
            artifact_options = [f"{a['name']}:{a['version']} ({a['type']})" for a in artifacts]
            selected_artifact_option = st.selectbox(
                "Select Artifact", 
                artifact_options,
                key=generate_unique_key("manage_artifact_select")
            )
            
            if selected_artifact_option:
                # Find the selected artifact
                artifact_name, artifact_version = selected_artifact_option.split(":", 1)[0], selected_artifact_option.split(":", 1)[1].split(" ")[0]
                selected_artifact = next((a for a in artifacts if a["name"] == artifact_name and a["version"] == artifact_version), None)
                
                if selected_artifact:
                    st.write(f"Selected: {selected_artifact['name']} (v{selected_artifact['version']})")
                    
                    # Delete option
                    if st.button("Delete Artifact", key=generate_unique_key("delete_artifact")):
                        with st.spinner("Deleting artifact..."):
                            try:
                                api = wandb.Api()
                                artifact = api.artifact(f"{org}/{selected_artifact['name']}:{selected_artifact['version']}")
                                artifact.delete()
                                
                                st.success(f"Artifact {selected_artifact['name']}:{selected_artifact['version']} deleted")
                                
                                # Refresh artifact list
                                if "registry_artifacts" in st.session_state:
                                    del st.session_state.registry_artifacts
                                
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting artifact: {str(e)}")
                    
                    # Update description
                    with st.form("update_description_form"):
                        new_description = st.text_area(
                            "Update Description", 
                            value=selected_artifact.get("description", ""),
                            key=generate_unique_key("update_description")
                        )
                        
                        update_button = st.form_submit_button("Update Description")
                        
                        if update_button:
                            with st.spinner("Updating artifact..."):
                                try:
                                    api = wandb.Api()
                                    artifact = api.artifact(f"{org}/{selected_artifact['name']}:{selected_artifact['version']}")
                                    
                                    # Update description
                                    artifact.description = new_description
                                    artifact.save()
                                    
                                    st.success("Description updated successfully")
                                    
                                    # Refresh artifact list
                                    if "registry_artifacts" in st.session_state:
                                        del st.session_state.registry_artifacts
                                except Exception as e:
                                    st.error(f"Error updating artifact: {str(e)}")
        else:
            st.info("Browse the registry first to see available artifacts")

def display_artifact_details(artifact):
    """Display detailed information about an artifact"""
    st.subheader(f"Artifact Details: {artifact['name']}")
    
    # Basic information
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Type:** {artifact['type']}")
        st.write(f"**Version:** {artifact['version']}")
        created_at = pd.to_datetime(artifact["created_at"]).strftime('%Y-%m-%d %H:%M:%S') if artifact["created_at"] else "Unknown"
        st.write(f"**Created:** {created_at}")
    
    with col2:
        st.write(f"**ID:** {artifact['id']}")
        st.write(f"**Size:** {format_size(artifact['size']) if artifact['size'] else 'Unknown'}")
        updated_at = pd.to_datetime(artifact["updated_at"]).strftime('%Y-%m-%d %H:%M:%S') if artifact["updated_at"] else "Unknown"
        st.write(f"**Updated:** {updated_at}")
    
    # Description
    if artifact.get("description"):
        st.subheader("Description")
        st.write(artifact["description"])
    
    # Metadata
    if artifact.get("metadata"):
        st.subheader("Metadata")
        try:
            st.json(artifact["metadata"])
        except:
            st.write(str(artifact["metadata"]))
    
    # View in W&B button
    org = st.session_state.registry_org
    artifact_url = f"https://wandb.ai/{org}/artifacts/{artifact['type']}/{artifact['name']}/v{artifact['version']}"
    st.markdown(f"[View in W&B]({artifact_url})")
    
    # Download button
    if st.button("Download Artifact", key=generate_unique_key("download_artifact")):
        with st.spinner("Downloading artifact..."):
            try:
                api = wandb.Api()
                artifact_obj = api.artifact(f"{org}/{artifact['name']}:{artifact['version']}")
                
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Download artifact
                    artifact_dir = artifact_obj.download(root=temp_dir)
                    
                    # Check what's in the artifact
                    files = os.listdir(artifact_dir)
                    
                    st.success(f"Downloaded {len(files)} files from artifact")
                    
                    # List files
                    with st.expander("Files in Artifact"):
                        for file in files:
                            file_path = os.path.join(artifact_dir, file)
                            file_size = os.path.getsize(file_path)
                            st.write(f"- {file} ({format_size(file_size)})")
                            
                            # Preview text files
                            if file.endswith(('.txt', '.md', '.py', '.json', '.qasm')) and file_size < 1000000:  # < 1MB
                                try:
                                    with open(file_path, 'r') as f:
                                        content = f.read()
                                    
                                    with st.expander(f"Preview: {file}"):
                                        if file.endswith('.md'):
                                            st.markdown(content)
                                        elif file.endswith('.json'):
                                            try:
                                                st.json(json.loads(content))
                                            except:
                                                st.code(content)
                                        else:
                                            st.code(content)
                                except Exception as e:
                                    st.error(f"Error previewing file: {str(e)}")
            except Exception as e:
                st.error(f"Error downloading artifact: {str(e)}")

def format_size(size_bytes):
    """Format bytes to human-readable size"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"