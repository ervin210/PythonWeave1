import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService
import os
import json
import io
import base64
from datetime import datetime

def ibm_quantum_integration():
    """
    IBM Quantum Integration Component - Connect to IBM Quantum backends
    """
    st.title("IBM Quantum Integration")
    st.markdown("""
    Connect to IBM Quantum backends to run your quantum circuits on real quantum hardware or premium simulators.
    This enterprise feature enables access to IBM's quantum computing resources.
    """)
    
    # Check for IBM Quantum token
    has_token = check_ibm_token()
    
    if not has_token:
        st.warning("No IBM Quantum API token found. Please enter your token to access IBM Quantum backends.")
        with st.form("ibm_quantum_token_form"):
            token = st.text_input("IBM Quantum API Token", type="password")
            save_token = st.checkbox("Save token for future sessions", value=True)
            submitted = st.form_submit_button("Connect to IBM Quantum")
            
            if submitted and token:
                save_ibm_token(token, save_token)
                st.success("IBM Quantum token saved! You can now access IBM Quantum backends.")
                st.rerun()
    else:
        # Show available backends
        st.subheader("Available IBM Quantum Backends")
        
        with st.spinner("Connecting to IBM Quantum..."):
            try:
                backends = get_ibm_backends()
                if backends:
                    # Convert to DataFrame for display
                    backend_data = []
                    for backend in backends:
                        backend_data.append({
                            "Name": backend.name,
                            "Status": "Online" if backend.status().operational else "Offline",
                            "Pending Jobs": backend.status().pending_jobs,
                            "Type": "Simulator" if backend.configuration().simulator else "QPU",
                            "Qubits": backend.configuration().n_qubits
                        })
                    
                    backends_df = pd.DataFrame(backend_data)
                    st.dataframe(backends_df, use_container_width=True)
                    
                    # Circuit creation and execution
                    circuit_tab, job_tab = st.tabs(["Create Circuit", "Manage Jobs"])
                    
                    with circuit_tab:
                        create_and_run_circuit()
                    
                    with job_tab:
                        manage_jobs()
                else:
                    st.error("No backends available. Please check your IBM Quantum token.")
            except Exception as e:
                st.error(f"Error connecting to IBM Quantum: {str(e)}")
                st.info("This might be due to an invalid token or network issues. Please try again.")
                
                # Option to reset token
                if st.button("Reset IBM Quantum Token"):
                    reset_ibm_token()
                    st.success("Token reset. Please enter a new token.")
                    st.rerun()

def check_ibm_token():
    """Check if IBM Quantum token exists and display datacenter/instance information"""
    # Check session state first
    if "ibm_quantum_token" in st.session_state and st.session_state.ibm_quantum_token:
        try:
            # If we're checking the token, try to get available instances/datacenters
            # Only do this if we don't already have the information to avoid repeated API calls
            if "ibm_quantum_instances" not in st.session_state:
                # Initialize the service
                service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
                
                # Get instances/regions
                instances = service.instances()
                
                # Store in session state
                st.session_state.ibm_quantum_instances = instances
                
                # Get total backends count
                backends = get_ibm_backends()
                if backends:
                    instance_counts = {}
                    simulator_count = 0
                    qpu_count = 0
                    
                    # Count backends by type and datacenter
                    for backend in backends:
                        datacenter = getattr(backend, "_instance", "Default")
                        if datacenter not in instance_counts:
                            instance_counts[datacenter] = {"simulators": 0, "qpus": 0}
                        
                        if backend.configuration().simulator:
                            instance_counts[datacenter]["simulators"] += 1
                            simulator_count += 1
                        else:
                            instance_counts[datacenter]["qpus"] += 1
                            qpu_count += 1
                    
                    # Display instance information
                    st.success(f"✓ IBM Quantum access verified: {len(instances)} datacenters available")
                    st.info(f"Connected to IBM Quantum with access to {qpu_count} QPUs and {simulator_count} simulators")
                    
                    # Show datacenter details in an expandable section
                    with st.expander("View Datacenter Details"):
                        for datacenter, counts in instance_counts.items():
                            st.write(f"**Datacenter: {datacenter}**")
                            st.write(f"- QPUs: {counts['qpus']}")
                            st.write(f"- Simulators: {counts['simulators']}")
            
            return True
        except Exception as e:
            st.error(f"Error connecting to IBM Quantum: {str(e)}")
            # Clear the token if it's invalid
            if "Invalid token" in str(e):
                st.session_state.pop("ibm_quantum_token", None)
            return False
    
    # Check secure storage
    try:
        # Check if file exists
        if os.path.exists("secure_assets/ibm_config.json"):
            with open("secure_assets/ibm_config.json", "r") as f:
                config = json.load(f)
                if "token" in config and config["token"]:
                    # Load token to session state
                    st.session_state.ibm_quantum_token = config["token"]
                    return check_ibm_token()  # Recursive call with the loaded token
    except Exception:
        pass
    
    return False

def save_ibm_token(token, permanent=True):
    """Save IBM Quantum token"""
    # Always save to session state
    st.session_state.ibm_quantum_token = token
    
    # Save to secure storage if requested
    if permanent:
        os.makedirs("secure_assets", exist_ok=True)
        config = {"token": token, "saved_at": datetime.now().isoformat()}
        
        with open("secure_assets/ibm_config.json", "w") as f:
            json.dump(config, f)

def reset_ibm_token():
    """Reset IBM Quantum token"""
    # Clear session state
    if "ibm_quantum_token" in st.session_state:
        del st.session_state.ibm_quantum_token
    
    # Remove from secure storage
    if os.path.exists("secure_assets/ibm_config.json"):
        os.remove("secure_assets/ibm_config.json")

def get_ibm_backends():
    """Get available IBM Quantum backends"""
    if "ibm_quantum_token" not in st.session_state:
        return []
    
    # Get available instances/regions
    available_instances = []
    
    try:
        # Initialize service with default instance first
        service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
        
        # Get list of available instances/regions
        instances = service.instances()
        
        # Store the instance information in session state if not already present
        if "ibm_quantum_instances" not in st.session_state:
            st.session_state.ibm_quantum_instances = instances
        
        # If instances found, get backends from all instances
        all_backends = []
        
        # Loop through all instances/regions to get backends from each datacenter
        for instance in instances:
            try:
                instance_service = QiskitRuntimeService(
                    channel="ibm_quantum", 
                    token=st.session_state.ibm_quantum_token,
                    instance=instance
                )
                
                # Get backends for this instance
                instance_backends = instance_service.backends()
                
                # Add instance information to each backend for display
                for backend in instance_backends:
                    backend._instance = instance  # Add instance info
                    all_backends.append(backend)
                    
            except Exception as e:
                # If we can't connect to a specific instance, just continue
                print(f"Could not connect to instance {instance}: {str(e)}")
                continue
        
        return all_backends
        
    except Exception as e:
        # Fall back to default instance if there's an error getting all instances
        print(f"Error getting instances: {str(e)}, falling back to default")
        service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
        return service.backends()

def create_and_run_circuit():
    """Create and run a quantum circuit on IBM Quantum backends"""
    st.subheader("Create and Run Quantum Circuit")
    
    # Circuit type selection
    circuit_type = st.selectbox(
        "Select Circuit Type",
        ["Bell State", "GHZ State", "Quantum Fourier Transform", "Custom Circuit"]
    )
    
    # Number of qubits selection
    max_qubits = 5  # Reasonable limit for demonstration
    num_qubits = st.slider("Number of Qubits", 2, max_qubits, 2)
    
    # Create circuit based on selection
    if circuit_type == "Bell State":
        circuit = create_bell_state()
        st.info("Created a Bell state circuit (maximally entangled state)")
    elif circuit_type == "GHZ State":
        circuit = create_ghz_state(num_qubits)
        st.info(f"Created a {num_qubits}-qubit GHZ state circuit")
    elif circuit_type == "Quantum Fourier Transform":
        circuit = create_qft_circuit(num_qubits)
        st.info(f"Created a {num_qubits}-qubit Quantum Fourier Transform circuit")
    else:  # Custom Circuit
        circuit = create_custom_circuit(num_qubits)
    
    # Display circuit
    circuit_fig = circuit.draw(output="mpl")
    st.pyplot(circuit_fig)
    
    # Backend selection
    try:
        # Get service instance
        service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
        
        # Get all available backends from all datacenters
        backends = get_ibm_backends()
        
        # Get all instances/datacenters
        if "ibm_quantum_instances" in st.session_state:
            instances = st.session_state.ibm_quantum_instances
            
            # Create a dropdown to filter by datacenter
            datacenter_options = ["All Datacenters"] + list(instances)
            selected_datacenter = st.selectbox(
                "Filter by Datacenter/Region", 
                options=datacenter_options,
                help="Select a specific datacenter or region to filter available quantum systems"
            )
            
            # Filter backends by selected datacenter if not "All Datacenters"
            if selected_datacenter != "All Datacenters":
                backends = [b for b in backends if hasattr(b, "_instance") and b._instance == selected_datacenter]
        
        # Filter backends that can handle this circuit
        compatible_backends = [backend for backend in backends if backend.configuration().n_qubits >= num_qubits]
        
        if compatible_backends:
            # Format backend names to include datacenter info
            backend_options = []
            for backend in compatible_backends:
                datacenter = getattr(backend, "_instance", "Default")
                n_qubits = backend.configuration().n_qubits
                is_simulator = backend.configuration().simulator
                
                # Create formatted display name
                backend_type = "Simulator" if is_simulator else "QPU"
                display_name = f"{backend.name} ({datacenter}, {n_qubits} qubits, {backend_type})"
                backend_options.append((backend.name, display_name))
            
            # Use the formatted display names for the selectbox
            selected_display = st.selectbox(
                "Select Backend", 
                options=[display for _, display in backend_options],
                help="Choose a quantum system to run your circuit"
            )
            
            # Map back to the actual backend name
            selected_backend = next(name for name, display in backend_options if display == selected_display)
            
            # Number of shots
            shots = st.slider("Number of Shots", 1, 10000, 1024)
            
            # Option to save job
            save_job = st.checkbox("Save Job Results", value=True)
            
            # Execute button
            if st.button("Run on IBM Quantum"):
                with st.spinner(f"Running circuit on {selected_backend}..."):
                    try:
                        # Get the selected backend from our compatible_backends list to know its instance/datacenter
                        selected_backend_obj = next((b for b in compatible_backends if b.name == selected_backend), None)
                        
                        # Make sure we found the backend object
                        if not selected_backend_obj:
                            raise ValueError(f"Could not find backend '{selected_backend}' in the list of compatible backends")
                        
                        # Determine if simulator or real hardware
                        is_simulator = selected_backend_obj.configuration().simulator
                        
                        # Get the instance/datacenter this backend belongs to
                        instance = getattr(selected_backend_obj, "_instance", None)
                        
                        # Initialize service with the correct instance if available
                        if instance:
                            service = QiskitRuntimeService(
                                channel="ibm_quantum", 
                                token=st.session_state.ibm_quantum_token,
                                instance=instance
                            )
                        else:
                            service = QiskitRuntimeService(
                                channel="ibm_quantum", 
                                token=st.session_state.ibm_quantum_token
                            )
                        
                        # Get the backend instance from the correct service
                        backend_instance = service.backend(selected_backend)
                        
                        # Use the Sampler primitive for all backends
                        from qiskit_ibm_runtime import Sampler, Options
                        
                        # Set options (shots, optimization level, etc)
                        options = Options()
                        options.execution.shots = shots
                        options.optimization_level = 1
                        
                        # Create the Sampler instance
                        sampler = Sampler(session=service, backend=selected_backend, options=options)
                        
                        # Submit the circuit
                        job = sampler.run(circuits=[circuit])
                        job_id = job.job_id()
                        st.success(f"Job submitted to {selected_backend}! Job ID: {job_id}")
                        
                        if is_simulator:
                            # For simulators, wait for completion (usually quick)
                            st.info("Waiting for job to complete...")
                            result = job.result()
                            
                            # Convert quasi_dists to traditional counts format
                            quasi_dist = result.quasi_dists[0]
                            counts = {}
                            for bitstring, probability in quasi_dist.items():
                                # Convert integer to binary string and format it
                                n_bits = circuit.num_clbits
                                binary = format(bitstring, f'0{n_bits}b')
                                counts[binary] = int(probability * shots)
                            
                            # Display results
                            fig = plot_histogram(counts)
                            st.pyplot(fig)
                            
                            # Create result object compatible with save_job_results
                            class CustomResult:
                                def __init__(self, counts):
                                    self._counts = counts
                                
                                def get_counts(self, _=None):
                                    return self._counts
                            
                            custom_result = CustomResult(counts)
                            
                            # Save results if requested
                            if save_job:
                                save_job_results(job_id, selected_backend, circuit, custom_result)
                        else:
                            # For real QPUs, provide status updates
                            st.info("The job has been submitted to the queue. Check the 'Manage Jobs' tab for status and results.")
                            
                            # Save job info for tracking
                            if "ibm_quantum_jobs" not in st.session_state:
                                st.session_state.ibm_quantum_jobs = []
                            
                            st.session_state.ibm_quantum_jobs.append({
                                "job_id": job.job_id(),
                                "backend": selected_backend,
                                "status": "QUEUED",
                                "shots": shots,
                                "qubits": num_qubits,
                                "circuit_type": circuit_type,
                                "created_at": datetime.now().isoformat()
                            })
                    except Exception as e:
                        st.error(f"Error running circuit: {str(e)}")
        else:
            st.warning(f"No compatible backends found for a {num_qubits}-qubit circuit.")
    except Exception as e:
        st.error(f"Error connecting to IBM Quantum: {str(e)}")

def create_bell_state():
    """Create a Bell state quantum circuit"""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit

def create_ghz_state(num_qubits):
    """Create a GHZ state quantum circuit"""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    circuit.measure_all()
    return circuit

def create_qft_circuit(num_qubits):
    """Create a Quantum Fourier Transform circuit"""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    # Initialize with non-trivial state
    for i in range(num_qubits):
        circuit.h(i)
    
    # Apply QFT
    for i in range(num_qubits):
        for j in range(i):
            circuit.cp(2*np.pi/2**(i-j), j, i)
        circuit.h(i)
    
    # Add measurements
    circuit.measure_all()
    
    return circuit

def create_custom_circuit(num_qubits):
    """Create a custom quantum circuit"""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    st.subheader("Custom Circuit Builder")
    
    # Add basic gates
    st.write("Add gates to your circuit:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Single qubit gates
        st.write("Single-qubit gates:")
        
        for i in range(num_qubits):
            gate_cols = st.columns(4)
            
            if gate_cols[0].button(f"H q{i}"):
                circuit.h(i)
            
            if gate_cols[1].button(f"X q{i}"):
                circuit.x(i)
            
            if gate_cols[2].button(f"Z q{i}"):
                circuit.z(i)
            
            if gate_cols[3].button(f"T q{i}"):
                circuit.t(i)
    
    with col2:
        # Two qubit gates
        st.write("Two-qubit gates:")
        
        for i in range(num_qubits-1):
            for j in range(i+1, num_qubits):
                gate_cols = st.columns(2)
                
                if gate_cols[0].button(f"CNOT q{i}→q{j}"):
                    circuit.cx(i, j)
                
                if gate_cols[1].button(f"CNOT q{j}→q{i}"):
                    circuit.cx(j, i)
    
    # Add measurement option
    if st.checkbox("Add measurements", value=True):
        circuit.measure_all()
    
    # Display current circuit while building
    st.write("Current circuit:")
    current_fig = circuit.draw(output="mpl")
    st.pyplot(current_fig)
    
    return circuit

def save_job_results(job_id, backend_name, circuit, result):
    """Save job results for later reference"""
    if "ibm_quantum_results" not in st.session_state:
        st.session_state.ibm_quantum_results = {}
    
    # Convert circuit to image for storage
    circuit_fig = circuit.draw(output="mpl")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    circuit_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Get counts
    counts = result.get_counts(circuit)
    
    # Create histogram for storage
    hist_fig = plot_histogram(counts)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    hist_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Save the data
    st.session_state.ibm_quantum_results[job_id] = {
        "job_id": job_id,
        "backend": backend_name,
        "counts": counts,
        "circuit_img": circuit_img,
        "histogram_img": hist_img,
        "completed_at": datetime.now().isoformat()
    }

def manage_jobs():
    """Manage and check IBM Quantum jobs"""
    st.subheader("Manage IBM Quantum Jobs")
    
    # Check if we have any saved jobs
    if "ibm_quantum_jobs" not in st.session_state or not st.session_state.ibm_quantum_jobs:
        st.info("No jobs found. Run a circuit to see jobs here.")
        return
    
    # Refresh job statuses
    if st.button("Refresh Job Status"):
        update_job_statuses()
    
    # Display jobs
    job_data = []
    for job in st.session_state.ibm_quantum_jobs:
        job_data.append({
            "Job ID": job["job_id"],
            "Backend": job["backend"],
            "Circuit": job.get("circuit_type", "Unknown"),
            "Status": job["status"],
            "Submitted": job["created_at"]
        })
    
    jobs_df = pd.DataFrame(job_data)
    st.dataframe(jobs_df, use_container_width=True)
    
    # View job results
    st.subheader("View Job Results")
    
    # Check if we have any completed jobs with saved results
    if "ibm_quantum_results" in st.session_state and st.session_state.ibm_quantum_results:
        completed_jobs = list(st.session_state.ibm_quantum_results.keys())
        
        if completed_jobs:
            selected_job = st.selectbox("Select a job to view results", completed_jobs)
            
            if selected_job in st.session_state.ibm_quantum_results:
                job_result = st.session_state.ibm_quantum_results[selected_job]
                
                st.write(f"**Backend:** {job_result['backend']}")
                st.write(f"**Completed:** {job_result['completed_at']}")
                
                # Display circuit
                st.write("**Circuit:**")
                st.image(f"data:image/png;base64,{job_result['circuit_img']}")
                
                # Display histogram
                st.write("**Results:**")
                st.image(f"data:image/png;base64,{job_result['histogram_img']}")
                
                # Display counts
                st.write("**Counts:**")
                st.json(job_result['counts'])
        else:
            st.info("No completed jobs with saved results found.")
    else:
        st.info("No completed jobs with saved results found.")

def update_job_statuses():
    """Update the status of all saved jobs"""
    if "ibm_quantum_jobs" not in st.session_state or not st.session_state.ibm_quantum_jobs:
        return
    
    try:
        # Create a default service
        service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
        
        # Create a dictionary to hold services for each instance/datacenter
        # This avoids recreating the service for each job in the same datacenter
        instance_services = {}
        
        with st.spinner("Updating job statuses..."):
            for i, job_info in enumerate(st.session_state.ibm_quantum_jobs):
                job_id = job_info["job_id"]
                
                try:
                    # Try to get job from default service first
                    job = None
                    status = "UNKNOWN"
                    try:
                        job = service.job(job_id)
                        status = job.status().value
                    except Exception as e:
                        # If that fails, the job might be in a different datacenter
                        # Try each available instance/datacenter
                        if "ibm_quantum_instances" in st.session_state:
                            job_found = False
                            for instance in st.session_state.ibm_quantum_instances:
                                # Reuse existing service for this instance if we already created one
                                if instance in instance_services:
                                    instance_service = instance_services[instance]
                                else:
                                    # Create a new service for this instance
                                    instance_service = QiskitRuntimeService(
                                        channel="ibm_quantum", 
                                        token=st.session_state.ibm_quantum_token,
                                        instance=instance
                                    )
                                    # Cache it for future use
                                    instance_services[instance] = instance_service
                                
                                try:
                                    # Try to get the job from this instance
                                    job = instance_service.job(job_id)
                                    status = job.status().value
                                    job_found = True
                                    break
                                except Exception:
                                    # Job not found in this instance, continue to next one
                                    continue
                            
                            if not job_found:
                                # If we tried all instances and didn't find the job
                                raise ValueError(f"Job {job_id} not found in any datacenter")
                        else:
                            # No instances available to try
                            raise e
                    
                    # Make sure we have a job object
                    if not job:
                        raise ValueError(f"Could not find job {job_id} in any datacenter")
                    
                    # Update status
                    st.session_state.ibm_quantum_jobs[i]["status"] = status
                    
                    # If job is done and we don't have results yet, save them
                    if status == "DONE" and job_id not in st.session_state.get("ibm_quantum_results", {}):
                        try:
                            # Initialize results dict if needed
                            if "ibm_quantum_results" not in st.session_state:
                                st.session_state.ibm_quantum_results = {}
                            
                            # Get the result using the Sampler primitive
                            result = job.result()
                            
                            # Create a circuit for display
                            if job_info.get("circuit_type") == "Bell State":
                                circuit = create_bell_state()
                            elif job_info.get("circuit_type") == "GHZ State":
                                circuit = create_ghz_state(job_info.get("qubits", 2))
                            elif job_info.get("circuit_type") == "Quantum Fourier Transform":
                                circuit = create_qft_circuit(job_info.get("qubits", 2))
                            else:
                                # Basic circuit if we don't know the type
                                circuit = QuantumCircuit(job_info.get("qubits", 2))
                                circuit.h(0)
                                circuit.measure_all()
                            
                            # Convert primitive result to traditional counts format
                            quasi_dist = result.quasi_dists[0]
                            shots = job_info.get("shots", 1024)
                            counts = {}
                            for bitstring, probability in quasi_dist.items():
                                # Convert integer to binary string
                                n_bits = circuit.num_clbits
                                binary = format(bitstring, f'0{n_bits}b')
                                counts[binary] = int(probability * shots)
                            
                            # Create result object compatible with save_job_results
                            class CustomResult:
                                def __init__(self, counts):
                                    self._counts = counts
                                
                                def get_counts(self, _=None):
                                    return self._counts
                            
                            custom_result = CustomResult(counts)
                            
                            # Save results
                            save_job_results(job_id, job_info["backend"], circuit, custom_result)
                        except Exception as e:
                            st.warning(f"Could not retrieve results for job {job_id}: {str(e)}")
                except Exception as e:
                    st.warning(f"Could not update status for job {job_id}: {str(e)}")
            
            st.success("Job statuses updated!")
    except Exception as e:
        st.error(f"Error updating job statuses: {str(e)}")