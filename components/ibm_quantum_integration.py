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
    """Check if IBM Quantum token exists"""
    # Check session state first
    if "ibm_quantum_token" in st.session_state and st.session_state.ibm_quantum_token:
        return True
    
    # Check secure storage
    try:
        # Check if file exists
        if os.path.exists("secure_assets/ibm_config.json"):
            with open("secure_assets/ibm_config.json", "r") as f:
                config = json.load(f)
                if "token" in config and config["token"]:
                    # Load token to session state
                    st.session_state.ibm_quantum_token = config["token"]
                    return True
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
    
    # Initialize service (using new QiskitRuntimeService instead of deprecated IBMQ)
    service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
    
    # Get available backends
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
        service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
        backends = service.backends()
        
        # Filter backends that can handle this circuit
        compatible_backends = [backend for backend in backends if backend.configuration().n_qubits >= num_qubits]
        
        if compatible_backends:
            backend_names = [backend.name for backend in compatible_backends]
            selected_backend = st.selectbox("Select Backend", backend_names)
            
            # Number of shots
            shots = st.slider("Number of Shots", 1, 10000, 1024)
            
            # Option to save job
            save_job = st.checkbox("Save Job Results", value=True)
            
            # Execute button
            if st.button("Run on IBM Quantum"):
                with st.spinner(f"Running circuit on {selected_backend}..."):
                    try:
                        # Get the backend instance
                        backend_instance = service.backend(selected_backend)
                        
                        # Determine if simulator or real hardware
                        is_simulator = backend_instance.configuration().simulator
                        
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
        service = QiskitRuntimeService(channel="ibm_quantum", token=st.session_state.ibm_quantum_token)
        
        with st.spinner("Updating job statuses..."):
            for i, job_info in enumerate(st.session_state.ibm_quantum_jobs):
                job_id = job_info["job_id"]
                
                try:
                    # Get job status
                    job = service.job(job_id)
                    status = job.status().value
                    
                    # Update status
                    st.session_state.ibm_quantum_jobs[i]["status"] = status
                    
                    # If job is done and we don't have results yet, save them
                    if status == "DONE" and job_id not in st.session_state.get("ibm_quantum_results", {}):
                        try:
                            # Initialize results dict if needed
                            if "ibm_quantum_results" not in st.session_state:
                                st.session_state.ibm_quantum_results = {}
                            
                            # Get the result
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
                            
                            # Save results
                            save_job_results(job_id, job_info["backend"], circuit, result)
                        except Exception as e:
                            st.warning(f"Could not retrieve results for job {job_id}: {str(e)}")
                except Exception as e:
                    st.warning(f"Could not update status for job {job_id}: {str(e)}")
            
            st.success("Job statuses updated!")
    except Exception as e:
        st.error(f"Error updating job statuses: {str(e)}")