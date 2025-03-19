import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
import pennylane as qml
import wandb
import json
import io
import base64
from PIL import Image

def quantum_assistant():
    """
    Quantum Computing AI Assistant for W&B experiment analysis
    """
    st.subheader("🧠 Quantum AI Assistant")
    
    # Introduction section
    st.markdown("""
    Welcome to the Quantum AI Assistant for your Weights & Biases experiments. 
    This assistant helps you analyze your machine learning experiments using quantum computing techniques 
    and provides AI-powered insights.
    """)
    
    # Assistant tabs
    assistant_tab, quantum_tab, insights_tab = st.tabs(["Assistant", "Quantum Computing", "AI Insights"])
    
    with assistant_tab:
        render_assistant_interface()
        
    with quantum_tab:
        render_quantum_tools()
        
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
    
    # Simple keyword matching for demo purposes
    # In a real system, you would use NLP/LLM here
    prompt_lower = prompt.lower()
    
    if "compare" in prompt_lower and "runs" in prompt_lower:
        return f"I can help you compare runs {project_info}. Use the Compare Runs feature in the Runs page to select multiple runs and analyze their performance on different metrics."
    
    elif "best" in prompt_lower and ("model" in prompt_lower or "run" in prompt_lower):
        return f"To find the best model {project_info}, I recommend looking at the Sweeps page where you can see which hyperparameters led to the best performance metrics."
    
    elif "quantum" in prompt_lower:
        return "I can help you apply quantum computing techniques to your ML experiments. Check out the Quantum Computing tab for quantum circuit simulation and integration with your experiment data."
    
    elif "optimize" in prompt_lower or "hyperparameter" in prompt_lower:
        return "For hyperparameter optimization, I recommend using W&B Sweeps which provides Bayesian, grid and random search methods. You can view your sweep results in the Sweeps page."
    
    elif "explain" in prompt_lower or "why" in prompt_lower:
        return f"I can help explain trends in your experimental results {project_info}. Look at the metrics visualizations to understand how your model's performance changed over time."
    
    elif "export" in prompt_lower or "download" in prompt_lower:
        return "You can export your experiment data using the Export buttons on various pages. This allows you to download metrics, parameters, and results as CSV files for further analysis."
    
    else:
        return f"I'm your quantum AI assistant for W&B experiments {project_info}. I can help you analyze runs, compare models, optimize hyperparameters, and apply quantum computing techniques to your ML workflows. Just ask me specific questions about your experiments."

def render_quantum_tools():
    """
    Renders quantum computing tools for ML experiment analysis
    """
    st.subheader("Quantum Computing Tools")
    
    st.markdown("""
    Apply quantum computing techniques to understand and enhance your machine learning experiments.
    """)
    
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

def render_ai_insights():
    """
    Renders AI-powered insights for W&B experiments
    """
    st.subheader("AI Insights for Your Experiments")
    
    if not st.session_state.authenticated:
        st.warning("Please log in with your W&B API key to access AI insights for your experiments.")
        return
    
    st.markdown("""
    Get AI-powered insights about your machine learning experiments.
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
    circuit_drawing = circuit.draw(output='mpl')
    buf = io.BytesIO()
    circuit_drawing.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def run_quantum_circuit(circuit):
    """Run a quantum circuit simulation"""
    simulator = Aer.get_backend('qasm_simulator')
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