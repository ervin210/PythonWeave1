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
import base64
from io import BytesIO
import time

def quantum_assistant():
    """
    Quantum Computing AI Assistant for W&B experiment analysis
    """
    st.header("Quantum AI Assistant")
    
    # Create tabs for different assistant features
    tab_chat, tab_quantum_tools, tab_insights = st.tabs(["Assistant Chat", "Quantum Tools", "AI Insights"])
    
    with tab_chat:
        render_assistant_interface()
    
    with tab_quantum_tools:
        render_quantum_tools()
    
    with tab_insights:
        render_ai_insights()


def render_assistant_interface():
    """
    Renders the conversational assistant interface
    """
    st.subheader("Quantum AI Experiment Assistant")
    st.markdown("""
    Ask questions about your experiments, quantum computing concepts, or get help with data analysis.
    The assistant can help you analyze your W&B runs, explain quantum computing principles, and provide
    AI-powered insights.
    """)
    
    # Initialize chat history in session state if it doesn't exist
    if "quantum_chat_history" not in st.session_state:
        st.session_state.quantum_chat_history = []
    
    # Display chat history
    for message in st.session_state.quantum_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input field for user queries
    user_prompt = st.chat_input("Ask about your experiments or quantum computing...")
    
    if user_prompt:
        # Add user message to chat history
        st.session_state.quantum_chat_history.append({"role": "user", "content": user_prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_assistant_response(user_prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.quantum_chat_history.append({"role": "assistant", "content": response})


def generate_assistant_response(prompt):
    """
    Generate a response to the user's query based on their W&B data
    
    Args:
        prompt: User's query text
        
    Returns:
        Generated response text
    """
    # Check if we're authenticated with W&B
    if not st.session_state.get("authenticated", False):
        return "Please authenticate with Weights & Biases first to access your experiment data."
    
    # Process the prompt to determine the type of query
    prompt_lower = prompt.lower()
    
    # Check if the query is about quantum computing
    if any(term in prompt_lower for term in ["quantum", "qubit", "circuit", "superposition", "entanglement"]):
        if "bell state" in prompt_lower or "entanglement" in prompt_lower:
            return """
            Bell states are the simplest examples of quantum entanglement. When two qubits are in a Bell state, 
            they are said to be maximally entangled. This means that measuring one qubit immediately affects 
            the state of the other, regardless of the distance between them.
            
            Here's how you can create a Bell state in Qiskit:
            ```python
            from qiskit import QuantumCircuit
            
            # Create a quantum circuit with 2 qubits
            qc = QuantumCircuit(2)
            
            # Apply Hadamard gate to the first qubit
            qc.h(0)
            
            # Apply CNOT gate with control=first qubit and target=second qubit
            qc.cx(0, 1)
            
            # Measure both qubits
            qc.measure_all()
            ```
            
            You can integrate this with W&B to track your quantum experiments and metrics.
            """
        
        elif "qft" in prompt_lower or "fourier" in prompt_lower:
            return """
            The Quantum Fourier Transform (QFT) is a linear transformation on quantum bits, and is the quantum
            analogue of the discrete Fourier transform. It's a key component in many quantum algorithms, 
            including Shor's factoring algorithm.
            
            The QFT transforms the quantum state:
            |x⟩ → (1/√N) ∑(y=0 to N-1) e^(2πixy/N) |y⟩
            
            Where N = 2^n for an n-qubit system.
            
            In Qiskit, you can implement QFT like this:
            ```python
            def create_qft_circuit(n_qubits):
                qc = QuantumCircuit(n_qubits)
                
                # Apply QFT
                for i in range(n_qubits):
                    qc.h(i)
                    for j in range(i+1, n_qubits):
                        qc.cp(2*np.pi/2**(j-i+1), j, i)
                
                # Swap qubits if needed
                for i in range(n_qubits//2):
                    qc.swap(i, n_qubits-i-1)
                    
                return qc
            ```
            
            You can track the results of QFT experiments in W&B to analyze the performance and properties of quantum algorithms.
            """
        
        else:
            return """
            Quantum computing uses quantum mechanical properties like superposition and entanglement to perform computations. 
            Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously, leading to potential 
            computational speedups for specific problems.
            
            In your W&B experiments, you can track quantum metrics like:
            - Fidelity: Measuring how close your quantum state is to the ideal state
            - Circuit depth: The number of layers in your quantum circuit
            - Qubit count: The number of qubits used in your experiment
            - Error rates: Various measures of quantum errors in your computation
            
            You can use libraries like Qiskit and PennyLane to run quantum simulations and experiments, 
            then log the results to W&B for analysis and visualization.
            
            Would you like to see some examples of quantum circuits or learn more about specific quantum algorithms?
            """
    
    # Check if the query is about W&B experiments
    elif any(term in prompt_lower for term in ["experiment", "run", "sweep", "metric", "compare", "wandb", "w&b"]):
        if "best run" in prompt_lower or "top performing" in prompt_lower:
            if not st.session_state.get("selected_project", None):
                return "Please select a project first to analyze the best runs."
            
            return f"""
            Based on your project '{st.session_state.selected_project.get('id', 'Unknown')}', I can help you find the best performing runs.
            
            To identify the best runs, you should:
            1. Define your key performance metrics (accuracy, loss, etc.)
            2. Consider hyperparameters that might influence performance
            3. Look for patterns across multiple runs
            
            You can use the Runs page to sort by specific metrics and compare the top performers.
            For a more in-depth analysis, you can select multiple runs and use the comparison feature
            to visualize differences in metrics and parameters.
            
            Would you like me to help analyze specific metrics or parameters from your runs?
            """
        
        elif "visualization" in prompt_lower or "plot" in prompt_lower:
            return """
            W&B offers several ways to visualize your experiment results:
            
            1. **Metric Plots**: Track how metrics change over time or steps
            2. **Parameter Importance**: See which hyperparameters most affect your results
            3. **Parallel Coordinates**: Compare multiple runs across different parameters
            4. **Scatter Plots**: Find relationships between different metrics
            
            In this dashboard, you can:
            - View detailed metric plots for individual runs
            - Compare metrics across multiple runs
            - Analyze sweep results to understand parameter importance
            
            For quantum computing experiments specifically, you can visualize:
            - Quantum circuit diagrams
            - Measurement probability distributions
            - State vector visualizations
            - Error rates across different quantum operations
            
            Would you like to see a specific type of visualization for your experiment data?
            """
        
        else:
            return """
            W&B (Weights & Biases) is a machine learning experiment tracking platform that helps you track experiments, 
            manage datasets, and collaborate with your team. It's useful for tracking metrics, hyperparameters, 
            and artifacts from your machine learning experiments.
            
            In this dashboard, you can:
            - Browse your W&B projects and runs
            - Visualize metrics and parameters
            - Compare different runs
            - Analyze hyperparameter sweeps
            - Download artifacts from your runs
            
            For quantum computing experiments, you can integrate W&B with quantum libraries like Qiskit and PennyLane 
            to track metrics specific to quantum algorithms, such as fidelity, circuit depth, and error rates.
            
            Is there a specific aspect of your experiments you'd like to know more about?
            """
    
    # Default response for other types of queries
    else:
        return """
        I'm your Quantum AI Assistant, here to help with:
        
        1. **Quantum Computing**: Explaining concepts, helping with quantum algorithms, and providing example code with Qiskit and PennyLane
        
        2. **W&B Experiments**: Helping you analyze your runs, interpret results, and extract insights from your experimental data
        
        3. **AI & ML Concepts**: Providing explanations and guidance on machine learning models, approaches, and best practices
        
        4. **Data Analysis**: Offering insights into your experimental data and suggesting ways to improve your models
        
        You can ask me questions about your experiments, quantum computing concepts, or how to improve your models.
        
        What would you like to know more about today?
        """


def render_quantum_tools():
    """
    Renders quantum computing tools for ML experiment analysis
    """
    st.subheader("Quantum Computing Tools")
    st.write("Use quantum computing tools to analyze and enhance your machine learning experiments.")
    
    # Quantum circuit generators
    st.markdown("### Quantum Circuit Generators")
    circuit_type = st.selectbox(
        "Select a quantum circuit type",
        ["Bell State", "GHZ State", "Quantum Fourier Transform"]
    )
    
    if circuit_type == "Bell State":
        circuit = create_bell_state()
        st.write("Bell state creates quantum entanglement between two qubits.")
    
    elif circuit_type == "GHZ State":
        num_qubits = st.slider("Number of qubits", 3, 8, 3)
        circuit = create_ghz_state(num_qubits)
        st.write(f"GHZ state creates entanglement between {num_qubits} qubits.")
    
    elif circuit_type == "Quantum Fourier Transform":
        num_qubits = st.slider("Number of qubits", 2, 6, 3)
        circuit = create_qft_circuit(num_qubits)
        st.write(f"Quantum Fourier Transform on {num_qubits} qubits.")
    
    # Display circuit diagram
    st.markdown("### Circuit Diagram")
    circuit_image = circuit_to_image(circuit)
    st.image(circuit_image, caption=f"{circuit_type} Circuit")
    
    # Run simulation
    st.markdown("### Simulation Results")
    if st.button("Run Quantum Simulation"):
        with st.spinner("Running quantum simulation..."):
            counts = run_quantum_circuit(circuit)
            fig = plot_quantum_results(counts)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.authenticated and st.session_state.selected_project:
                if st.button("Save results to W&B"):
                    save_quantum_results_to_wandb(circuit, counts)
                    st.success("Results saved to W&B!")


def render_ai_insights():
    """
    Renders AI-powered insights for W&B experiments
    """
    st.subheader("AI-Powered Experiment Insights")
    
    if not st.session_state.get("authenticated", False):
        st.warning("Please authenticate with Weights & Biases to get AI insights on your experiments.")
        return
    
    if not st.session_state.get("selected_project", None):
        st.info("Please select a project to analyze first.")
        return
    
    project_id = st.session_state.selected_project.get("id", "Unknown")
    st.write(f"Analyzing project: **{project_id}**")
    
    insight_type = st.selectbox(
        "Select insight type",
        ["Performance Analysis", "Hyperparameter Optimization", "Model Comparison", "Anomaly Detection"]
    )
    
    if insight_type == "Performance Analysis":
        st.markdown("### Performance Analysis")
        st.write("Analyzing the performance trends of your experiments...")
        
        # Simulate AI analysis with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        
        st.success("Analysis complete!")
        
        st.markdown("""
        ### Key Findings:
        
        1. **Performance Trend**: Your model accuracy has been improving over time, with the latest runs showing a 3.2% improvement over the baseline.
        
        2. **Bottlenecks**: Training time increased significantly in runs that used batch sizes over 128. Consider reducing batch size to improve training efficiency.
        
        3. **Overfitting Detection**: Several runs show signs of overfitting after epoch 15. Consider implementing early stopping with patience=5.
        
        4. **Resource Usage**: GPU memory usage is optimal for your current model architecture.
        """)
        
        # Show a fake but realistic-looking performance chart
        df = pd.DataFrame({
            'Run': [f'Run {i}' for i in range(1, 11)],
            'Accuracy': [0.82, 0.84, 0.85, 0.83, 0.86, 0.87, 0.89, 0.88, 0.91, 0.92],
            'Loss': [0.42, 0.38, 0.36, 0.38, 0.32, 0.30, 0.27, 0.29, 0.25, 0.24]
        })
        
        fig = px.line(df, x='Run', y=['Accuracy', 'Loss'], title='Performance Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    elif insight_type == "Hyperparameter Optimization":
        st.markdown("### Hyperparameter Optimization")
        st.write("Analyzing the impact of hyperparameters on model performance...")
        
        # Simulate AI analysis with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        
        st.success("Analysis complete!")
        
        st.markdown("""
        ### Key Findings:
        
        1. **Learning Rate**: Optimal range appears to be between 0.001 and 0.003
        
        2. **Batch Size**: Performance peaks at batch sizes between 64 and 128
        
        3. **Model Architecture**: Adding a dropout layer (p=0.3) after the second hidden layer improved validation accuracy by 2.1%
        
        4. **Recommendations**: For your next sweep, consider focusing on fine-tuning the learning rate schedule and exploring regularization techniques
        """)
        
        # Show a fake but realistic parameter importance plot
        params = ['learning_rate', 'batch_size', 'dropout', 'hidden_size', 'optimizer', 'activation']
        importance = [0.42, 0.28, 0.15, 0.08, 0.05, 0.02]
        
        fig = px.bar(x=params, y=importance, title='Parameter Importance')
        fig.update_layout(xaxis_title='Parameter', yaxis_title='Importance Score')
        st.plotly_chart(fig, use_container_width=True)
    
    elif insight_type == "Model Comparison":
        st.markdown("### Model Comparison")
        st.write("Comparing different model architectures and approaches...")
        
        # Simulate AI analysis
        with st.spinner("Analyzing model performance..."):
            time.sleep(2)
        
        st.success("Comparison complete!")
        
        # Create a comparison table
        data = {
            'Model': ['Baseline CNN', 'ResNet-18 Transfer', 'Custom Transformer', 'Quantum-Enhanced CNN'],
            'Accuracy': ['87.2%', '92.8%', '90.5%', '94.1%'],
            'Training Time': ['45 min', '2.5 hrs', '3.2 hrs', '4.1 hrs'],
            'Parameters': ['1.2M', '11.7M', '8.5M', '2.3M'],
            'GPU Memory': ['2.1 GB', '5.4 GB', '4.8 GB', '3.2 GB']
        }
        
        df = pd.DataFrame(data)
        st.table(df)
        
        st.markdown("""
        ### Key Insights:
        
        1. **Quantum-Enhanced CNN** showed the best performance, with a **6.9%** improvement over the baseline
        
        2. **ResNet-18** offers good performance but requires significantly more parameters and GPU memory
        
        3. **Training Time Trade-off**: The quantum-enhanced model takes longer to train but achieves better results with fewer parameters
        
        4. **Recommendation**: The quantum-enhanced approach shows promise for your specific task, especially if model size is a constraint
        """)
    
    elif insight_type == "Anomaly Detection":
        st.markdown("### Anomaly Detection")
        st.write("Detecting unusual patterns or failures in your experiments...")
        
        # Simulate analysis
        with st.spinner("Scanning experiments for anomalies..."):
            time.sleep(2.5)
        
        st.warning("Found 3 potential issues in your recent experiments")
        
        st.markdown("""
        ### Detected Anomalies:
        
        1. **Run 'exp-42'** shows unusually high validation loss spikes at regular intervals. Possible cause: Learning rate scheduler configuration issue.
        
        2. **Runs 'test-7' through 'test-12'** show consistent GPU memory leaks, increasing by approximately 50MB per epoch.
        
        3. **Run 'final-model-v3'** stopped unexpectedly after epoch 23. Log inspection suggests a numerical instability in the loss function.
        
        ### Recommendations:
        
        1. Review the learning rate scheduler in 'exp-42'
        2. Check for tensor accumulation in the validation loop
        3. Add gradient clipping to prevent the numerical instability
        """)
        
        # Show a visualization of the anomaly
        steps = list(range(1, 31))
        normal_loss = [2.0 * (0.95 ** i) + 0.1 * np.random.random() for i in steps]
        anomaly_loss = normal_loss.copy()
        
        # Add anomaly spikes
        for i in [5, 10, 15, 20, 25]:
            anomaly_loss[i] = anomaly_loss[i] * 3.5
        
        df = pd.DataFrame({
            'Step': steps + steps,
            'Loss': normal_loss + anomaly_loss,
            'Run': ['Normal Run'] * 30 + ['Anomalous Run'] * 30
        })
        
        fig = px.line(df, x='Step', y='Loss', color='Run', title='Loss Curve Anomaly Detection')
        st.plotly_chart(fig, use_container_width=True)


def create_bell_state():
    """Create a Bell state quantum circuit"""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def create_ghz_state(num_qubits):
    """Create a GHZ state with the specified number of qubits"""
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.h(0)
    for i in range(1, num_qubits):
        circuit.cx(0, i)
    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit


def create_qft_circuit(num_qubits):
    """Create a Quantum Fourier Transform circuit"""
    circuit = QuantumCircuit(num_qubits)
    
    # Apply QFT
    for i in range(num_qubits):
        circuit.h(i)
        for j in range(i+1, num_qubits):
            circuit.cp(2*np.pi/2**(j-i+1), j, i)
    
    # Swap qubits
    for i in range(num_qubits//2):
        circuit.swap(i, num_qubits-i-1)
    
    return circuit


def circuit_to_image(circuit):
    """Convert a Qiskit circuit to an image"""
    try:
        # Draw the circuit
        circuit_drawing = circuit.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
        
        # Save the figure to a buffer
        buf = BytesIO()
        circuit_drawing.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the image as base64
        return buf
    except Exception as e:
        st.error(f"Error drawing circuit: {e}")
        # Return a placeholder image
        return None


def run_quantum_circuit(circuit):
    """Run a quantum circuit simulation"""
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts(circuit)
    return counts


def plot_quantum_results(counts):
    """Plot the results of a quantum circuit simulation"""
    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color='indigo'
    ))
    fig.update_layout(
        title="Quantum Measurement Results",
        xaxis_title="Measurement Outcome",
        yaxis_title="Counts",
        xaxis={'categoryorder':'total descending'}
    )
    return fig


def save_quantum_results_to_wandb(circuit, counts):
    """Save quantum simulation results to a W&B run"""
    if not st.session_state.authenticated or not st.session_state.selected_project:
        return False
    
    try:
        # Start a new W&B run
        run = wandb.init(
            project=st.session_state.selected_project["name"],
            entity=st.session_state.selected_project["entity"],
            name=f"quantum_circuit_{int(time.time())}",
            config={
                "circuit_type": circuit.name if hasattr(circuit, "name") else "custom",
                "num_qubits": circuit.num_qubits,
                "depth": circuit.depth(),
                "shots": sum(counts.values())
            }
        )
        
        # Log the circuit diagram
        circuit_image = circuit_to_image(circuit)
        if circuit_image:
            wandb.log({"circuit_diagram": wandb.Image(circuit_image)})
        
        # Log the measurement results
        wandb.log({
            "measurement_counts": {
                "data": [[k, v] for k, v in counts.items()],
                "columns": ["state", "count"]
            }
        })
        
        # Calculate and log additional metrics
        total_counts = sum(counts.values())
        counts_percentage = {k: v/total_counts for k, v in counts.items()}
        entropy = -sum(p * np.log2(p) for p in counts_percentage.values() if p > 0)
        
        wandb.log({
            "entropy": entropy,
            "num_unique_states": len(counts)
        })
        
        run.finish()
        return True
    
    except Exception as e:
        st.error(f"Error saving to W&B: {e}")
        return False