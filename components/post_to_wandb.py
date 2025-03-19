import streamlit as st
import wandb
import pandas as pd
import numpy as np
import tempfile
import json
import os
import io
from datetime import datetime

import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt
from PIL import Image

def post_to_wandb():
    """
    Post experiments, subscription data, and feature updates directly to W&B
    """
    st.header("Post to Weights & Biases")
    
    # Check authentication
    if not st.session_state.authenticated:
        st.warning("You need to log in first to post to Weights & Biases")
        return
    
    # Get projects
    api = wandb.Api()
    username = api.viewer.entity
    
    # Post options tabs
    post_tab, subscription_tab, feature_tab = st.tabs([
        "Post Experiment", "Post Subscription Data", "Post New Feature"
    ])
    
    # Post Experiment Tab
    with post_tab:
        st.subheader("Post Quantum Experiment to W&B")
        
        # Get project list for selection
        projects = []
        try:
            for project in api.projects(username):
                projects.append(f"{project.entity}/{project.name}")
        except Exception as e:
            st.error(f"Error fetching projects: {str(e)}")
        
        if not projects:
            st.warning("No projects found. Create a project in W&B first.")
            return
        
        # Project selection
        selected_project = st.selectbox(
            "Select a project to post to:",
            options=projects,
            index=0
        )
        
        # Experiment settings
        st.subheader("Experiment Settings")
        
        exp_name = st.text_input("Experiment Name", value=f"quantum_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Quantum circuit type
        circuit_type = st.selectbox(
            "Quantum Circuit Type",
            options=["Bell State", "GHZ State", "QFT Circuit", "Random Circuit"],
            index=0
        )
        
        # Circuit parameters
        n_qubits = st.slider("Number of Qubits", min_value=2, max_value=10, value=2)
        
        if circuit_type == "Random Circuit":
            circuit_depth = st.slider("Circuit Depth", min_value=1, max_value=10, value=3)
        
        shots = st.slider("Number of Shots", min_value=100, max_value=10000, value=1024)
        
        # Execution
        if st.button("Run and Post Experiment"):
            with st.spinner("Running quantum circuit..."):
                # Create the circuit based on the selected type
                if circuit_type == "Bell State":
                    circuit = create_bell_state(n_qubits)
                elif circuit_type == "GHZ State":
                    circuit = create_ghz_state(n_qubits)
                elif circuit_type == "QFT Circuit":
                    circuit = create_qft_circuit(n_qubits)
                elif circuit_type == "Random Circuit":
                    circuit = create_random_circuit(n_qubits, circuit_depth)
                
                # Run the circuit
                sampler = Sampler()
                job = sampler.run(circuit, shots=shots)
                result = job.result()
                counts = result.quasi_dists[0].binary_probabilities(num_bits=n_qubits)
                # Convert to expected format for visualization
                counts = {k: int(v * shots) for k, v in counts.items()}
                
                # Generate circuit diagram
                circuit_img = circuit_to_image(circuit)
                
                # Generate results histogram
                results_img = plot_results(counts)
                
                # Initialize W&B run
                run = wandb.init(project=selected_project.split("/")[1], entity=selected_project.split("/")[0], name=exp_name)
                
                # Log metadata
                wandb.config.update({
                    "circuit_type": circuit_type,
                    "n_qubits": n_qubits,
                    "shots": shots
                })
                
                if circuit_type == "Random Circuit":
                    wandb.config.update({"circuit_depth": circuit_depth})
                
                # Log circuit diagram and results
                wandb.log({
                    "circuit_diagram": wandb.Image(circuit_img),
                    "results_histogram": wandb.Image(results_img),
                    "counts": counts
                })
                
                # Add quantum-specific metrics
                counts_array = np.array(list(counts.values()))
                wandb.log({
                    "max_probability": np.max(counts_array) / shots,
                    "entropy": calculate_entropy(counts, shots),
                    "num_unique_states": len(counts),
                })
                
                # Finish the run
                wandb.finish()
                
                st.success(f"Experiment posted successfully to {selected_project}")
                st.markdown(f"[View experiment on W&B](https://wandb.ai/{selected_project}/runs/{run.id})")
    
    # Post Subscription Data Tab
    with subscription_tab:
        st.subheader("Post Subscription Data to W&B")
        
        # Project selection
        selected_subscription_project = st.selectbox(
            "Select a project for subscription data:",
            options=projects,
            index=0,
            key="sub_project"
        )
        
        # Subscription metrics
        st.subheader("Subscription Metrics")
        
        # Active subscriptions count
        active_basic = st.number_input("Active Basic Subscriptions", min_value=0, value=12)
        active_pro = st.number_input("Active Professional Subscriptions", min_value=0, value=8)
        active_enterprise = st.number_input("Active Enterprise Subscriptions", min_value=0, value=3)
        
        # Revenue metrics
        total_mrr = active_basic * 9.99 + active_pro * 29.99 + active_enterprise * 99.99
        
        # Calculate statistics
        subscription_stats = {
            "active_basic": active_basic,
            "active_pro": active_pro,
            "active_enterprise": active_enterprise,
            "total_subscribers": active_basic + active_pro + active_enterprise,
            "monthly_recurring_revenue": total_mrr,
            "average_revenue_per_user": total_mrr / max(1, (active_basic + active_pro + active_enterprise)),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Show calculated stats
        st.metric("Total Subscribers", subscription_stats["total_subscribers"])
        st.metric("Monthly Recurring Revenue", f"${subscription_stats['monthly_recurring_revenue']:.2f}")
        st.metric("Average Revenue Per User", f"${subscription_stats['average_revenue_per_user']:.2f}")
        
        # Post to W&B
        if st.button("Post Subscription Data"):
            with st.spinner("Posting subscription data..."):
                # Initialize W&B run
                run = wandb.init(
                    project=selected_subscription_project.split("/")[1], 
                    entity=selected_subscription_project.split("/")[0],
                    name=f"subscription_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                # Log subscription data
                wandb.config.update({
                    "data_type": "subscription_metrics",
                    "timestamp": subscription_stats["timestamp"]
                })
                
                # Log metrics
                wandb.log({
                    "active_basic_subscriptions": subscription_stats["active_basic"],
                    "active_pro_subscriptions": subscription_stats["active_pro"],
                    "active_enterprise_subscriptions": subscription_stats["active_enterprise"],
                    "total_subscribers": subscription_stats["total_subscribers"],
                    "monthly_recurring_revenue": subscription_stats["monthly_recurring_revenue"],
                    "average_revenue_per_user": subscription_stats["average_revenue_per_user"]
                })
                
                # Create a pie chart of subscription distribution
                fig, ax = plt.subplots(figsize=(8, 8))
                labels = ['Basic', 'Professional', 'Enterprise']
                sizes = [active_basic, active_pro, active_enterprise]
                colors = ['#ff9999','#66b3ff','#99ff99']
                explode = (0.1, 0, 0)
                ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                plt.title('Subscription Distribution')
                
                # Save the figure to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Log the chart as an image
                wandb.log({"subscription_distribution": wandb.Image(Image.open(buf))})
                
                # Create a bar chart of MRR by plan
                fig, ax = plt.subplots(figsize=(10, 6))
                plan_labels = ['Basic', 'Professional', 'Enterprise']
                mrr_values = [active_basic * 9.99, active_pro * 29.99, active_enterprise * 99.99]
                
                ax.bar(plan_labels, mrr_values, color=['#ff9999','#66b3ff','#99ff99'])
                ax.set_xlabel('Subscription Plan')
                ax.set_ylabel('Monthly Recurring Revenue ($)')
                ax.set_title('MRR by Subscription Plan')
                
                # Add values on top of the bars
                for i, v in enumerate(mrr_values):
                    ax.text(i, v + 5, f'${v:.2f}', ha='center')
                
                # Save the figure to a buffer
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png')
                buf2.seek(0)
                
                # Log the chart as an image
                wandb.log({"mrr_by_plan": wandb.Image(Image.open(buf2))})
                
                # Finish the run
                wandb.finish()
                
                st.success(f"Subscription data posted successfully to {selected_subscription_project}")
    
    # Post New Feature Tab
    with feature_tab:
        st.subheader("Post New Feature to W&B")
        
        # Project selection
        selected_feature_project = st.selectbox(
            "Select a project for feature announcement:",
            options=projects,
            index=0,
            key="feature_project"
        )
        
        # Feature details
        feature_name = st.text_input("Feature Name", value="")
        feature_description = st.text_area("Feature Description", value="")
        
        # Feature type
        feature_type = st.selectbox(
            "Feature Type",
            options=["New Feature", "Enhancement", "Bug Fix", "Performance Improvement"],
            index=0
        )
        
        # Version
        version = st.text_input("Version", value="1.0.0")
        
        # Feature screenshot
        feature_image = st.file_uploader("Feature Screenshot (optional)", type=["jpg", "png", "jpeg"])
        
        # Post feature
        if st.button("Post Feature"):
            if not feature_name or not feature_description:
                st.error("Feature name and description are required.")
                return
                
            with st.spinner("Posting feature..."):
                # Initialize W&B run
                run = wandb.init(
                    project=selected_feature_project.split("/")[1], 
                    entity=selected_feature_project.split("/")[0],
                    name=f"feature_{feature_name.lower().replace(' ', '_')}_{version}"
                )
                
                # Log feature details
                wandb.config.update({
                    "data_type": "feature_announcement",
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "version": version,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Create a feature card
                feature_card = f"""
                # {feature_name}
                
                **Type:** {feature_type}  
                **Version:** {version}  
                **Released:** {datetime.now().strftime("%Y-%m-%d")}
                
                ## Description
                
                {feature_description}
                """
                
                # Log feature card as markdown
                wandb.log({"feature_details": wandb.Html(markdown_to_html(feature_card))})
                
                # Log feature image if provided
                if feature_image:
                    img = Image.open(feature_image)
                    wandb.log({"feature_screenshot": wandb.Image(img)})
                
                # Finish the run
                wandb.finish()
                
                st.success(f"Feature posted successfully to {selected_feature_project}")

# Helper functions

def create_bell_state(n_qubits=2):
    """Create a Bell state quantum circuit"""
    if n_qubits < 2:
        n_qubits = 2  # Minimum 2 qubits for Bell state
    
    circuit = QuantumCircuit(n_qubits)
    
    # Apply Hadamard to first qubit
    circuit.h(0)
    
    # Apply CNOT gates to create entanglement
    for i in range(1, n_qubits):
        circuit.cx(0, i)
    
    # Measure all qubits
    circuit.measure_all()
    
    return circuit

def create_ghz_state(n_qubits):
    """Create a GHZ state with the specified number of qubits"""
    circuit = QuantumCircuit(n_qubits)
    
    # Apply Hadamard to first qubit
    circuit.h(0)
    
    # Apply CNOT gates to create entanglement
    for i in range(1, n_qubits):
        circuit.cx(0, i)
    
    # Measure all qubits
    circuit.measure_all()
    
    return circuit

def create_qft_circuit(n_qubits):
    """Create a Quantum Fourier Transform circuit"""
    circuit = QuantumCircuit(n_qubits)
    
    # Initialize with a non-trivial state
    for i in range(n_qubits):
        circuit.h(i)
    
    # Apply QFT
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            circuit.cp(np.pi / 2**(j-i), i, j)
        circuit.h(i)
    
    # Measure all qubits
    circuit.measure_all()
    
    return circuit

def create_random_circuit(n_qubits, depth):
    """Create a random quantum circuit"""
    import numpy.random as random
    
    circuit = QuantumCircuit(n_qubits)
    
    # Apply random gates
    for _ in range(depth):
        # Apply random single-qubit gates
        for i in range(n_qubits):
            gate_type = random.randint(0, 3)
            if gate_type == 0:
                circuit.h(i)
            elif gate_type == 1:
                circuit.x(i)
            elif gate_type == 2:
                circuit.y(i)
            else:
                circuit.z(i)
        
        # Apply random two-qubit gates
        for i in range(n_qubits - 1):
            if random.random() > 0.5:
                circuit.cx(i, (i + 1) % n_qubits)
    
    # Measure all qubits
    circuit.measure_all()
    
    return circuit

def circuit_to_image(circuit):
    """Convert a Qiskit circuit to an image"""
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(10, 6))
    circuit_diagram = circuit.draw(output='mpl')
    
    # Save figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return Image.open(buf)

def plot_results(counts):
    """Plot the results of a quantum circuit simulation"""
    from matplotlib import pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plot_histogram(counts)
    
    # Save figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return Image.open(buf)

def calculate_entropy(counts, shots):
    """Calculate the Shannon entropy of the measurement results"""
    probabilities = [count / shots for count in counts.values()]
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

def markdown_to_html(markdown_text):
    """Convert markdown to HTML"""
    import markdown
    return markdown.markdown(markdown_text)