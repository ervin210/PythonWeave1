import wandb
import numpy as np
import datetime
import qiskit
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import Aer, AerSimulator
import matplotlib.pyplot as plt
import io
from PIL import Image

def create_bell_state():
    """Create a Bell state quantum circuit"""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
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

def run_quantum_circuit(circuit, shots=1024):
    """Run a quantum circuit simulation"""
    # Create the simulator
    simulator = AerSimulator()
    
    # Execute the circuit using the updated Qiskit API
    result = simulator.run(circuit, shots=shots).result()
    
    # Get the counts (measurement results)
    counts = result.get_counts(circuit)
    
    return counts

def plot_quantum_results(counts):
    """Plot the results of a quantum circuit simulation"""
    # Create bar chart of results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(counts.keys(), counts.values())
    ax.set_xlabel('Measurement Outcome')
    ax.set_ylabel('Counts')
    ax.set_title('Quantum Measurement Results')
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    return Image.open(buf)

def main():
    # Initialize W&B run
    run = wandb.init(
        project="quantum-assistant-demo",
        name=f"bell-state-experiment-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "circuit_type": "Bell State",
            "n_qubits": 2,
            "shots": 1024,
            "backend": "qasm_simulator"
        }
    )
    
    print("Creating and running Bell state circuit...")
    
    # Create Bell state circuit
    circuit = create_bell_state()
    
    # Save circuit diagram
    circuit_img = circuit_to_image(circuit)
    
    # Run the circuit
    counts = run_quantum_circuit(circuit)
    
    # Create a plot of the results
    results_img = plot_quantum_results(counts)
    
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
    
    print("Experiment complete and data pushed to W&B!")
    
    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    main()