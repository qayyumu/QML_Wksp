import pennylane as qml
from pennylane import numpy as np

# Create a quantum device with 2 qubits
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def basic_gates_example():
    # Pauli X gate (NOT gate)
    qml.PauliX(wires=0)
    
    # Pauli Y gate
    qml.PauliY(wires=1)
    
    # Pauli Z gate
    qml.PauliZ(wires=0)
    
    # Hadamard gate
    qml.Hadamard(wires=0)
    
    # CNOT gate (controlled-X)
    qml.CNOT(wires=[0, 1])
    
    # Rotation gates
    qml.RX(np.pi/2, wires=0)  # Rotation around X-axis
    qml.RY(np.pi/4, wires=1)  # Rotation around Y-axis
    qml.RZ(np.pi/3, wires=0)  # Rotation around Z-axis
    
    # Phase shift gate
    qml.PhaseShift(np.pi/2, wires=1)
    
    # SWAP gate
    qml.SWAP(wires=[0, 1])
    
    return qml.state()

# Print the quantum state after applying all gates
print("Final quantum state:")
print(basic_gates_example())

# Create another circuit to demonstrate measurement
@qml.qnode(dev)
def measurement_example():
    # Prepare a Bell state
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    
    # Measure in the computational basis
    return qml.probs(wires=[0, 1])

print("\nMeasurement probabilities:")
print(measurement_example())
