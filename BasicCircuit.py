import pennylane as qml
from pennylane import numpy as np

# Create a quantum device with 2 qubits
dev = qml.device("default.qubit", wires=2)

# Define a quantum circuit
@qml.qnode(dev)
def circuit(phi):
    # Apply Hadamard gate to first qubit
    qml.Hadamard(wires=0)
    
    # Apply Pauli-X gate to second qubit
    qml.PauliX(wires=1)
    
    # Apply controlled rotation
    qml.CRX(phi, wires=[0, 1])
    
    # Measure the expectation value of Pauli-Z on first qubit
    return qml.expval(qml.PauliZ(0))

# Define a circuit to get the state
@qml.qnode(dev)
def state_circuit(phi):
    qml.Hadamard(wires=0)
    qml.PauliX(wires=1)
    qml.CRX(phi, wires=[0, 1])
    return qml.state()

# Define the parameter
phi = np.pi/4

# Execute the circuits
result = circuit(phi)
state = state_circuit(phi)

print(f"Circuit result: {result}")
print(f"Quantum state: {state}")
