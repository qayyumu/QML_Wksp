import pennylane as qml
from pennylane import numpy as np

# Create a quantum device with 3 qubits
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def circuit(params):
    # Apply parameterized rotations to all qubits
    for i in range(3):
        qml.RX(params[i], wires=i)
        qml.RY(params[i+3], wires=i)
    
    # Create entanglement between qubits
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    
    # Apply more parameterized gates
    for i in range(3):
        qml.RZ(params[i+6], wires=i)
    
    # Create a GHZ state
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    
    # Return multiple measurements
    return [
        qml.probs(wires=[0, 1, 2]),  # Probability distribution
        qml.expval(qml.PauliZ(0)),   # Expectation value of Z on first qubit
        qml.expval(qml.PauliX(1)),   # Expectation value of X on second qubit
        qml.expval(qml.PauliY(2))    # Expectation value of Y on third qubit
    ]

# Initialize random parameters
params = np.random.uniform(0, 2*np.pi, size=9)

# Execute the circuit
probs, z_exp, x_exp, y_exp = circuit(params)

# Print the results
print("Measurement probabilities for all basis states:")
for i, state in enumerate(['000', '001', '010', '011', '100', '101', '110', '111']):
    print(f"|{state}⟩: {probs[i]:.4f}")

print("\nExpectation values:")
print(f"⟨Z⟩ on qubit 0: {z_exp:.4f}")
print(f"⟨X⟩ on qubit 1: {x_exp:.4f}")
print(f"⟨Y⟩ on qubit 2: {y_exp:.4f}")

# Draw the circuit
print("\nCircuit diagram:")
print(qml.draw(circuit)(params))
