import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Create a quantum device with 2 qubits
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def quantum_circuit(x, weights):
    # Encode classical data into quantum state
    qml.RX(x[0], wires=0)
    qml.RX(x[1], wires=1)
    
    # Apply trainable weights
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    
    # Create entanglement
    qml.CNOT(wires=[0, 1])
    
    # Apply more trainable weights
    qml.RY(weights[2], wires=0)
    qml.RY(weights[3], wires=1)
    
    # Return measurement
    return qml.expval(qml.PauliZ(0))

def cost(weights, x, y):
    # Compute predictions
    predictions = np.array([quantum_circuit(x_i, weights) for x_i in x])
    
    # Calculate mean squared error
    return np.mean((predictions - y) ** 2)

# Generate synthetic data
def generate_data(n_samples):
    X = np.random.uniform(0, 2*np.pi, (n_samples, 2))
    y = np.array([np.sin(x[0]) * np.cos(x[1]) for x in X])
    return X, y

# Generate training data
X_train, y_train = generate_data(100)

# Initialize random weights with proper type and correct number of parameters
n_weights = 4  # Number of weights needed for the circuit
weights = np.random.uniform(0, 2*np.pi, size=n_weights, requires_grad=True)

# Set up the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# Training loop
n_epochs = 100
cost_history = []

for epoch in range(n_epochs):
    # Compute gradient and update weights
    weights, cost_val = opt.step_and_cost(lambda w: cost(w, X_train, y_train), weights)
    
    # Convert cost value to float and store
    cost_history.append(float(cost_val))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}, Cost: {cost_val:.4f}")

# Plot the cost history
plt.figure(figsize=(8, 4))
plt.plot(cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Training Progress')
plt.grid(True)
plt.show()

# Test the trained model
X_test, y_test = generate_data(10)
predictions = np.array([quantum_circuit(x, weights) for x in X_test])

print("\nTest Results:")
for i, (true, pred) in enumerate(zip(y_test, predictions)):
    print(f"Sample {i+1}: True = {true:.4f}, Predicted = {pred:.4f}")

# Draw the circuit
print("\nCircuit diagram:")
print(qml.draw(quantum_circuit)(X_test[0], weights)) 