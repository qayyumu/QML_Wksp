import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set up the quantum device
dev = qml.device("default.qubit", wires=2)

# Define the quantum circuit
@qml.qnode(dev, interface="autograd")
def circuit(params, x):
    # Encode the input data
    qml.RX(x[0], wires=0)
    qml.RY(x[1], wires=1)
    
    # Apply trainable parameters
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    
    # Return the expectation value
    return qml.expval(qml.PauliZ(0))

# Define the cost function
def cost(params, x, y):
    predictions = [circuit(params, x_i) for x_i in x]
    return np.mean((np.array(predictions) - y) ** 2)

# Generate some simple training data
np.random.seed(42)
X = np.random.uniform(0, 2 * np.pi, (20, 2))
Y = np.array([1 if x[0] > np.pi else -1 for x in X], dtype=float)

# Initialize parameters
params = np.random.uniform(0, 2 * np.pi, 2, requires_grad=True)

# Set up the optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# Training loop
cost_history = []
for i in range(100):
    params, cost_val = opt.step_and_cost(lambda p: cost(p, X, Y), params)
    cost_history.append(float(cost_val))
    if i % 10 == 0:
        print(f"Step {i}: Cost = {float(cost_val):.4f}")

# Plot the cost history
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Training Progress")
plt.show()

# Test the trained model
test_X = np.random.uniform(0, 2 * np.pi, (5, 2))
predictions = [float(circuit(params, x_i)) for x_i in test_X]
print("\nTest predictions:")
for i, pred in enumerate(predictions):
    print(f"Input {test_X[i]}: Prediction = {pred:.4f}")
