import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and prepare the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# We'll use only two classes for simplicity (setosa and versicolor)
X = X[y != 2]
y = y[y != 2]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to float32 for better compatibility
X = np.array(X, requires_grad=False)
y = np.array(y, requires_grad=False)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the quantum device
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit with simplified architecture
@qml.qnode(dev)
def circuit(weights, inputs):
    # Data encoding with rotation gates
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i] * np.pi, wires=i)
    
    # Single parameterized layer
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    
    # Simple entanglement (nearest neighbor)
    for i in range(n_qubits-1):
        qml.CNOT(wires=[i, i+1])
    
    # Final measurement
    return qml.expval(qml.PauliZ(0))

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss

def cost(weights, features, labels):
    predictions = [circuit(weights, f) for f in features]
    return square_loss(labels, predictions)

# Initialize weights (reduced number of parameters due to simplified architecture)
np.random.seed(42)
weights = 0.1 * np.random.randn(n_qubits, requires_grad=True)

# Training parameters
opt = qml.GradientDescentOptimizer(stepsize=0.4)
batch_size = 5
epochs = 30

# Training loop
for epoch in range(epochs):
    # Shuffle the training data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[shuffle_idx]
    y_train_shuffled = y_train[shuffle_idx]
    
    # Train in batches
    for batch_idx in range(0, len(X_train_shuffled), batch_size):
        batch_features = X_train_shuffled[batch_idx:batch_idx + batch_size]
        batch_labels = y_train_shuffled[batch_idx:batch_idx + batch_size]
        
        weights = opt.step(lambda w: cost(w, batch_features, batch_labels), weights)
    
    # Calculate training cost
    train_cost = cost(weights, X_train_shuffled, y_train_shuffled)
    print(f"Epoch {epoch + 1}/{epochs}, Cost: {train_cost:.4f}")

# Test the model
test_predictions = np.array([circuit(weights, x) for x in X_test])
test_predictions = (test_predictions > 0.0).astype(int)

# Calculate accuracy
accuracy = np.mean(test_predictions == y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_predictions, cmap='viridis')
plt.title('Quantum Classifier Predictions on Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Predicted Class')
plt.show()
