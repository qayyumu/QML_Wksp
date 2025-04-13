import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the number of qubits based on the number of features
n_qubits = min(X.shape[1], 4)  # Iris has 4 features
n_layers = 4  # Increased number of layers

# Create a quantum device
dev = qml.device("default.qubit", wires=n_qubits)

def layer(weights, wires):
    # More expressive layer with multiple rotations
    for i in range(len(wires)):
        qml.RX(weights[0, i], wires=wires[i])
        qml.RY(weights[1, i], wires=wires[i])
        qml.RZ(weights[2, i], wires=wires[i])
    
    # Entanglement
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode classical data into quantum state
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
    
    # Apply multiple layers
    for layer_idx in range(n_layers):
        layer(weights[layer_idx], range(n_qubits))
    
    # Return measurements for each class using PauliZ measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]  # 3 classes for Iris

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Evaluate the model
def predict(X, weights):
    predictions = []
    for x in X:
        pred = quantum_circuit(x, weights)
        predictions.append(pred)
    predictions = np.array(predictions)
    probs = np.array([softmax(p) for p in predictions])
    return np.argmax(probs, axis=1)

def cost(weights, X, y):
    # Get predictions from quantum circuit
    predictions = []
    for x in X:
        pred = quantum_circuit(x, weights)
        predictions.append(pred)
    predictions = np.array(predictions)
    
    # Convert predictions to probabilities using softmax
    probs = np.array([softmax(p) for p in predictions])
    
    # Convert true labels to one-hot encoding
    one_hot = np.zeros((len(y), 3))
    one_hot[np.arange(len(y)), y] = 1
    
    # Calculate cross-entropy loss
    epsilon = 1e-10  # Small constant to avoid log(0)
    loss = -np.sum(one_hot * np.log(probs + epsilon)) / len(y)
    
    return loss

# Initialize random weights with more parameters per layer
weights = np.random.uniform(0, 2*np.pi, size=(n_layers, 3, n_qubits), requires_grad=True)

# Set up the optimizer with a smaller learning rate
opt = qml.GradientDescentOptimizer(stepsize=0.01)

# Training loop with more epochs
n_epochs = 100
history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

for epoch in range(n_epochs):
    weights, cost_val = opt.step_and_cost(lambda w: cost(w, X_train, y_train), weights)
    history['loss'].append(float(cost_val))
    
    # Calculate validation loss
    val_loss = cost(weights, X_test, y_test)
    history['val_loss'].append(float(val_loss))
    
    # Calculate accuracies
    train_pred = predict(X_train, weights)
    test_pred = predict(X_test, weights)
    history['accuracy'].append(np.mean(train_pred == y_train))
    history['val_accuracy'].append(np.mean(test_pred == y_test))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch + 1}, Cost: {cost_val:.4f}, Val Cost: {val_loss:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss History')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy History')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



train_predictions = predict(X_train, weights)
test_predictions = predict(X_test, weights)

train_accuracy = np.mean(train_predictions == y_train)
test_accuracy = np.mean(test_predictions == y_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print classification report with explicit labels
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, test_predictions, 
                          labels=[0, 1, 2],
                          target_names=data.target_names))

# Draw the circuit
print("\nCircuit diagram:")
print(qml.draw(quantum_circuit)(X_train[0], weights)) 