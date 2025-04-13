import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import boto3
from braket.aws import AwsDevice
from braket.circuits import Circuit
from braket.devices import LocalSimulator
import time
import logging

AWS_USE = False
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_aws_device():
    try:
        # AWS configuration
        session = boto3.Session()
        region = session.region_name
        if region is None:
            region = 'us-east-1'  # Default to us-east-1 if region not set
            logger.info(f"Using default region: {region}")
        
        # Check AWS credentials
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        logger.info(f"Connected to AWS as: {identity['Arn']}")
        
        # Get the quantum simulator device
        device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")
        logger.info(f"Using AWS Braket device: {device.name}")
        
        return device
    except Exception as e:
        logger.error(f"Error setting up AWS device: {str(e)}")
        raise

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

# Set up AWS device
device = setup_aws_device()
n_qubits = 4
if AWS_USE:
    dev = qml.device("braket.aws.qubit", device_arn=device.arn, wires=n_qubits, shots=1000)
else:
    dev = qml.device("braket.local.qubit", wires=n_qubits, shots=1000)
    
# Define the quantum circuit with improved architecture
@qml.qnode(dev)
def circuit(weights, inputs):
    # First layer: Data encoding with rotation gates
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
        qml.RY(inputs[i] * np.pi, wires=i)
    
    # First parameterized layer
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    
    # First entanglement layer (ring topology)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i+1) % n_qubits])
    
    # Second parameterized layer
    for i in range(n_qubits):
        qml.RZ(weights[i+n_qubits], wires=i)
    
    # Second entanglement layer (all-to-all connectivity)
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            qml.CNOT(wires=[i, j])
    
    # Third parameterized layer
    for i in range(n_qubits):
        qml.RX(weights[i+2*n_qubits], wires=i)
    
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

# Initialize weights (increased number of parameters due to new architecture)
np.random.seed(42)
weights = 0.1 * np.random.randn(3 * n_qubits, requires_grad=True)

# Training parameters
opt = qml.GradientDescentOptimizer(stepsize=0.4)
batch_size = 20
epochs = 5

# Training loop with AWS-specific error handling
for epoch in range(epochs):
    try:
        # Shuffle the training data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[shuffle_idx]
        y_train_shuffled = y_train[shuffle_idx]
        
        # Train in batches
        for batch_idx in range(0, len(X_train_shuffled), batch_size):
            batch_features = X_train_shuffled[batch_idx:batch_idx + batch_size]
            batch_labels = y_train_shuffled[batch_idx:batch_idx + batch_size]
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.1)
            print("Batch Processing",batch_idx)
            weights = opt.step(lambda w: cost(w, batch_features, batch_labels), weights)
        
        # Calculate training cost
        train_cost = cost(weights, X_train_shuffled, y_train_shuffled)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Cost: {train_cost:.4f}")
    except Exception as e:
        logger.error(f"Error during training epoch {epoch + 1}: {str(e)}")
        raise

# Test the model with error handling
try:
    test_predictions = np.array([circuit(weights, x) for x in X_test])
    test_predictions = (test_predictions > 0.0).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(test_predictions == y_test)
    logger.info(f"\nTest accuracy: {accuracy:.4f}")
except Exception as e:
    logger.error(f"Error during testing: {str(e)}")
    raise

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_predictions, cmap='viridis')
plt.title('Quantum Classifier Predictions on Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Predicted Class')
plt.show()
