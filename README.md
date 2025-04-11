# Quantum Machine Learning with PennyLane

This project demonstrates quantum machine learning implementations using PennyLane, a quantum computing framework.

## Project Contents

### 1. Basic Quantum Circuit
A simple quantum circuit implementation that demonstrates:
- Hadamard gate on the first qubit
- Pauli-X gate on the second qubit
- Controlled rotation (CRX) between the qubits
- Measurement of the Pauli-Z observable on the first qubit

### 2. Iris Classification Quantum Circuit
A quantum machine learning implementation for classifying Iris flowers using a simplified quantum circuit architecture:

#### Circuit Architecture
- Data encoding using RX and RY rotation gates
- Single parameterized layer with RY gates
- Nearest-neighbor entanglement using CNOT gates
- Final measurement on the first qubit

#### Features
- Uses 4 qubits for processing
- Implements binary classification (setosa vs versicolor)
- Includes data preprocessing and scaling
- Training with gradient descent optimization
- Visualization of classification results

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv qml_venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\qml_venv\Scripts\activate
```
- Unix/MacOS:
```bash
source qml_venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the quantum circuits:
```bash
# For basic circuit
python BasicCircuit.py

# For Iris classification
python Iris_qml_lowperf.py
```

## Requirements
- Python 3.9+
- See requirements.txt for package dependencies 