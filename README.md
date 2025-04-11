# Basic Quantum Circuit using PennyLane

This project demonstrates a basic quantum circuit implementation using PennyLane, a quantum computing framework.

## Circuit Description

The quantum circuit implements:
- Hadamard gate on the first qubit
- Pauli-X gate on the second qubit
- Controlled rotation (CRX) between the qubits
- Measurement of the Pauli-Z observable on the first qubit

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

4. Run the quantum circuit:
```bash
python BasicCircuit.py
```

## Requirements
- Python 3.9+
- See requirements.txt for package dependencies 