# Quantum Diffusion Language Model (Toy Experiment)

A proof-of-concept demonstrating the full pipeline:
**classical training -> quantum circuit conversion -> inference comparison**

## What this does

1. **Trains** a tiny diffusion-style language model (MLP denoiser, ~800 params) classically on M1 Max
2. **Converts** the trained denoiser into a 6-qubit quantum circuit using parameterized rotations
3. **Compares** inference across methods: classical MLP, quantum simulation, random baseline
4. **Predicts** expected T-9 hardware accuracy using tau-chrono (tau = 1 - F)

## Vocabulary

8 tokens (3 bits = 3 qubits per position):

| ID | Token |
|----|-------|
| 0  | the   |
| 1  | cat   |
| 2  | sat   |
| 3  | on    |
| 4  | a     |
| 5  | mat   |
| 6  | dog   |
| 7  | ran   |

## Architecture

### Classical Model
- **Task**: Masked token prediction (diffusion-style denoising)
- **Input**: All tokens in sentence with one masked + position encoding (61 dims)
- **Hidden layers**: 32 -> 16 neurons (ReLU)
- **Output**: Softmax over 8 tokens
- **Total params**: ~800
- **Training**: Adam optimizer, cross-entropy loss, 5000 steps

### Quantum Circuit (6 qubits)
```
Input qubits (3):  |0> --[X if bit=1]--[CNOT]--[barrier]--
Output qubits (3): |0> ---------------[CNOT]--[Ry]--[CNOT]--[Ry,Rz]--[CNOT]--[Ry]--[CNOT]--[Ry]-- Measure
```

- 3 input qubits encode the context token
- 3 output qubits produce the predicted token
- Rotation angles optimized to match the trained MLP's probability distribution
- ~15 rotation gates + ~8 CNOTs per circuit

## Usage

```bash
# Step 1: Train classically
python train_classical.py

# Step 2: Run inference comparison
python run_inference.py
```

## Dependencies

- numpy (required)
- qiskit + qiskit-aer (for quantum simulation)

```bash
pip install numpy qiskit qiskit-aer
```

## Files

| File | Description |
|------|-------------|
| `data.py` | Vocabulary, training sentences, input encoding |
| `model.py` | Classical MLP denoiser + Adam optimizer |
| `train_classical.py` | Training loop (5000 steps, saves weights) |
| `convert_to_quantum.py` | MLP -> quantum circuit conversion |
| `run_inference.py` | Run all methods + comparison table |

## tau-chrono Prediction

The tau-chrono framework predicts T-9 hardware accuracy before running:

```
tau = 1 - F_total
F_total = F_gates * F_readout
F_gates = prod(1 - e_gate) over all gates
```

With T-9 error rates (e_1q ~ 0.1%, e_2q ~ 1%, e_ro ~ 2%), the predicted
accuracy accounts for gate errors and decoherence.
