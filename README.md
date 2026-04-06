# M2 Deep Learning Coursework

This repository implements a **normalizing flow** model using affine coupling layers, trained on a 2D moons dataset.

## Getting Started

### Prerequisites

- Python >= 3.10
- Dependencies: PyTorch (>= 2.0), NumPy (>= 1.24), Matplotlib (>= 3.7)

### Installation

```bash
pip install .
```

### Running

Open and run `coursework.ipynb` from top to bottom. The notebook is self-contained: it defines the model, trains it, generates all figures, and writes results to `results.json`.

### Loading a Trained Checkpoint

The trained model is saved at `checkpoints/flow_full.pt`. To load it without re-training, run the model definition cells in the notebook first (Q1a and Q1b), then:

```python
ckpt = torch.load("checkpoints/flow_full.pt", weights_only=False)
flow = NormalizingFlow(**ckpt["config"])
flow.load_state_dict(ckpt["state_dict"])
flow.eval()
```

The checkpoint contains `state_dict`, `config` (model hyperparameters), and `seed`.

> **Note:** `NormalizingFlow` and `AffineCouplingLayer` are defined in the notebook. If loading from a standalone `.py` file, copy these class definitions from the notebook first.

### Reading the Writeup and Metrics

All correctness checks, training metrics, and written analysis are stored in `results.json`:

```python
import json

with open("results.json") as f:
    results = json.load(f)

# Correctness checks
print(results["correctness"])

# Training metrics (tinyset, train, val, test NLL)
print(results["training"])

# Written analysis
print(results["writeup"])
```

### Viewing Training Curves

Training and validation NLL over epochs are logged in `logs/training_curves.json`:

```python
import json

with open("logs/training_curves.json") as f:
    curves = json.load(f)
```

## Repository Structure

```
bc654/
├── coursework.ipynb        # Jupyter notebook with all code and analysis
├── pyproject.toml          # Project metadata and dependencies
├── results.json            # Correctness checks, training metrics, and writeup
├── M2_Coursework.pdf       # Coursework specification
├── data/
│   ├── moons_train.csv     # Training set (800 samples)
│   ├── moons_val.csv       # Validation set
│   └── moons_test.csv      # Test set
├── checkpoints/
│   └── flow_full.pt        # Trained model checkpoint
├── figs/
│   ├── Figure1c.pdf        # Correctness check plots
│   ├── Figure2a.pdf        # Tiny-subset training results
│   ├── Figure2c.pdf        # Full training results
│   └── Figure3b.pdf        # Flow surgery (shear family) visualisation
└── logs/
    └── training_curves.json # Training and validation NLL over epochs
```

#### Use of generative AI
Claude (Anthropic) was used to assist with code development and report writing. All outputs were reviewed and tested by the author, who takes full responsibility for the final submission.