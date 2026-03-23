# The Geometry of Privacy: A Two-Stage Analysis of Generative Membership Inference in Federated Learning

This is the repo implementing the experiments. It contains everything needed to run the theory simulation (`simulate_results.py`) with uv.

## Setup

```bash
# Install dependencies with uv
uv sync

# Run the simulation
uv run python simulate_results.py
```

## Contents

- `simulate_results.py` - Main simulation script
- `src/` - Minimal source modules (VAE, data loading)
- `tex/` - LaTeX report files
- `results/` - Output directory for simulation results
