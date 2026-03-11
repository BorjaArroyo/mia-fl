# Workspace Agents Information

## Project Goal
Analyze Membership Inference Attack (MIA) risk in Federated Learning (FL) using a two-stage framework:
1.  **Stage I (Signal Survival):** Can an update survive aggregation and be detected?
2.  **Stage II (Signal Attribution):** Does a surviving update change a generative score in a way attributable to the target data?

The goal is to verify theoretical bounds and simulate risk trajectories (TS-MIA) under various data distributions (IID, Non-IID).

## Current Status
-   **Theory Verification**:
    -   **Stage I**: `verify_stage_one` confirms the Signal-to-Update Ratio (SUR) decomposition and the role of target-background alignment ($\lambda$) in aggregation cancellation.
    -   **Stage II**: `verify_stage_two` confirms the smoothness bound for attribution $J(u)$ using a VAE on MedMNIST data. A calibration/test split is used to estimate the smoothness constant $L$.
    -   **Geometry**: `generate_stage_two_geometry` maps the attribution surface ($J$ vs. alignment $\cos \theta$) across different perturbation magnitudes.

-   **Risk Simulation**:
    -   `simulate_trajectory` simulates Federated Learning rounds to track TS-MIA risk (SUR > threshold AND Attribution > 0).
    -   Scenarios covered:
        -   **IID**: Baseline.
        -   **Non-IID**: Label skew (heterogeneous).
        -   **Inverted**: Adversarial scenario ($\lambda \approx -1$).

## Environment
-   **Python Manager**: `uv`
-   **Dependencies**: `torch`, `torchvision`, `medmnist`, `numpy`, `matplotlib`, `ray` (via `pyproject.toml`)

## Key Entry Points
-   **Synchronous Simulation (Cluster)**: `uv run simulate_results.py --cluster ray://192.168.0.158:10001`
    -   Connects via Ray Client for real-time logs and synchronous result collection.
    -   Leverages `uv` integration for automatic dependency syncing on workers.
-   **Local Simulation**: `uv run simulate_results.py`
    -   Runs Stage I/II verification, Geometry generation, and Trajectory simulation locally.
    -   Outputs results to `results/theory_simulation.json`.
-   **Connectivity Check**: `uv run check_ray_sync.py --address ray://192.168.0.158:10001`
    -   Verifies that the cluster is reachable and that `uv` is correctly installing dependencies on remote workers.
-   **Plotting**: `uv run plot_results.py`
    -   Generates figures in `figs/` with professional styling (large fonts, seaborn-whitegrid) for LaTeX embedding:
        -   `verify_sur_geometry.png`: Stage I survival zones.
        -   `verify_attribution_bound.png`: Empirical vs Theoretical smoothness bounds.
        -   `stage_two_geometry.png`: $J(u)$ vs Alignment.
        -   `risk_trajectory.png`: Risk evolution over FL rounds.
        -   `non_iid_image_risk.png`: Non-IID example and risk curve.
-   **Report**: `tex/report.tex`
    -   Mathematical formulation of the two-stage framework and proofs (recently refined derivation steps for Stage I alignment $\lambda$ bounds).
    -   Includes foundational citations (e.g., McMahan 2017 for FL).
    -   Recently fixed compilation warnings, overfull hboxes, and hyperref grouping errors.
    -   Stage II "Generative score" subsection grounded in the Galende 2025 Bayesian hypothesis-testing framework.

## Directory Structure
-   `src/`: Source code
    -   `models/`: VAE implementation (`vae.py`)
    -   `data/`: Data loading and partitioning (`datasets.py`, `partitions.py`)
    -   `tasks/`: Simulation logic (`simulation_tasks.py`)
-   `results/`: Simulation output JSONs
-   `figs/`: Generated plots
-   `tex/`: LaTeX report and bibliography

## Key Concepts
-   **SUR (Signal-to-Update Ratio)**: Proxy for Stage I survival. Defined as `‖u‖/‖ΔW‖` (original definition). Closed form: SUR² = 1/(1 + r² + 2rλ). Both SUR and attribution are auditor-evaluable quantities.
-   **Attribution $J(u)$**: Centered advantage of the target update on the target's own data (Generative MIA score).
-   **TS-MIA**: Two-Stage MIA metric combining survival and attribution probabilities.

## Recent Changes
-   **Reviewer revision pass** (Feb 2026): Threat model reframed (auditor risk metric, SUR requires per-client access), TS-MIA subset claim fixed, client-level membership (round-level baseline W_t+B), γ-dominant lemma recast as remark with corrected concavity. SUR kept as original ‖u‖/‖ΔW‖ throughout.
