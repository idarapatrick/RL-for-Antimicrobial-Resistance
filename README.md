# AMR Microscopy RL Project


This repository implements and compares multiple Reinforcement Learning agents that control a simulated microscopy analysis pipeline for antimicrobial resistance (AMR) detection in E. coli under antibiotic stress.

## 
![Video Demo](AMR.gif)

## Problem Summary
Design Reinforcement Learning agents that decide per-frame analysis effort under a constrained compute budget while maximizing detection performance and minimizing unnecessary computation. The environment simulates microscopy frames and trade-offs between accuracy and compute cost.

The agent must decide how much analysis to run at each frame under a limited compute budget. It balances:

- Detection accuracy (catch resistance events)
- Compute efficiency (avoid wasting budget)
- False alarm control

The environment is implemented as a custom Gymnasium environment with biologically motivated features and action costs.

## Dataset used for Calibration

DeepBcs Dataset:
https://github.com/HenriquesLab/DeepBacs.git

## Implemented Algorithms

- DQN (Stable Baselines3)
- PPO (Stable Baselines3)
- A2C (Stable Baselines3)
- REINFORCE (custom PyTorch implementation)
- Random policy baseline demo

Each algorithm is trained using a 10-run hyperparameter sweep.

## Project Structure

```text
environment/
	custom_env.py         # Gymnasium environment
	data_simulator.py     # Episode and feature simulation
	rendering.py          # Pygame visualizer

training/
	dqn_training.py       # DQN sweep
	pg_training.py        # PPO, A2C, REINFORCE sweeps

models/
	dqn/
	pg/

results/
	dqn/
	ppo/
	a2c/
	reinforce/
	figures/

logs/
	dqn/
	ppo/
	a2c/

main.py                 # Run best trained model in live simulation
demo_random_agent.py    # Random policy visual demo
plot_results.py         # Generate report-ready figures
requirements.txt
```

## Setup

### 1. Clone and enter project

```bash
git clone https://github.com/idarapatrick/Idara_Essien_rl_summative.git
cd Idara_Essien_rl_summative
```

### 2. Create a Python environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

Run random baseline demo:

```bash
python demo_random_agent.py
```

Run headless random demo and save screenshot:

```bash
python demo_random_agent.py --frames 200 --no-display
```

## Training

### DQN sweep (10 runs)

```bash
python training/dqn_training.py
```

### Policy gradient sweeps

Run all three:

```bash
python training/pg_training.py --algo all
```

Or run individually:

```bash
python training/pg_training.py --algo ppo
python training/pg_training.py --algo a2c
python training/pg_training.py --algo reinforce
```

Notes:

- PPO, A2C, and REINFORCE scripts support resume behavior and skip completed runs.
- Best runs are saved to `best_run.json` in each model directory.

## Run a Trained Agent

Auto-select best algorithm across available models:

```bash
python main.py
```

Run a specific algorithm:

```bash
python main.py --algo dqn
python main.py --algo ppo
python main.py --algo a2c
python main.py --algo reinforce
python main.py --algo random
```

Run multiple episodes:

```bash
python main.py --algo ppo --episodes 3
```

## Generate Figures

After training is complete:

```bash
python plot_results.py
```

This generates figures in `results/figures`, including:

- Learning curves
- DQN objective/evaluation curve
- Policy-gradient entropy proxy curves
- Convergence comparison
- Hyperparameter sensitivity
- Compute efficiency analysis

## Output Artifacts

- Models: `models/`
- Metrics summaries: `results/*/*_results_summary.csv`
- Per-run training data: `results/<algo>/<run_name>/`
- TensorBoard logs: `logs/`
- Best run metadata: `best_run.json`

## Reproducibility

- All training scripts set explicit seeds per run.
- Environment episodes are generated from seeded simulator instances.
- You can rerun sweeps to reproduce trends, then compare summary CSV files and generated figures.

## Suggested Workflow

1. Install dependencies
2. Run DQN and PG sweeps
3. Generate figures
4. Run `main.py` to visualize best-performing policy
5. Use summary CSV files and figures for report analysis

## Troubleshooting

- If pygame window does not open in headless environments, use `--no-display` for demo scripts.
- If no trained model is found, `main.py` falls back to a random agent.
- If training is interrupted, rerunning sweep scripts resumes completed runs where supported.
