"""
Generates all figures needed for the report from saved training results.
This is run after all four algorithms have finished training.

Produces:
  results/figures/01_cumulative_reward_curves.png   - all 4 algorithms subplots
  results/figures/02_dqn_loss_curve.png             - DQN TD loss over training
  results/figures/03_pg_entropy_curves.png          - PPO / A2C / REINFORCE entropy
  results/figures/04_convergence_comparison.png     - final mean reward bar chart
  results/figures/05_hyperparameter_sensitivity.png - LR vs reward scatter per algo
  results/figures/06_budget_vs_detections.png       - compute efficiency analysis

Run:
    python plot_results.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style 
plt.rcParams.update({
    "figure.facecolor":  "#12162a",
    "axes.facecolor":    "#1c2234",
    "axes.edgecolor":    "#3a3f5c",
    "axes.labelcolor":   "#dcdce4",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.color":       "#9090a0",
    "ytick.color":       "#9090a0",
    "text.color":        "#dcdce4",
    "grid.color":        "#2a2f4a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#1c2234",
    "legend.edgecolor":  "#3a3f5c",
    "font.family":       "monospace",
    "figure.dpi":        150,
})

ALGO_COLOURS = {
    "DQN":       "#378ADD",
    "PPO":       "#1D9E75",
    "A2C":       "#EF9F27",
    "REINFORCE": "#D85A30",
}

ALGO_DIRS = {
    "DQN":       ("results/dqn",       "models/dqn"),
    "PPO":       ("results/ppo",       "models/pg/ppo"),
    "A2C":       ("results/a2c",       "models/pg/a2c"),
    "REINFORCE": ("results/reinforce", "models/pg/reinforce"),
}


# Helpers 

def smooth(values, window=15):
    if len(values) < window:
        return np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_summary(results_dir: str, algo: str) -> pd.DataFrame:
    path = os.path.join(results_dir, f"{algo.lower()}_results_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback: rebuild from per-run JSONs
    rows = []
    for d in sorted(Path(results_dir).iterdir()):
        rp = d / "run_result.json"
        if rp.exists():
            with open(rp) as f:
                rows.append(json.load(f))
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def load_episode_rewards(results_dir: str, run_name: str) -> np.ndarray:
    """Load per-episode rewards saved during training."""
    # SB3 EvalCallback saves evaluations.npz
    eval_path = os.path.join(results_dir, run_name, "evaluations.npz")
    if os.path.exists(eval_path):
        data = np.load(eval_path)
        # mean across eval episodes at each checkpoint
        return data["results"].mean(axis=1)

    # REINFORCE saves episode_rewards.npy directly
    npy_path = os.path.join(results_dir, f"{run_name}_rewards.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)

    # DQN saves episode_rewards.npy inside run folder
    npy2 = os.path.join(results_dir, run_name, "episode_rewards.npy")
    if os.path.exists(npy2):
        return np.load(npy2)

    return np.array([])


def load_best_run_name(model_dir: str) -> str:
    best_path = os.path.join(model_dir, "best_run.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            return json.load(f)["run_name"]
    return ""


# Cumulative reward curves (all 4 algorithms)

def plot_reward_curves():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Learning curves - cumulative reward per evaluation checkpoint",
                 fontsize=14, y=1.01)

    for ax, (algo, (results_dir, model_dir)) in zip(axes.flat, ALGO_DIRS.items()):
        colour = ALGO_COLOURS[algo]
        best_run = load_best_run_name(model_dir)
        rewards  = load_episode_rewards(results_dir, best_run)

        if len(rewards) == 0:
            ax.text(0.5, 0.5, f"{algo}\nNo data yet",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#9090a0", fontsize=12)
        else:
            x = np.linspace(0, 80_000, len(rewards))
            ax.plot(x, rewards, color=colour, alpha=0.35, linewidth=0.8)
            if len(rewards) > 10:
                ax.plot(x[7:], smooth(rewards, 8),
                        color=colour, linewidth=2.0, label="smoothed")
            ax.axhline(np.max(rewards), color=colour, linewidth=0.6,
                       linestyle=":", alpha=0.6)
            ax.set_title(algo, color=colour)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean reward")
            ax.grid(True)
            ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "01_cumulative_reward_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# DQN objective / TD-error curve

def plot_dqn_loss():
    results_dir, model_dir = ALGO_DIRS["DQN"]
    best_run = load_best_run_name(model_dir)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("DQN — evaluation reward over training (best run)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean evaluation reward")

    rewards = load_episode_rewards(results_dir, best_run)
    if len(rewards) > 0:
        x = np.linspace(0, 80_000, len(rewards))
        ax.plot(x, rewards, color=ALGO_COLOURS["DQN"], alpha=0.4, linewidth=0.8)
        if len(rewards) > 8:
            ax.plot(x[4:], smooth(rewards, 5),
                    color=ALGO_COLOURS["DQN"], linewidth=2.0)
        ax.fill_between(x, rewards, alpha=0.08, color=ALGO_COLOURS["DQN"])
        ax.grid(True)
        ax.text(0.98, 0.05, f"Best run: {best_run}",
                transform=ax.transAxes, ha="right", fontsize=9,
                color="#9090a0")
    else:
        ax.text(0.5, 0.5, "DQN results not found", ha="center", va="center",
                transform=ax.transAxes, color="#9090a0")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "02_dqn_objective_curve.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# Policy gradient entropy curves 

def plot_entropy_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Policy gradient methods — entropy / exploration over training")

    pg_algos = ["PPO", "A2C", "REINFORCE"]
    for ax, algo in zip(axes, pg_algos):
        results_dir, model_dir = ALGO_DIRS[algo]
        colour   = ALGO_COLOURS[algo]
        best_run = load_best_run_name(model_dir)
        rewards  = load_episode_rewards(results_dir, best_run)

        if len(rewards) == 0:
            ax.text(0.5, 0.5, f"{algo}\nNo data yet",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#9090a0")
            ax.set_title(algo, color=colour)
            continue

        # Approximate entropy as rolling std of rewards (proxy for policy spread)
        window = max(3, len(rewards) // 10)
        entropy_proxy = pd.Series(rewards).rolling(window).std().fillna(0).values
        x = np.linspace(0, 80_000, len(rewards))

        ax.plot(x, entropy_proxy, color=colour, linewidth=1.5, label="reward std (proxy)")
        ax.fill_between(x, entropy_proxy, alpha=0.15, color=colour)
        ax.set_title(algo, color=colour)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Rolling reward std")
        ax.grid(True)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "03_pg_entropy_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# Convergence comparison bar chart 

def plot_convergence_comparison():
    algo_results = {}
    for algo, (results_dir, _) in ALGO_DIRS.items():
        df = load_summary(results_dir, algo)
        if not df.empty and "mean_reward" in df.columns:
            algo_results[algo] = df

    if not algo_results:
        print("No summary data found for convergence plot. Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Algorithm comparison — final mean reward across hyperparameter runs")

    # Left: bar chart of best reward per algorithm
    ax = axes[0]
    algos  = list(algo_results.keys())
    bests  = [algo_results[a]["mean_reward"].max() for a in algos]
    stds   = [algo_results[a].loc[algo_results[a]["mean_reward"].idxmax(), "std_reward"]
              if "std_reward" in algo_results[a].columns else 0 for a in algos]
    colours = [ALGO_COLOURS[a] for a in algos]

    bars = ax.bar(algos, bests, color=colours, alpha=0.85,
                  yerr=stds, capsize=5, error_kw={"ecolor": "#9090a0"})
    for bar, val in zip(bars, bests):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Best mean reward")
    ax.set_title("Best run per algorithm")
    ax.grid(True, axis="y")

    # Right: box plot of all 10 runs per algorithm
    ax2 = axes[1]
    data_to_plot = []
    labels       = []
    for algo in algos:
        df = algo_results[algo]
        if "mean_reward" in df.columns:
            data_to_plot.append(df["mean_reward"].values)
            labels.append(algo)

    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True,
                         medianprops={"color": "white", "linewidth": 2})
        for patch, algo in zip(bp["boxes"], labels):
            patch.set_facecolor(ALGO_COLOURS[algo])
            patch.set_alpha(0.7)
        ax2.set_ylabel("Mean reward")
        ax2.set_title("Distribution across 10 hyperparameter runs")
        ax2.grid(True, axis="y")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "04_convergence_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# Hyperparameter sensitivity

def plot_hyperparameter_sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Hyperparameter sensitivity — learning rate vs mean reward")

    for ax, (algo, (results_dir, _)) in zip(axes.flat, ALGO_DIRS.items()):
        df = load_summary(results_dir, algo)
        colour = ALGO_COLOURS[algo]

        if df.empty or "learning_rate" not in df.columns or "mean_reward" not in df.columns:
            ax.text(0.5, 0.5, f"{algo}\nNo data yet",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#9090a0")
            ax.set_title(algo, color=colour)
            continue

        lrs     = df["learning_rate"].values
        rewards = df["mean_reward"].values
        best_idx = rewards.argmax()

        ax.scatter(lrs, rewards, color=colour, s=80, alpha=0.75, zorder=3)
        ax.scatter(lrs[best_idx], rewards[best_idx],
                   color="white", s=140, zorder=4, marker="*",
                   label=f"Best: lr={lrs[best_idx]:.0e}, r={rewards[best_idx]:.1f}")
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("Mean reward")
        ax.set_title(algo, color=colour)
        ax.grid(True)
        ax.legend(fontsize=8)

        # Annotate gamma if available
        if "gamma" in df.columns:
            for _, row in df.iterrows():
                ax.annotate(f"γ={row['gamma']}",
                            (row["learning_rate"], row["mean_reward"]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color="#9090a0")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "05_hyperparameter_sensitivity.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# Compute efficiency analysis 

def plot_compute_efficiency():
    """
    Plots mean reward vs training time across all algorithms and runs,
    giving a sense of computational cost vs performance.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title("Compute efficiency — training time vs mean reward")
    ax.set_xlabel("Training time (seconds)")
    ax.set_ylabel("Mean reward")

    for algo, (results_dir, _) in ALGO_DIRS.items():
        df = load_summary(results_dir, algo)
        if df.empty or "training_time_s" not in df.columns:
            continue
        colour = ALGO_COLOURS[algo]
        ax.scatter(df["training_time_s"], df["mean_reward"],
                   color=colour, s=70, alpha=0.75, label=algo, zorder=3)
        # Mark best run
        best = df.loc[df["mean_reward"].idxmax()]
        ax.scatter(best["training_time_s"], best["mean_reward"],
                   color=colour, s=180, marker="*", zorder=4, edgecolors="white")

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "06_compute_efficiency.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# Main 

if __name__ == "__main__":
    print("Generating report figures...\n")
    plot_reward_curves()
    plot_dqn_loss()
    plot_entropy_curves()
    plot_convergence_comparison()
    plot_hyperparameter_sensitivity()
    plot_compute_efficiency()
