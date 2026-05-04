"""
Serves the best trained RL model as a REST API endpoint.
This demonstrates how the trained agent can be serialised and
integrated into any frontend or production imaging pipeline.

The client sends a JSON observation vector and receives an action
decision back as JSON.

Usage:
    python api_server.py                  # starts on http://localhost:8000
    python api_server.py --port 8080      # custom port

Example request (curl):
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"observation": [0.98, 0.99, 0.04, 0.05, 0.03, 0.96, 0.10, 0.50,
                              0.04, 0.01, 0.85, 0.04, 0.04, 0.04, 0.04, 0.04,
                              0.04, 0.04, 0.04, 0.04, 0.04, 0.06]}'

Example response:
    {
      "action": 0,
      "action_name": "SKIP",
      "confidence": 0.91,
      "anomaly_score": 0.04,
      "recommendation": "No resistance detected. Frame skipped to conserve budget.",
      "compute_cost_percent": 0.02
    }
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn

from environment.custom_env import ACTION_NAMES, COMPUTE_COST

app = FastAPI(
    title="AMR Reinforcement Learning Agent API",
    description=(
        "Serves a trained reinforcement learning model that decides the optimal "
        "analysis depth for each frame in an AMR microscopy pipeline. "
        "Send an observation vector, receive an action decision."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loaded once at startup 

MODEL = None
ALGO  = None

RECOMMENDATIONS = {
    0: "No resistance signal detected. Frame skipped to conserve compute budget.",
    1: "Low anomaly detected. Quick density and motility check performed.",
    2: "Moderate anomaly detected. Full morphological feature extraction performed.",
    3: "Elevated anomaly detected. Deep analysis pipeline triggered. Resistance check active.",
    4: "High-confidence resistance signal. Deep analysis complete. Frame flagged for human review.",
    5: "Anomaly trend detected across recent frames. Temporal comparison performed. Resistance check active.",
}


def load_model():
    global MODEL, ALGO

    candidates = []
    for algo, path in [
        ("ppo",       "models/pg/ppo/best_run.json"),
        ("dqn",       "models/dqn/best_run.json"),
        ("a2c",       "models/pg/a2c/best_run.json"),
        ("reinforce", "models/pg/reinforce/best_run.json"),
    ]:
        if os.path.exists(path):
            with open(path) as f:
                info = json.load(f)
            candidates.append((algo, info["mean_reward"], info["run_name"]))

    if not candidates:
        raise RuntimeError(
            "No trained models found. Run training first: "
            "python training/dqn_training.py && python training/pg_training.py --algo all"
        )

    best = max(candidates, key=lambda x: x[1])
    algo, mean_reward, run_name = best
    print(f"Loading best model: {algo.upper()} — {run_name} (mean_reward={mean_reward:.3f})")

    if algo == "ppo":
        from stable_baselines3 import PPO
        MODEL = PPO.load(os.path.join("models/pg/ppo", run_name, "final_model"))
    elif algo == "dqn":
        from stable_baselines3 import DQN
        MODEL = DQN.load(os.path.join("models/dqn", run_name, "final_model"))
    elif algo == "a2c":
        from stable_baselines3 import A2C
        MODEL = A2C.load(os.path.join("models/pg/a2c", run_name, "final_model"))
    elif algo == "reinforce":
        import torch
        from training.pg_training import PolicyNetwork
        from environment.custom_env import MicroscopyAMREnv
        obs_dim   = MicroscopyAMREnv().observation_space.shape[0]
        n_actions = MicroscopyAMREnv().action_space.n
        policy = PolicyNetwork(obs_dim, n_actions)
        weights_path = os.path.join("models/pg/reinforce", run_name, "policy.pt")
        policy.load_state_dict(torch.load(weights_path, map_location="cpu"))
        policy.eval()
        MODEL = policy

    ALGO = algo
    print(f"Model ready. API accepting requests.")


# Request / Response schemas 

class ObservationRequest(BaseModel):
    observation: List[float] = Field(
        ...,
        min_length=22,
        max_length=22,
        description=(
            "22-dimensional observation vector in order: "
            "cell_length_ratio, cell_width_ratio, filamentation_index, "
            "cell_rounding_score, vesicle_score, membrane_integrity, "
            "nucleoid_compaction, colony_density, anomaly_score, "
            "frames_since_last_alert, compute_budget, "
            "recent_anomaly_history[0..9], detection_confidence. "
            "All values must be in range [0, 1]."
        ),
        example=[
            0.98, 0.99, 0.04, 0.05, 0.03, 0.96, 0.10, 0.50,
            0.04, 0.01, 0.85, 0.04, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04, 0.06
        ]
    )


class PredictionResponse(BaseModel):
    action: int
    action_name: str
    confidence: float
    anomaly_score: float
    recommendation: str
    compute_cost_percent: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    algorithm: str
    observation_dims: int
    action_space: dict


# Endpoints 

@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health", response_model=HealthResponse, summary="Check server and model status")
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        algorithm=ALGO.upper() if ALGO else "none",
        observation_dims=22,
        action_space={str(k): v for k, v in ACTION_NAMES.items()},
    )


@app.post("/predict", response_model=PredictionResponse, summary="Get action decision for one frame")
def predict(request: ObservationRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    obs = np.array(request.observation, dtype=np.float32)

    if obs.min() < 0.0 or obs.max() > 1.0:
        raise HTTPException(
            status_code=422,
            detail=f"All observation values must be in [0, 1]. "
                   f"Got min={obs.min():.4f}, max={obs.max():.4f}."
        )

    # Get action from model
    if ALGO in ("ppo", "dqn", "a2c"):
        action, _ = MODEL.predict(obs, deterministic=True)
        action = int(action)
        # Estimate confidence from action probabilities where available
        try:
            import torch
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            if hasattr(MODEL, "policy"):
                with torch.no_grad():
                    dist = MODEL.policy.get_distribution(obs_tensor)
                    probs = dist.distribution.probs.squeeze().numpy()
                    confidence = float(probs[action])
            else:
                confidence = 0.0
        except Exception:
            confidence = 0.0

    elif ALGO == "reinforce":
        import torch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = MODEL(obs_tensor).squeeze().numpy()
        action = int(probs.argmax())
        confidence = float(probs[action])

    else:
        raise HTTPException(status_code=500, detail="Unknown algorithm.")

    anomaly_score = float(obs[8])

    return PredictionResponse(
        action=action,
        action_name=ACTION_NAMES[action],
        confidence=round(confidence, 4),
        anomaly_score=round(anomaly_score, 4),
        recommendation=RECOMMENDATIONS[action],
        compute_cost_percent=COMPUTE_COST[action],
    )


@app.get("/actions", summary="List all available actions and their costs")
def list_actions():
    return {
        str(action_id): {
            "name": name,
            "compute_cost_percent": COMPUTE_COST[action_id],
            "detects_resistance": action_id in {3, 4, 5},
        }
        for action_id, name in ACTION_NAMES.items()
    }


# Entry point 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting AMR RL API server on http://{args.host}:{args.port}")
    print(f"API docs available at http://localhost:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)