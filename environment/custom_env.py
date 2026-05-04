"""
Custom Gymnasium environment: AMR Microscopy Adaptive Analysis.

The Reinforcement Learning agent controls a microscopy analysis pipeline monitoring E. coli cultures
under antibiotic stress. It must allocate limited computational resources to detect
resistance-related morphological changes while balancing:
  - Detection accuracy  (catch resistance events)
  - Computational cost  (conserve processing budget)
  - False alarm rate    (avoid unnecessary deep-analysis alerts)

State/observation space: biologically grounded morphological features derived from
the DeepBacs E. coli antibiotic phenotyping dataset (Zenodo: 10.5281/zenodo.5551057)

Action space: 6 discrete analysis depth levels (SKIP -> ALERT_AND_DEEP)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from environment.data_simulator import AMRDataSimulator, FEATURE_NAMES


# Action definitions 
ACTION_SKIP             = 0   # No processing, saves budget
ACTION_QUICK_SCAN       = 1   # Density + motility check only (low cost)
ACTION_MORPHO_ANALYSIS  = 2   # Full morphological feature extraction (medium cost)
ACTION_DEEP_ANALYSIS    = 3   # Morphology + anomaly pipeline (high cost)
ACTION_ALERT_AND_DEEP   = 4   # Deep analysis + flag for human review (highest cost)
ACTION_TEMPORAL_COMPARE = 5   # Compare with previous N frames (medium-high cost)

ACTION_NAMES = {
    0: "SKIP",
    1: "QUICK_SCAN",
    2: "MORPHO_ANALYSIS",
    3: "DEEP_ANALYSIS",
    4: "ALERT_AND_DEEP",
    5: "TEMPORAL_COMPARE",
}

# Computational cost per action (as % of total budget consumed per step)
# Scaled so that a mixed-usage policy comfortably completes 1000 frames.
# Pure DEEP_ANALYSIS every frame depletes budget around step 550,
# realistic mixed usage (skip + selective deep) leaves ~30% at episode end.
COMPUTE_COST = {
    ACTION_SKIP:             0.02,
    ACTION_QUICK_SCAN:       0.10,
    ACTION_MORPHO_ANALYSIS:  0.25,
    ACTION_DEEP_ANALYSIS:    0.18,
    ACTION_ALERT_AND_DEEP:   0.22,
    ACTION_TEMPORAL_COMPARE: 0.15,
}

# Which actions can detect resistance events
DETECTION_ACTIONS = {ACTION_DEEP_ANALYSIS, ACTION_ALERT_AND_DEEP, ACTION_TEMPORAL_COMPARE}
ANALYSIS_ACTIONS  = {ACTION_MORPHO_ANALYSIS, ACTION_DEEP_ANALYSIS,
                     ACTION_ALERT_AND_DEEP, ACTION_TEMPORAL_COMPARE}


class MicroscopyAMREnv(gym.Env):
    """
    Adaptive microscopy resource allocation environment for AMR detection.

    Observation space (flat Box, shape=(23,)):
      [0:8]   morphological_features  - cell_length_ratio, cell_width_ratio,
                                        filamentation_index, cell_rounding_score,
                                        vesicle_score, membrane_integrity,
                                        nucleoid_compaction, colony_density
      [8]     anomaly_score           - weighted composite of stress indicators
      [9]     frames_since_last_alert - normalised to [0,1]
      [10]    compute_budget          - normalised to [0,1]
      [11:21] recent_anomaly_history  - last 10 anomaly scores
      [21]    colony_density          - duplicate for explicit access
      [22]    resistance_event_active - ground truth leak? No: this is hidden.
                                        Instead: detection_confidence from last action

    Note: observation is a flat float32 vector (required by SB3 DQN/PPO/A2C).
    The Dict space is preserved internally for rendering / logging.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Observation vector length
    OBS_DIM = 22

    def __init__(
        self,
        render_mode: Optional[str] = None,
        total_frames: int = 1000,
        seed: Optional[int] = None,
        max_critical_misses: int = 3,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.total_frames = total_frames
        self.max_critical_misses = max_critical_misses
        self._seed = seed

        # Action space: 6 discrete analysis actions
        self.action_space = spaces.Discrete(6)

        # Observation space: flat float32 vector of length OBS_DIM
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        # Internal simulator, generates episode data
        self._simulator = AMRDataSimulator(seed=seed, total_frames=total_frames)

        # Episode state, initialised in reset()
        self._episode_data = None
        self._current_frame = 0
        self._compute_budget = 100.0
        self._frames_since_last_alert = 0
        self._critical_misses = 0
        self._recent_anomaly_history = np.zeros(10, dtype=np.float32)
        self._detection_confidence = 0.0
        self._total_reward = 0.0
        self._episode_detections = 0
        self._episode_misses = 0
        self._episode_false_alarms = 0

        # Rendering
        self._renderer = None

    # Core Gymnasium API 

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Generate a fresh episode from the simulator
        ep_seed = seed if seed is not None else self._seed
        self._simulator = AMRDataSimulator(
            seed=ep_seed, total_frames=self.total_frames
        )
        self._episode_data = self._simulator.generate_episode()

        # Reset episode state
        self._current_frame = 0
        self._compute_budget = 100.0
        self._frames_since_last_alert = 0
        self._critical_misses = 0
        self._recent_anomaly_history = np.zeros(10, dtype=np.float32)
        self._detection_confidence = 0.0
        self._total_reward = 0.0
        self._episode_detections = 0
        self._episode_misses = 0
        self._episode_false_alarms = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        frame = self._current_frame
        is_resistance = bool(self._episode_data.ground_truth_resistance[frame])

        # Compute reward
        reward = self._calculate_reward(action, is_resistance)
        self._total_reward += reward

        # Update detection confidence based on action depth
        self._detection_confidence = self._estimate_detection_confidence(action, frame)

        # Update anomaly history
        current_anomaly = self._compute_anomaly_score(frame)
        self._recent_anomaly_history = np.roll(self._recent_anomaly_history, -1)
        self._recent_anomaly_history[-1] = current_anomaly

        # Deduct compute budget
        self._compute_budget = max(
            0.0, self._compute_budget - COMPUTE_COST[action]
        )

        # Track alert timing
        if action == ACTION_ALERT_AND_DEEP:
            self._frames_since_last_alert = 0
        else:
            self._frames_since_last_alert = min(
                100, self._frames_since_last_alert + 1
            )

        # Track miss statistics
        if is_resistance and action not in DETECTION_ACTIONS:
            self._episode_misses += 1
            if current_anomaly > 0.5:
                self._critical_misses += 1
        if is_resistance and action in DETECTION_ACTIONS:
            self._episode_detections += 1
        if action == ACTION_ALERT_AND_DEEP and not is_resistance:
            self._episode_false_alarms += 1

        # Advance frame
        self._current_frame += 1

        # Check termination
        terminated = self._is_terminated()
        truncated = False

        obs = self._get_observation()
        info = self._get_info()

        # Note: rendering is handled externally by main.py which has access
        # to last_action and last_reward. Internal render() call is disabled
        # to prevent a second renderer being created without action context.

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._renderer is None:
                from environment.rendering import MicroscopyRenderer
                self._renderer = MicroscopyRenderer()
            self._renderer.render(
                frame=self._current_frame,
                total_frames=self.total_frames,
                obs=self._get_observation(),
                compute_budget=self._compute_budget,
                anomaly_history=self._recent_anomaly_history,
                detection_confidence=self._detection_confidence,
                episode_data=self._episode_data,
                stats={
                    "detections": self._episode_detections,
                    "misses": self._episode_misses,
                    "false_alarms": self._episode_false_alarms,
                    "total_reward": self._total_reward,
                    "critical_misses": self._critical_misses,
                },
            )
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                from environment.rendering import MicroscopyRenderer
                self._renderer = MicroscopyRenderer(headless=True)
            return self._renderer.get_rgb_array()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # Reward calculation 

    def _calculate_reward(self, action: int, is_resistance: bool) -> float:
        reward = 0.0
        frame = self._current_frame
        current_anomaly = self._compute_anomaly_score(frame)

        # Detection reward (catch resistance events)
        if is_resistance:
            if action in DETECTION_ACTIONS:
                # Scale reward by action depth: deeper analysis - more confident detection
                depth_bonus = {
                    ACTION_DEEP_ANALYSIS:    10.0,
                    ACTION_ALERT_AND_DEEP:   12.0,
                    ACTION_TEMPORAL_COMPARE: 8.0,
                }
                reward += depth_bonus[action]
                reward += current_anomaly * 3.0   # bonus for catching high-severity events
            else:
                # Missed a resistance event (penalty scales with severity)
                base_penalty = -15.0
                severity_scale = 1.0 + current_anomaly
                reward += base_penalty * severity_scale

        # Computational efficiency (penalise expensive actions)
        cost_penalty = COMPUTE_COST[action] * 0.5
        reward -= cost_penalty

        # False alarm penalty
        if action == ACTION_ALERT_AND_DEEP and not is_resistance:
            reward -= 5.0

        # Budget guard (penalise expensive actions when budget is low)
        if self._compute_budget < 10.0 and action in {
            ACTION_DEEP_ANALYSIS, ACTION_ALERT_AND_DEEP, ACTION_TEMPORAL_COMPARE
        }:
            reward -= 1.5

        # Temporal efficiency bonus (reward catching events after a quiet period)
        if action in DETECTION_ACTIONS and is_resistance:
            if self._frames_since_last_alert > 50:
                reward += 2.0

        # Smart skip reward (reward skipping genuinely quiet frames)
        if action == ACTION_SKIP and not is_resistance and current_anomaly < 0.15:
            reward += 0.5

        # Incremental quality bonus for analysis actions
        if action in ANALYSIS_ACTIONS:
            quality = self._estimate_detection_confidence(action, frame)
            reward += quality * 1.5

        return float(reward)

    # Termination 

    def _is_terminated(self) -> bool:
        if self._current_frame >= self.total_frames:
            return True
        if self._compute_budget <= 0.0:
            return True
        if self._critical_misses >= self.max_critical_misses:
            return True
        return False

    # Observation construction

    def _get_observation(self) -> np.ndarray:
        """Build flat float32 observation vector of length OBS_DIM=22."""
        frame = min(self._current_frame, self.total_frames - 1)
        morpho = self._episode_data.frame_features[frame]  # (8,)

        obs = np.concatenate([
            morpho,                                                       # [0:8]
            [self._compute_anomaly_score(frame)],                         # [8]
            [self._frames_since_last_alert / 100.0],                      # [9]
            [self._compute_budget / 100.0],                               # [10]
            self._recent_anomaly_history,                                 # [11:21]
            [self._detection_confidence],                                 # [21]
        ]).astype(np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _compute_anomaly_score(self, frame: int) -> float:
        """
        Composite anomaly score from the morphological stress indicators.
        Weighted average of filamentation, rounding, vesicle, and membrane features.
        """
        morpho = self._episode_data.frame_features[frame]
        # Feature indices: filamentation=2, rounding=3, vesicle=4, membrane_integrity=5
        filament  = float(morpho[2])
        rounding  = float(morpho[3])
        vesicle   = float(morpho[4])
        membrane  = 1.0 - float(morpho[5])   # invert: low integrity = high anomaly
        nucleoid  = float(morpho[6])
        score = (
            0.30 * filament +
            0.20 * rounding +
            0.25 * vesicle  +
            0.15 * membrane +
            0.10 * nucleoid
        )
        return float(np.clip(score, 0.0, 1.0))

    def _estimate_detection_confidence(self, action: int, frame: int) -> float:
        """
        Simulate detection confidence as a function of action depth and anomaly level.
        Deeper actions produce higher confidence, especially on high-anomaly frames.
        """
        anomaly = self._compute_anomaly_score(frame)
        base_conf = {
            ACTION_SKIP:             0.0,
            ACTION_QUICK_SCAN:       0.2,
            ACTION_MORPHO_ANALYSIS:  0.45,
            ACTION_DEEP_ANALYSIS:    0.70,
            ACTION_ALERT_AND_DEEP:   0.85,
            ACTION_TEMPORAL_COMPARE: 0.60,
        }
        conf = base_conf[action] + anomaly * 0.3
        noise = float(np.random.normal(0, 0.03))
        return float(np.clip(conf + noise, 0.0, 1.0))

    def _get_info(self) -> dict:
        return {
            "frame": self._current_frame,
            "compute_budget": self._compute_budget,
            "critical_misses": self._critical_misses,
            "episode_detections": self._episode_detections,
            "episode_misses": self._episode_misses,
            "episode_false_alarms": self._episode_false_alarms,
            "total_reward": self._total_reward,
            "antibiotic_class": self._episode_data.antibiotic_class
            if self._episode_data else None,
        }

    # Properties for external access

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def compute_budget(self) -> float:
        return self._compute_budget

    @property
    def episode_data(self):
        return self._episode_data

    @property
    def action_names(self) -> dict:
        return ACTION_NAMES.copy()