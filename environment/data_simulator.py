"""
Generates synthetic per-frame morphological feature vectors for RL episodes,
calibrated from:
  - DeepBacs E. coli antibiotic phenotyping dataset
    (Spahn & Heilemann, 2021. Zenodo DOI: 10.5281/zenodo.5551057)
  - Cushnie et al. (2020). Effects of antibiotics on bacterial cell morphology. PMC10152891
  - Cama et al. (2023). Deep learning + single-cell phenotyping. Nature Commun. Biol.

Each episode samples one antibiotic class and generates a 1000-frame time series
that mimics morphological changes during antibiotic stress exposure. Resistance
events are stochastically placed based on realistic biological timelines.

Antibiotic classes and their morphological signatures:
  CONTROL      - Normal rod-shaped E. coli, no treatment
  DNA_DAMAGE   - Ciprofloxacin / Nalidixate: SOS response, filamentation (cell_length_ratio 2.5-4.0)
  CELL_WALL    - Mecillinam / MP265: PBP2 inhibition, cell rounding/bloating (cell_width_ratio 1.5-2.5)
  MEMBRANE     - Rifampicin / CAM: nucleoid compaction, vesicle formation, membrane disruption
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# Antibiotic class identifiers
CONTROL    = 0
DNA_DAMAGE = 1   # Ciprofloxacin, Nalidixate — causes filamentation via SOS
CELL_WALL  = 2   # Mecillinam, MP265 — causes cell bloating / rounding
MEMBRANE   = 3   # Rifampicin, CAM — causes vesicles, nucleoid compaction

CLASS_NAMES = {
    CONTROL:    "Control (untreated)",
    DNA_DAMAGE: "DNA-damage (Cipro/Nalidixate)",
    CELL_WALL:  "Cell-wall (Mecillinam/MP265)",
    MEMBRANE:   "Membrane (Rifampicin/CAM)",
}

# Per-class morphological parameter distributions
# Values derived from DeepBacs dataset statistics and cited literature.
# Format: (baseline_mean, baseline_std, stressed_mean, stressed_std)
# "stressed" = values observed in antibiotic-treated populations.

MORPHOLOGY_PARAMS = {
    # Feature: cell_length_ratio (treated / untreated length)
    "cell_length_ratio": {
        CONTROL:    (1.00, 0.08, 1.00, 0.08),
        DNA_DAMAGE: (1.00, 0.10, 3.20, 0.60),   # filamentation: 4:1 ratio post-exposure (Cushnie 2020)
        CELL_WALL:  (1.00, 0.08, 1.05, 0.10),   # length mostly unchanged
        MEMBRANE:   (1.00, 0.08, 1.15, 0.20),
    },
    # Feature: cell_width_ratio (treated / untreated width)
    "cell_width_ratio": {
        CONTROL:    (1.00, 0.06, 1.00, 0.06),
        DNA_DAMAGE: (1.00, 0.06, 1.02, 0.08),   # width constant during filamentation
        CELL_WALL:  (1.00, 0.07, 1.90, 0.40),   # bloating: PBP2 inhibition causes rounding
        MEMBRANE:   (1.00, 0.06, 1.10, 0.15),
    },
    # Feature: filamentation_index (fraction of filamentous cells in field of view)
    "filamentation_index": {
        CONTROL:    (0.04, 0.02, 0.04, 0.02),
        DNA_DAMAGE: (0.04, 0.02, 0.72, 0.15),   # ~80% filamentous post Cipro exposure
        CELL_WALL:  (0.04, 0.02, 0.08, 0.04),
        MEMBRANE:   (0.04, 0.02, 0.12, 0.06),
    },
    # Feature: cell_rounding_score (0=rod-shaped, 1=fully round/spheroplast)
    "cell_rounding_score": {
        CONTROL:    (0.05, 0.03, 0.05, 0.03),
        DNA_DAMAGE: (0.05, 0.03, 0.08, 0.04),
        CELL_WALL:  (0.05, 0.03, 0.65, 0.18),   # PBP2 inhibition → spheroplast formation
        MEMBRANE:   (0.05, 0.03, 0.20, 0.08),
    },
    # Feature: vesicle_score (membrane blebbing intensity, 0–1)
    "vesicle_score": {
        CONTROL:    (0.03, 0.02, 0.03, 0.02),
        DNA_DAMAGE: (0.03, 0.02, 0.15, 0.06),   # some vesicles in oblique class
        CELL_WALL:  (0.03, 0.02, 0.10, 0.05),
        MEMBRANE:   (0.03, 0.02, 0.58, 0.18),   # Rifampicin: clear vesicle class in DeepBacs
    },
    # Feature: membrane_integrity (1=intact, 0=fully disrupted)
    "membrane_integrity": {
        CONTROL:    (0.96, 0.03, 0.96, 0.03),
        DNA_DAMAGE: (0.96, 0.03, 0.55, 0.15),   # DiBAC uptake ~68% after Cipro (Cushnie 2020)
        CELL_WALL:  (0.96, 0.03, 0.70, 0.12),
        MEMBRANE:   (0.96, 0.03, 0.38, 0.18),   # most disrupted: membrane-targeting drug
    },
    # Feature: nucleoid_compaction (0=diffuse normal, 1=highly compacted)
    "nucleoid_compaction": {
        CONTROL:    (0.10, 0.04, 0.10, 0.04),
        DNA_DAMAGE: (0.10, 0.04, 0.35, 0.12),   # DNA damage disperses nucleoid
        CELL_WALL:  (0.10, 0.04, 0.15, 0.06),
        MEMBRANE:   (0.10, 0.04, 0.70, 0.14),   # CAM/Rifampicin: centralised DNA region
    },
    # Feature: colony_density (relative cell density in frame, 0–1)
    "colony_density": {
        CONTROL:    (0.50, 0.15, 0.50, 0.15),
        DNA_DAMAGE: (0.50, 0.15, 0.45, 0.18),   # slight reduction due to growth inhibition
        CELL_WALL:  (0.50, 0.15, 0.40, 0.16),
        MEMBRANE:   (0.50, 0.15, 0.38, 0.14),
    },
}

FEATURE_NAMES = list(MORPHOLOGY_PARAMS.keys())  # 8 morphological features


@dataclass
class ResistanceEvent:
    """A single resistance event in an episode timeline."""
    frame: int
    intensity: float      # 0.0 – 1.0, scales how abnormal the morphology appears
    antibiotic_class: int
    duration: int         # frames the event persists


@dataclass
class EpisodeData:
    """All data needed to run one RL episode."""
    antibiotic_class: int
    class_name: str
    total_frames: int
    resistance_events: List[ResistanceEvent]
    # Pre-generated feature array: shape (total_frames, n_features)
    frame_features: np.ndarray
    # Ground truth: which frames contain a resistance event
    ground_truth_resistance: np.ndarray   # bool array, shape (total_frames,)


class AMRDataSimulator:
    """
    Generates RL episode data calibrated from the DeepBacs antibiotic
    phenotyping dataset and published AMR morphology literature.

    Usage:
        sim = AMRDataSimulator(seed=42)
        episode = sim.generate_episode()
        features_at_frame_5 = episode.frame_features[5]  # shape (8,)
        is_resistance = episode.ground_truth_resistance[5]  # bool
    """

    def __init__(self, seed: Optional[int] = None, total_frames: int = 1000):
        self.rng = np.random.default_rng(seed)
        self.total_frames = total_frames
        self.n_features = len(FEATURE_NAMES)

    def generate_episode(self, antibiotic_class: Optional[int] = None) -> EpisodeData:
        """
        Generate one complete episode.
        If antibiotic_class is None, randomly sample one of the four classes.
        """
        if antibiotic_class is None:
            antibiotic_class = int(self.rng.integers(0, 4))

        resistance_events = self._place_resistance_events(antibiotic_class)
        ground_truth = self._build_ground_truth(resistance_events)
        frame_features = self._generate_frame_features(antibiotic_class, resistance_events)

        return EpisodeData(
            antibiotic_class=antibiotic_class,
            class_name=CLASS_NAMES[antibiotic_class],
            total_frames=self.total_frames,
            resistance_events=resistance_events,
            frame_features=frame_features,
            ground_truth_resistance=ground_truth,
        )

    def _place_resistance_events(self, antibiotic_class: int) -> List[ResistanceEvent]:
        """
        Place resistance events stochastically.
        Control class has rare spurious events; treated classes have more frequent events.
        Timing modelled on realistic AMR development timescales.
        """
        if antibiotic_class == CONTROL:
            # Very few spurious detections in untreated cultures
            n_events = int(self.rng.integers(0, 2))
            event_frames = self.rng.integers(100, 900, size=n_events).tolist()
            intensities = self.rng.uniform(0.1, 0.3, size=n_events).tolist()
            durations = self.rng.integers(5, 20, size=n_events).tolist()
        else:
            # Treated cultures: 4–7 resistance events per episode
            # Events tend to cluster in the middle/late episode as stress accumulates
            n_events = int(self.rng.integers(4, 8))
            # Sample from a distribution weighted toward frames 200–900
            weights = np.concatenate([
                np.linspace(0.1, 1.0, 400),   # frames 0-399: rising probability
                np.ones(400) * 1.0,            # frames 400-799: plateau
                np.linspace(1.0, 0.6, 200),    # frames 800-999: slight decline
            ])
            weights /= weights.sum()
            event_frames = self.rng.choice(
                self.total_frames, size=n_events, replace=False, p=weights
            ).tolist()
            event_frames.sort()
            intensities = self.rng.uniform(0.4, 1.0, size=n_events).tolist()
            durations = self.rng.integers(15, 60, size=n_events).tolist()

        events = []
        for frame, intensity, duration in zip(event_frames, intensities, durations):
            events.append(ResistanceEvent(
                frame=int(frame),
                intensity=float(intensity),
                antibiotic_class=antibiotic_class,
                duration=int(duration),
            ))
        return events

    def _build_ground_truth(self, events: List[ResistanceEvent]) -> np.ndarray:
        """Boolean array: True at frames where a resistance event is active."""
        gt = np.zeros(self.total_frames, dtype=bool)
        for event in events:
            start = event.frame
            end = min(event.frame + event.duration, self.total_frames)
            gt[start:end] = True
        return gt

    def _generate_frame_features(
        self,
        antibiotic_class: int,
        resistance_events: List[ResistanceEvent],
    ) -> np.ndarray:
        """
        Generate per-frame feature vectors.
        - Baseline: sampled from the class's baseline distribution
        - Near resistance events: interpolate toward stressed distribution
        - Add temporal autocorrelation (features don't jump instantaneously)
        """
        features = np.zeros((self.total_frames, self.n_features), dtype=np.float32)

        # Build a "stress level" curve: how stressed the culture is at each frame
        stress_level = np.zeros(self.total_frames, dtype=np.float32)
        for event in resistance_events:
            for t in range(self.total_frames):
                dist = abs(t - event.frame)
                if dist < event.duration:
                    # Gaussian-shaped stress peak around the event frame
                    sigma = event.duration / 3.0
                    contribution = event.intensity * np.exp(-0.5 * (dist / sigma) ** 2)
                    stress_level[t] = min(1.0, stress_level[t] + contribution)

        for i, feature_name in enumerate(FEATURE_NAMES):
            params = MORPHOLOGY_PARAMS[feature_name][antibiotic_class]
            base_mean, base_std, stress_mean, stress_std = params

            # Generate baseline trajectory with temporal smoothing
            raw = self.rng.normal(base_mean, base_std, self.total_frames).astype(np.float32)

            # Blend toward stressed values based on local stress level
            for t in range(self.total_frames):
                s = stress_level[t]
                mean_t = (1 - s) * base_mean + s * stress_mean
                std_t  = (1 - s) * base_std  + s * stress_std
                raw[t] = float(self.rng.normal(mean_t, std_t))

            # Apply exponential smoothing for temporal continuity
            smoothed = np.zeros_like(raw)
            alpha = 0.3  # smoothing factor
            smoothed[0] = raw[0]
            for t in range(1, self.total_frames):
                smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t - 1]

            # Clip to biologically plausible range [0, 1] for ratio/score features
            features[:, i] = np.clip(smoothed, 0.0, 1.0)

        return features

    def get_frame_observation(
        self,
        episode: EpisodeData,
        frame: int,
        frames_since_alert: int,
        compute_budget: float,
        recent_anomaly_history: np.ndarray,
    ) -> dict:
        """
        Build the full observation dict for a single frame, combining morphological
        features with resource state and temporal context.

        Returns a dict matching the gymnasium observation space in custom_env.py.
        """
        morpho = episode.frame_features[frame]          # shape (8,)
        anomaly_score = float(np.mean(morpho[[2, 3, 4, 5]]))  # filament+round+vesicle+membrane

        return {
            "morphological_features": morpho,           # (8,) float32
            "anomaly_score":          np.array([anomaly_score], dtype=np.float32),
            "frames_since_last_alert": np.array([frames_since_alert], dtype=np.int32),
            "compute_budget":         np.array([compute_budget], dtype=np.float32),
            "recent_anomaly_history": recent_anomaly_history.astype(np.float32),  # (10,)
            "colony_density":         np.array([morpho[7]], dtype=np.float32),
        }

    @property
    def feature_names(self) -> List[str]:
        return FEATURE_NAMES.copy()

    @property
    def antibiotic_classes(self) -> dict:
        return CLASS_NAMES.copy()


if __name__ == "__main__":
    # For a quick sanity check, this file is run directly to verify simulator output
    sim = AMRDataSimulator(seed=42)
    for cls in [CONTROL, DNA_DAMAGE, CELL_WALL, MEMBRANE]:
        ep = sim.generate_episode(antibiotic_class=cls)
        n_events = len(ep.resistance_events)
        n_resist_frames = ep.ground_truth_resistance.sum()
        mean_features = ep.frame_features.mean(axis=0)
        print(f"\n{ep.class_name}")
        print(f"  Resistance events : {n_events}")
        print(f"  Resistance frames : {n_resist_frames} / {ep.total_frames}")
        print(f"  Mean features     : {dict(zip(FEATURE_NAMES, mean_features.round(3)))}")