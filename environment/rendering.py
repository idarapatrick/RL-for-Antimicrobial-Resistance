"""
Pygame-based visualisation for the AMR Microscopy RL environment.
Displays: simulated microscopy view, agent decision panel,
resource gauges, anomaly timeline, and episode statistics.
"""

import numpy as np
import pygame
import sys
from typing import Optional

from environment.data_simulator import (
    CLASS_NAMES, CONTROL, DNA_DAMAGE, CELL_WALL, MEMBRANE
)
from environment.custom_env import ACTION_NAMES, COMPUTE_COST


# Colour palette
BG_DARK       = (18,  22,  36)
BG_PANEL      = (28,  34,  52)
BG_PANEL2     = (36,  44,  64)
ACCENT_TEAL   = (29, 185, 148)
ACCENT_PURPLE = (127, 119, 221)
ACCENT_AMBER  = (239, 159,  39)
ACCENT_CORAL  = (216,  90,  48)
ACCENT_GREEN  = (99,  153,  34)
TEXT_PRIMARY  = (220, 220, 228)
TEXT_MUTED    = (130, 130, 150)
TEXT_BRIGHT   = (255, 255, 255)
RED_ALERT     = (226,  75,  74)
GREEN_OK      = (99,  220, 140)

# Action colours
ACTION_COLOURS = {
    0: (80,  80,  100),   # SKIP - muted
    1: (60, 120, 180),    # QUICK_SCAN - blue
    2: ACCENT_TEAL,       # MORPHO_ANALYSIS
    3: ACCENT_AMBER,      # DEEP_ANALYSIS
    4: ACCENT_CORAL,      # ALERT_AND_DEEP
    5: ACCENT_PURPLE,     # TEMPORAL_COMPARE
}

# Antibiotic class colours
CLASS_COLOURS = {
    CONTROL:    ACCENT_GREEN,
    DNA_DAMAGE: ACCENT_PURPLE,
    CELL_WALL:  ACCENT_AMBER,
    MEMBRANE:   ACCENT_CORAL,
}

W, H = 1280, 780


class MicroscopyRenderer:
    """
    Pygame renderer for the AMR environment.
    Call render() each step; close() when done.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        if not pygame.get_init():
            pygame.init()

        if headless:
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            self.screen = pygame.Surface((W, H))
        else:
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("AMR Microscopy RL Agent")

        self.clock = pygame.time.Clock()
        self._init_fonts()

        # Rolling history for the timeline strip
        self._action_history  = []
        self._anomaly_history = []
        self._resist_history  = []

    def _init_fonts(self):
        pygame.font.init()
        self.font_lg  = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 14)
        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.font_xs  = pygame.font.SysFont("monospace", 10)

    # Public API

    def render(
        self,
        frame: int,
        total_frames: int,
        obs: np.ndarray,
        compute_budget: float,
        anomaly_history: np.ndarray,
        detection_confidence: float,
        episode_data,
        stats: dict,
        last_action: Optional[int] = None,
        last_reward: Optional[float] = None,
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # Store rolling history
        frame = int(frame)  # guard: ensure frame is int not a method object
        if last_action is not None:
            self._action_history.append(last_action)
            self._anomaly_history.append(float(obs[8]))
            resist_idx = max(0, frame - 1)
            resist_idx = min(resist_idx, len(episode_data.ground_truth_resistance) - 1)
            self._resist_history.append(
                bool(episode_data.ground_truth_resistance[resist_idx])
            )

        self.screen.fill(BG_DARK)

        # Layout regions
        # Left panel: microscopy simulation (0, 0, 520, 520)
        # Right panel: agent info (540, 0, 720, 520)
        # Bottom strip: timeline (0, 540, 1280, 240)

        self._draw_microscopy_panel(frame, obs, episode_data, pygame.Rect(20, 20, 500, 500))
        self._draw_agent_panel(frame, total_frames, obs, compute_budget,
                               detection_confidence, last_action, last_reward,
                               episode_data, pygame.Rect(540, 20, 720, 240))
        self._draw_stats_panel(stats, episode_data, pygame.Rect(540, 276, 720, 240))
        self._draw_timeline(frame, total_frames, episode_data,
                            pygame.Rect(20, 540, 1240, 220))

        if not self.headless:
            pygame.display.flip()
            self.clock.tick(30)

    def get_rgb_array(self) -> np.ndarray:
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self):
        pygame.quit()

    # Sub-panels

    def _draw_microscopy_panel(self, frame, obs, episode_data, rect):
        """Simulated microscopy view: animated bacteria representation."""
        pygame.draw.rect(self.screen, BG_PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_TEAL, rect, width=1, border_radius=10)

        # Title
        cls_name = episode_data.class_name if episode_data else "Unknown"
        cls_col  = CLASS_COLOURS.get(
            episode_data.antibiotic_class if episode_data else 0, TEXT_MUTED
        )
        self._text(f"Microscopy view — {cls_name}", rect.x + 10, rect.y + 8,
                   self.font_sm, cls_col)

        # Inner simulation area
        sim = pygame.Rect(rect.x + 10, rect.y + 30, rect.w - 20, rect.h - 80)
        pygame.draw.rect(self.screen, (8, 12, 22), sim)

        if episode_data is not None:
            morpho = episode_data.frame_features[min(frame, len(episode_data.frame_features)-1)]
            self._draw_bacteria(sim, morpho, episode_data.antibiotic_class, frame)

        # Frame progress bar
        bar_y = rect.bottom - 36
        pygame.draw.rect(self.screen, BG_PANEL2,
                         pygame.Rect(rect.x + 10, bar_y, rect.w - 20, 16), border_radius=4)
        progress = frame / max(1, (episode_data.total_frames if episode_data else 1000))
        pygame.draw.rect(self.screen, ACCENT_TEAL,
                         pygame.Rect(rect.x + 10, bar_y,
                                     int((rect.w - 20) * progress), 16), border_radius=4)
        self._text(f"Frame {frame} / {episode_data.total_frames if episode_data else '?'}",
                   rect.x + 10, bar_y + 20, self.font_xs, TEXT_MUTED)

        # Resistance event indicator
        if episode_data is not None and episode_data.ground_truth_resistance[min(frame, len(episode_data.ground_truth_resistance)-1)]:
            pygame.draw.rect(self.screen, RED_ALERT,
                             pygame.Rect(rect.right - 120, rect.y + 5, 110, 20), border_radius=4)
            self._text("RESISTANCE EVENT", rect.right - 117, rect.y + 8,
                       self.font_xs, TEXT_BRIGHT)

    def _draw_bacteria(self, sim_rect, morpho, ab_class, frame):
        """
        Draw stylised bacteria in the simulation area.
        Morphology visually reflects the current feature values.
        """
        rng = np.random.default_rng(frame // 5)  # slow-changing seed for stability
        n_cells = 40

        filament_idx  = float(morpho[2])
        rounding      = float(morpho[3])
        vesicle_score = float(morpho[4])
        membrane_int  = float(morpho[5])

        # Colour shifts with stress
        r = int(60  + 180 * (1 - membrane_int))
        g = int(180 * membrane_int)
        b = int(120 * (1 - filament_idx) + 60)
        cell_colour = (min(255, r), min(255, g), min(255, b))

        for _ in range(n_cells):
            cx = int(rng.integers(sim_rect.x + 10, sim_rect.right - 10))
            cy = int(rng.integers(sim_rect.y + 10, sim_rect.bottom - 10))

            # Base size
            w_base = int(rng.integers(3, 7))
            h_base = w_base

            # DNA-damage class: elongated filamentous cells
            if ab_class == DNA_DAMAGE and rng.random() < filament_idx:
                cell_w = w_base
                cell_h = int(w_base * (2.0 + filament_idx * 5.0))
            # Cell-wall class: bloated/rounded cells
            elif ab_class == CELL_WALL and rng.random() < rounding:
                cell_w = int(w_base * (1.5 + rounding))
                cell_h = int(w_base * (1.5 + rounding))
            else:
                cell_w = w_base
                cell_h = h_base

            pygame.draw.ellipse(self.screen, cell_colour,
                                pygame.Rect(cx - cell_w//2, cy - cell_h//2, cell_w, cell_h))

            # Draw vesicle blebs for membrane class
            if ab_class == MEMBRANE and rng.random() < vesicle_score * 0.4:
                bx = cx + int(rng.integers(-8, 8))
                by = cy + int(rng.integers(-8, 8))
                pygame.draw.circle(self.screen, ACCENT_AMBER, (bx, by), 2)

    def _draw_agent_panel(self, frame, total_frames, obs, compute_budget,
                          detection_confidence, last_action, last_reward,
                          episode_data, rect):
        pygame.draw.rect(self.screen, BG_PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_PURPLE, rect, width=1, border_radius=10)

        self._text("Agent state", rect.x + 10, rect.y + 8, self.font_sm, ACCENT_PURPLE)

        # Last action
        y = rect.y + 30
        if last_action is not None:
            col = ACTION_COLOURS.get(last_action, TEXT_MUTED)
            self._text(f"Action : {ACTION_NAMES.get(last_action, '?')} [{last_action}]",
                       rect.x + 12, y, self.font_md, col)
            cost_str = f"Cost: -{COMPUTE_COST.get(last_action, 0):.2f}%"
            self._text(cost_str, rect.x + 12, y + 18, self.font_xs, TEXT_MUTED)
        else:
            self._text("Action : initialising...", rect.x + 12, y, self.font_md, TEXT_MUTED)
        y += 44

        # Reward
        if last_reward is not None:
            rcol = GREEN_OK if last_reward >= 0 else RED_ALERT
            self._text(f"Reward : {last_reward:+.2f}", rect.x + 12, y, self.font_md, rcol)
        else:
            self._text("Reward : --", rect.x + 12, y, self.font_md, TEXT_MUTED)
        y += 28

        # Compute budget bar
        self._text("Compute budget", rect.x + 12, y, self.font_xs, TEXT_MUTED)
        y += 14
        bar_w = rect.w - 24
        pygame.draw.rect(self.screen, BG_PANEL2,
                         pygame.Rect(rect.x + 12, y, bar_w, 14), border_radius=4)
        budget_frac = max(0.0, compute_budget / 100.0)
        bcol = GREEN_OK if budget_frac > 0.4 else (ACCENT_AMBER if budget_frac > 0.2 else RED_ALERT)
        pygame.draw.rect(self.screen, bcol,
                         pygame.Rect(rect.x + 12, y, int(bar_w * budget_frac), 14), border_radius=4)
        self._text(f"{compute_budget:.1f}%", rect.x + 12, y + 16, self.font_xs, TEXT_MUTED)
        y += 34

        # Detection confidence bar
        self._text("Detection confidence", rect.x + 12, y, self.font_xs, TEXT_MUTED)
        y += 14
        pygame.draw.rect(self.screen, BG_PANEL2,
                         pygame.Rect(rect.x + 12, y, bar_w, 14), border_radius=4)
        dcol = ACCENT_TEAL if detection_confidence > 0.5 else TEXT_MUTED
        pygame.draw.rect(self.screen, dcol,
                         pygame.Rect(rect.x + 12, y, int(bar_w * detection_confidence), 14),
                         border_radius=4)
        self._text(f"{detection_confidence:.2f}", rect.x + 12, y + 16, self.font_xs, TEXT_MUTED)
        y += 34

        # Anomaly score (current frame)
        anomaly = float(obs[8])
        acol = (RED_ALERT if anomaly > 0.6 else
                ACCENT_AMBER if anomaly > 0.3 else GREEN_OK)
        self._text(f"Anomaly score : {anomaly:.3f}", rect.x + 12, y, self.font_md, acol)

    def _draw_stats_panel(self, stats, episode_data, rect):
        pygame.draw.rect(self.screen, BG_PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_TEAL, rect, width=1, border_radius=10)

        self._text("Episode statistics", rect.x + 10, rect.y + 8, self.font_sm, ACCENT_TEAL)

        y = rect.y + 30
        items = [
            (f"Total reward     : {stats.get('total_reward', 0):+.1f}", TEXT_PRIMARY),
            (f"Detections       : {stats.get('detections', 0)}", GREEN_OK),
            (f"Misses           : {stats.get('misses', 0)}", RED_ALERT),
            (f"False alarms     : {stats.get('false_alarms', 0)}", ACCENT_AMBER),
            (f"Critical misses  : {stats.get('critical_misses', 0)}", RED_ALERT),
        ]
        if episode_data:
            total_events = sum(
                1 for r in episode_data.resistance_events
            )
            items.append((f"Total events     : {total_events}", TEXT_MUTED))

        for text, colour in items:
            self._text(text, rect.x + 12, y, self.font_md, colour)
            y += 22

    def _draw_timeline(self, frame, total_frames, episode_data, rect):
        """Bottom strip: scrolling timeline of actions, anomaly scores, and resistance events."""
        pygame.draw.rect(self.screen, BG_PANEL, rect, border_radius=10)
        pygame.draw.rect(self.screen, ACCENT_TEAL, rect, width=1, border_radius=10)
        self._text("Timeline — actions | anomaly | resistance events",
                   rect.x + 10, rect.y + 6, self.font_xs, TEXT_MUTED)

        inner = pygame.Rect(rect.x + 10, rect.y + 22, rect.w - 20, rect.h - 30)
        n_ticks = inner.w
        ticks_per_frame = max(1, total_frames // n_ticks)

        # Draw ground truth resistance events as red background
        if episode_data is not None:
            for i in range(n_ticks):
                f_idx = i * ticks_per_frame
                if f_idx < total_frames and episode_data.ground_truth_resistance[f_idx]:
                    pygame.draw.rect(self.screen, (80, 20, 20),
                                     pygame.Rect(inner.x + i, inner.y, 1, inner.h))

        # Draw action history as coloured bars
        hist_len = len(self._action_history)
        if hist_len > 0:
            step = max(1, hist_len // n_ticks)
            for i in range(min(hist_len, n_ticks)):
                idx = int(i * hist_len / n_ticks)
                action = self._action_history[idx]
                col = ACTION_COLOURS.get(action, TEXT_MUTED)
                bar_h = int(inner.h * 0.4)
                pygame.draw.rect(self.screen, col,
                                 pygame.Rect(inner.x + i, inner.bottom - bar_h, 1, bar_h))

        # Draw anomaly score as a line
        if len(self._anomaly_history) > 1:
            n = len(self._anomaly_history)
            pts = []
            for i in range(n):
                x = inner.x + int(i * inner.w / max(1, n - 1))
                y = inner.y + int((1.0 - self._anomaly_history[i]) * inner.h * 0.55)
                pts.append((x, y))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, ACCENT_TEAL, False, pts, 1)

        # Draw current frame cursor
        cursor_x = inner.x + int(frame * inner.w / max(1, total_frames))
        pygame.draw.rect(self.screen, TEXT_BRIGHT,
                         pygame.Rect(cursor_x, inner.y, 1, inner.h))

        # Legend
        legend_items = [
            ("Actions", ACCENT_AMBER),
            ("Anomaly", ACCENT_TEAL),
            ("Resistance", RED_ALERT),
        ]
        lx = rect.x + 12
        for label, col in legend_items:
            pygame.draw.rect(self.screen, col, pygame.Rect(lx, rect.bottom - 16, 10, 8))
            self._text(label, lx + 14, rect.bottom - 16, self.font_xs, TEXT_MUTED)
            lx += 90

    # Utility

    def _text(self, msg, x, y, font, colour):
        surf = font.render(str(msg), True, colour)
        self.screen.blit(surf, (x, y))