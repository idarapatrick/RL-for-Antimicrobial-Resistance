"""
Demonstrates the environment visualization with a RANDOM agent.
No model is loaded, and the agent takes random actions every step.

Run:
    python demo_random_agent.py
    python demo_random_agent.py --frames 200   # run for 200 steps then exit
    python demo_random_agent.py --no-display   # headless, saves screenshot
"""

import pygame
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import MicroscopyAMREnv, ACTION_NAMES
from environment.rendering import MicroscopyRenderer
from environment.data_simulator import CLASS_NAMES


def run_random_demo(max_frames: int = 500, headless: bool = False):
    print("=" * 60)
    print("  AMR Microscopy RL - Random Agent Demo")
    print("  No trained model - purely random action selection")
    print("=" * 60)

    env = MicroscopyAMREnv(render_mode=None, seed=42)
    renderer = MicroscopyRenderer(headless=headless)

    obs, info = env.reset(seed=42)

    print(f"\nEpisode antibiotic class: {env.episode_data.class_name}")
    print(f"Total frames in episode:  {env.episode_data.total_frames}")
    print(f"Resistance events:        {len(env.episode_data.resistance_events)}")
    print(f"Observation shape:        {obs.shape}")
    print(f"Action space:             {env.action_space}")
    print("\nAction in Progress (close the window or press Ctrl+C to stop)\n")

    total_reward = 0.0
    step = 0
    action_counts = {i: 0 for i in range(6)}
    
    # Initialize clock to control simulation speed
    clock = pygame.time.Clock()

    try:
        while step < max_frames:
            # Handle pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Window closed by user.")
                    return

            # Random action with no model involved
            action = env.action_space.sample()
            action_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Verbose terminal output every 50 steps
            if step % 50 == 0:
                is_resist = env.episode_data.ground_truth_resistance[
                    min(env.current_frame(), len(env.episode_data.ground_truth_resistance)-1)
                ]
                print(
                    f"Frame {step:>4} | "
                    f"Action: {ACTION_NAMES[action]:<20} | "
                    f"Reward: {reward:+.2f} | "
                    f"Budget: {info['compute_budget']:>5.1f}% | "
                    f"Anomaly: {obs[8]:.3f} | "
                    f"{'[RESISTANCE EVENT]' if is_resist else ''}"
                )

            # Render the current state
            renderer.render(
                frame=env.current_frame(),
                total_frames=env.episode_data.total_frames,
                obs=obs,
                compute_budget=env.compute_budget,
                anomaly_history=env._recent_anomaly_history,
                detection_confidence=env._detection_confidence,
                episode_data=env.episode_data,
                stats={
                    "detections":     env._episode_detections,
                    "misses":         env._episode_misses,
                    "false_alarms":   env._episode_false_alarms,
                    "total_reward":   env._total_reward,
                    "critical_misses": env._critical_misses,
                },
                last_action=action,
                last_reward=reward,
            )

            step += 1
            
            # Limit the framerate to 30 FPS so it's viewable
            clock.tick(30)

            if terminated or truncated:
                print("\nEpisode terminated early.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Save screenshot if headless
    if headless:
        os.makedirs("results", exist_ok=True)
        # Removed local 'import pygame' here
        pygame.image.save(renderer.screen, "results/demo_random_agent_screenshot.png")
        print("Screenshot saved to results/demo_random_agent_screenshot.png")

    print(f"  Random agent summary - {step} frames")
    print(f"  Total reward    : {total_reward:.2f}")
    print(f"  Detections      : {info['episode_detections']}")
    print(f"  Misses          : {info['episode_misses']}")
    print(f"  False alarms    : {info['episode_false_alarms']}")
    print(f"  Budget remaining: {info['compute_budget']:.1f}%")
    print("\n  Action distribution:")
    for a, count in action_counts.items():
        bar = "#" * (count // 5)
        print(f"    {ACTION_NAMES[a]:<22}: {count:>4}  {bar}")
  

    # Keep window open until user closes it manually
    if not headless:
        print("\nSimulation finished. Close the pygame window to exit.")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False

    env.close()
    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=500,
                        help="Max frames to run (default: 500)")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless and save screenshot")
    args = parser.parse_args()
    run_random_demo(max_frames=args.frames, headless=args.no_display)
