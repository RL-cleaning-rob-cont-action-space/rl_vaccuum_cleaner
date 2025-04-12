import argparse
import os
import sys
import time

import numpy as np
import torch

# Add parent directories to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from algos.ppo import EnhancedPPO
from environments.environment import EnhancedVacuumCleanerEnv


def train(args):
    # Create the environment
    env = EnhancedVacuumCleanerEnv(
        size_x=args.size_x,
        size_y=args.size_y,
        coverage_radius=args.coverage_radius,
        max_steps=args.max_steps,
        env_type=args.env_type,
        dirt_percentage=args.dirt_percentage,
        dirt_reward_multiplier=args.dirt_reward_multiplier,
        random_seed=args.seed,
    )

    # Initialize the PPO agent
    agent = EnhancedPPO(env)

    # Load model if continuing training
    if args.load_model:
        agent.load(args.load_model)
        print(f"Loaded model from {args.load_model}")

    # Create directory for saving models if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Training loop
    max_episodes = args.episodes
    update_interval = args.update_interval
    best_coverage = 0
    best_dirt_cleaning = 0

    try:
        for episode in range(max_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            rollout = []

            while not done:
                # Render the environment if enabled
                if args.render and (episode % args.render_freq == 0):
                    env.render()

                # Get action from policy
                action, log_prob, value = agent.act(obs)

                # Take step in environment
                next_obs, reward, done, info = env.step(action)

                # Store experience
                rollout.append(
                    {
                        "coverage": obs["coverage"].reshape(env.size_y, env.size_x),
                        "walls": obs["walls"].reshape(env.size_y, env.size_x),
                        "dirt": obs["dirt"].reshape(env.size_y, env.size_x),
                        "position": obs["position"],
                        "action": action,
                        "reward": reward,
                        "done": done,
                        "log_prob": log_prob,
                        "value": value,
                    }
                )

                total_reward += reward
                obs = next_obs

                # Update policy if needed
                if len(rollout) >= update_interval:
                    agent.update(rollout)
                    rollout = []

            # Final update with remaining samples
            if len(rollout) > 0:
                agent.update(rollout)

            # Track metrics
            coverage = info["coverage_percentage"]
            dirt_cleaned = info["dirt_cleaned_percentage"]

            # Save model if it's the best so far
            if coverage > best_coverage:
                best_coverage = coverage
                agent.save(f"../../models/best_coverage_model_{best_coverage:.2f}.pth")

            if dirt_cleaned > best_dirt_cleaning:
                best_dirt_cleaning = dirt_cleaned
                agent.save(f"../../models/best_dirt_model_{best_dirt_cleaning:.2f}.pth")

            # Periodically save model
            if episode % args.save_freq == 0:
                agent.save(f"../../models/model_ep{episode}.pth")

            # Print progress
            print(
                f"Episode {episode}/{max_episodes} | "
                f"Coverage: {coverage:.2%} | "
                f"Dirt Cleaned: {dirt_cleaned:.2%} | "
                f"Steps: {info['steps']}/{env.max_steps} | "
                f"Reward: {total_reward:.2f}"
            )

    except KeyboardInterrupt:
        print("Training interrupted by user")

    # Save final model
    agent.save("../../models/final_model.pth")
    print("Training completed. Final model saved.")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent for EnhancedVacuumCleanerEnv"
    )
    parser.add_argument("--size_x", type=int, default=15, help="Environment width")
    parser.add_argument("--size_y", type=int, default=10, help="Environment height")
    parser.add_argument(
        "--coverage_radius", type=float, default=1.0, help="Vacuum coverage radius"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--update_interval", type=int, default=200, help="Steps between policy updates"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="maze",
        choices=["empty", "maze"],
        help="Environment type",
    )
    parser.add_argument(
        "--dirt_percentage",
        type=float,
        default=0.2,
        help="Percentage of cells with dirt",
    )
    parser.add_argument(
        "--dirt_reward_multiplier",
        type=float,
        default=3.0,
        help="Reward multiplier for cleaning dirt",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering during training"
    )
    parser.add_argument(
        "--render_freq", type=int, default=10, help="Render every N episodes"
    )
    parser.add_argument(
        "--save_freq", type=int, default=50, help="Save model every N episodes"
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to load model from"
    )

    args = parser.parse_args()
    train(args)
