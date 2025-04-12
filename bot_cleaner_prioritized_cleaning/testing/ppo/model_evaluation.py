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


def evaluate(args):
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

    # Load the trained model
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Set the policy to evaluation mode
    agent.policy.eval()

    # Run evaluation episodes
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # Render the environment
            if args.render:
                env.render()
                time.sleep(args.render_delay)

            # Get action using policy mean (deterministic)
            coverage = (
                torch.FloatTensor(obs["coverage"])
                .view(1, env.size_y, env.size_x)
                .to(agent.device)
            )
            walls = (
                torch.FloatTensor(obs["walls"])
                .view(1, env.size_y, env.size_x)
                .to(agent.device)
            )
            dirt = (
                torch.FloatTensor(obs["dirt"])
                .view(1, env.size_y, env.size_x)
                .to(agent.device)
            )
            position = torch.FloatTensor(obs["position"]).view(1, -1).to(agent.device)

            with torch.no_grad():
                mean, _, _ = agent.policy(coverage, walls, dirt, position)
                action = mean.squeeze().cpu().numpy()

            # Clip action to environment limits
            action = np.clip(action, env.action_space.low, env.action_space.high)

            # Take step in environment
            next_obs, reward, done, info = env.step(action)

            total_reward += reward
            obs = next_obs
            step += 1

            # Print step info if verbose
            if args.verbose:
                print(f"Step {step}: Action = {action}, Reward = {reward:.2f}")

        # Print episode results
        print(f"Evaluation Episode {episode+1}/{args.episodes}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Coverage: {info['coverage_percentage']:.2%}")
        print(f"Dirt Cleaned: {info['dirt_cleaned_percentage']:.2%}")
        print(f"Steps Used: {info['steps']}/{env.max_steps}")
        print("-" * 40)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PPO agent for EnhancedVacuumCleanerEnv"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file"
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
        "--episodes", type=int, default=5, help="Number of evaluation episodes"
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--render_delay", type=float, default=0.05, help="Delay between rendered frames"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed step information"
    )

    args = parser.parse_args()
    evaluate(args)
