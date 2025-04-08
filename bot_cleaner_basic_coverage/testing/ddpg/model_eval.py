import numpy as np
import torch
from bot_cleaner_basic_coverage.algos.ddpg import DDPG
from bot_cleaner_basic_coverage.environments.environment import (
    ContinuousVacuumCleanerEnv,
)


def evaluate_ddpg(
    model_path="bot_cleaner_basic_coverage/models/ddpg/final_model.pth", episodes=3
):
    # Create environment
    env = ContinuousVacuumCleanerEnv(size=10.0, resolution=50, coverage_radius=0.5)

    # Initialize DDPG agent
    agent = DDPG(env)

    # Load the trained model
    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()  # Set to evaluation mode

    coverage_percentages = []
    total_rewards = []

    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Preprocess observations
            coverage = obs["coverage"].reshape(1, 50, 50)
            position = obs["position"]

            # Take deterministic action (without noise)
            action = agent.select_action(coverage, position, evaluate=True)

            # Step the environment
            next_obs, reward, done, info = env.step(action)

            # Render
            env.render(mode="human")

            total_reward += reward
            obs = next_obs

        # Record metrics
        coverage_percentages.append(info["coverage_percentage"])
        total_rewards.append(total_reward)

        print(f"Evaluation Episode {episode+1}/{episodes}")
        print(f"Final Coverage: {info['coverage_percentage']:.2%}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps Taken: {info['steps']}")
        print("-" * 50)

    # Print summary statistics
    avg_coverage = np.mean(coverage_percentages)
    avg_reward = np.mean(total_rewards)

    print("=" * 50)
    print("Evaluation Summary:")
    print(f"Average Coverage: {avg_coverage:.2%}")
    print(f"Average Total Reward: {avg_reward:.2f}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    evaluate_ddpg()
