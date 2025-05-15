"""DDPG Model Evaluation for Continuous Vacuum Cleaner Environment"""
import torch
import numpy as np
from bot_cleaner_dynamic_environment.environments.environment import ContinuousVacuumCleanerEnv
from bot_cleaner_dynamic_environment.algos.ddpg import DDPG

def preprocess_observation(obs: dict) -> tuple:
    """Process observation for DDPG network input."""
    coverage = obs['coverage'].reshape(1, 1, 50, 50)
    position = obs['position'].reshape(1, -1)
    return coverage, position

def evaluate(model_path: str = "bot_cleaner_dynamic_environment/models/ddpg/ddpg_cleaner_final.pth",
            num_episodes: int = 3,
            render: bool = True):
    """
    Evaluate a trained DDPG agent.
    
    Args:
        model_path: Path to the saved model checkpoint
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment
    """
    # Initialize environment
    env = ContinuousVacuumCleanerEnv(size=10.0, coverage_radius=0.5)
    agent = DDPG(env)

    # Load trained model
    try:
        checkpoint = torch.load(model_path)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.actor.eval()  # Set to evaluation mode
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Track metrics
    episode_rewards = []
    episode_coverages = []
    episode_steps = []

    try:
        # Evaluation loop
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                # Get preprocessed observations
                coverage, position = preprocess_observation(obs)

                # Get deterministic action
                with torch.no_grad():
                    action = agent.get_action(coverage, position, evaluate=True)
                    action = np.clip(action, env.action_space.low, env.action_space.high)

                # Step environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                if render:
                    env.render()

            # Store episode statistics
            episode_rewards.append(total_reward)
            episode_coverages.append(info['coverage_percentage'])
            episode_steps.append(steps)

            # Print episode results
            print(f"Episode {ep+1}/{num_episodes} | "
                  f"Total Reward: {total_reward:.1f} | "
                  f"Coverage: {info['coverage_percentage']:.2%} | "
                  f"Steps: {steps}")

        # Print evaluation summary
        print("\nEvaluation Summary:")
        print(f"Average Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
        print(f"Average Coverage: {np.mean(episode_coverages):.2%} ± {np.std(episode_coverages):.2%}")
        print(f"Average Steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    finally:
        env.close()

if __name__ == "__main__":
    evaluate(
        model_path="bot_cleaner_dynamic_environment/models/ddpg/ddpg_cleaner_final.pth",
        num_episodes=3,
        render=True
    )
