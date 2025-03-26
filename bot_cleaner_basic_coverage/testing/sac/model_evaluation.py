"""SAC Model Evaluation for Continuous Vacuum Cleaner Environment"""
import torch
import numpy as np
from bot_cleaner_basic_coverage.environments.environment import ContinuousVacuumCleanerEnv
from bot_cleaner_basic_coverage.algos.sac import SAC  # Updated import path

def preprocess_observation(obs: dict) -> np.ndarray:
    """Match the preprocessing used in training"""
    return np.concatenate([
        obs['coverage'].flatten(),  # Ensure 2D array is flattened
        obs['position']             # Include position coordinates
    ])

def evaluate(model_path: str = "bot_cleaner_basic_coverage/models/sac/sac_cleaner_actor_final.pth",
            num_episodes: int = 3,
            render: bool = True):

    # Initialize environment (match training parameters)
    env = ContinuousVacuumCleanerEnv(size=10.0, coverage_radius=0.5)

    # Agent configuration (must match training setup)
    state_dim = 50*50 + 3  # 50x50 grid + 3 position values
    action_dim = env.action_space.shape[0]
    action_range = (env.action_space.low, env.action_space.high)  # As tuple

    # Create agent (using SAC class directly)
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_range=action_range,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load trained model
    agent.load_model(model_path)

    # Evaluation loop
    for ep in range(num_episodes):
        obs = env.reset()
        state = preprocess_observation(obs)
        total_reward = 0.0
        done = False

        while not done:
            with torch.no_grad():  # Disable gradient calculation
                action = agent.get_action(state, deterministic=True)

            obs, reward, done, _ = env.step(action)
            state = preprocess_observation(obs)
            total_reward += reward

            if render:
                env.render()

        print(f"Episode {ep+1}/{num_episodes} | "
            f"Total Reward: {total_reward:.1f} | "
            f"Coverage: {env.coverage_percentage:.2%} | "
            f"Steps: {env.step_count}")

    env.close()

if __name__ == "__main__":
    evaluate(
        model_path="bot_cleaner_basic_coverage/models/sac/sac_cleaner_actor_final.pth",
        num_episodes=3,
        render=True
    )
