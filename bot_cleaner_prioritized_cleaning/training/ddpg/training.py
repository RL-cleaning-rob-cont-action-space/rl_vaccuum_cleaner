import os
import sys
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from local modules
from bot_cleaner_prioritized_cleaning.environments.environment import EnhancedVacuumCleanerEnv
from bot_cleaner_prioritized_cleaning.algos.ddpg import EnhancedDDPG


def parse_args():
    # Get the project root directory (bot_cleaner_prioritized_cleaning)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_model_dir = os.path.join(project_root, "models")
    default_log_dir = os.path.join(project_root, "logs")
    
    parser = argparse.ArgumentParser(description="Train a robot cleaner agent using DDPG")
    parser.add_argument("--env_size_x", type=int, default=10, help="Environment width")
    parser.add_argument("--env_size_y", type=int, default=10, help="Environment height")
    parser.add_argument("--wall_density", type=float, default=0.1, help="Density of walls (>0 for maze environment)")
    parser.add_argument("--dirt_density", type=float, default=0.3, help="Initial density of dirt")
    parser.add_argument("--dirt_spawn_rate", type=float, default=0.01, help="Rate at which new dirt spawns (not used)")
    parser.add_argument("--prioritize_dirt", type=float, default=2.0, help="Reward multiplier for cleaning dirt")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--coverage_radius", type=float, default=1.0, help="Radius of the vacuum cleaner coverage")
    parser.add_argument("--total_steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--eval_interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval")
    parser.add_argument("--noise_scale", type=float, default=0.5, help="Scale of exploration noise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_dir", type=str, default=default_model_dir, help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default=default_log_dir, help="Directory to save logs")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering (overrides --render)")
    
    return parser.parse_args()


def make_env(args, seed=None):
    """Create the environment with the specified parameters"""
    env = EnhancedVacuumCleanerEnv(
        size_x=args.env_size_x,
        size_y=args.env_size_y,
        coverage_radius=args.coverage_radius,
        max_steps=args.max_steps,
        env_type="maze" if args.wall_density > 0 else "empty",
        dirt_percentage=args.dirt_density,
        dirt_reward_multiplier=args.prioritize_dirt,
        random_seed=seed
    )
        
    return env


def evaluate_policy(agent, env, n_episodes=5):
    """Evaluate the agent performance"""
    total_rewards = []
    coverage_percentages = []
    dirt_cleaned_percentages = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.get_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
        
        total_rewards.append(episode_reward)
        coverage_percentages.append(info.get('coverage_percentage', 0))
        dirt_cleaned_percentages.append(info.get('dirt_cleaned_percentage', 0))
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_coverage': np.mean(coverage_percentages),
        'mean_dirt_cleaned': np.mean(dirt_cleaned_percentages)
    }


def plot_learning_curve(steps, scores, avg_scores, filename):
    """Plot and save the learning curve"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(steps, scores, 'b', alpha=0.3)
    ax.plot(steps, avg_scores, 'r')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Score')
    ax.set_title('DDPG Learning Curve')
    ax.grid(True)
    plt.savefig(filename)
    plt.close()


def train_ddpg():
    """Main training function"""
    args = parse_args()
    
    # Ensure steps are limited for testing
    if args.total_steps > 100 and "--total_steps" not in sys.argv:
        print("Limiting to 100 steps for testing. Use --total_steps to specify more.")
        args.total_steps = 100
    
    # Force rendering on for testing unless explicitly disabled
    if "--no_render" not in sys.argv:
        args.render = True
        print("Enabling rendering for testing. Use --no_render to disable.")
    
    print(f"Configuration:")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Rendering: {'Enabled' if args.render else 'Disabled'}")
    print(f"  Environment size: {args.env_size_x}x{args.env_size_y}")
    print(f"  Dirt density: {args.dirt_density}")
    print(f"  Model directory: {os.path.abspath(args.model_dir)}")
    print(f"  Log directory: {os.path.abspath(args.log_dir)}")
    print(f"  Noise scale: {args.noise_scale}")
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    try:
        # Create environment and agent
        env = make_env(args, seed=args.seed)
        agent = EnhancedDDPG(env)
        agent.noise_scale = args.noise_scale  # Set exploration noise scale
        
        # Initialize tracking variables
        start_time = time.time()
        episode_rewards = []
        episode_coverages = []
        episode_dirt_cleaned = []
        steps_tracking = []
        episode_reward = 0
        best_eval_reward = -float('inf')
        
        # Training tracking
        total_steps = 0
        episode = 0
        
        # Initial observation
        obs = env.reset()
        
        print(f"Starting DDPG training for {args.total_steps} steps...")
        
        # Main training loop
        while total_steps < args.total_steps:
            # Render if enabled
            if args.render:
                env.render()
                time.sleep(0.05)  # Add small delay for better visualization
                
            # Get action from the agent
            action = agent.get_action(obs)
            
            # Take a step in the environment
            next_obs, reward, done, info = env.step(action)
            total_steps += 1
            episode_reward += reward
            
            # Store experience in replay buffer
            agent.add_to_replay_buffer(obs, action, reward, next_obs, done)
            
            # Update agent (every 5 steps to speed up testing)
            if total_steps % 5 == 0 and total_steps >= agent.min_buffer_size:
                print(f"Updating agent at step {total_steps}...")
                agent.update()
            
            # Move to next state
            obs = next_obs
            
            # Log current status periodically
            if total_steps % 10 == 0:
                print(f"Step {total_steps}/{args.total_steps} - "
                      f"Current reward: {episode_reward:.2f}")
            
            # Episode completed
            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                episode_coverages.append(info.get('coverage_percentage', 0))
                episode_dirt_cleaned.append(info.get('dirt_cleaned_percentage', 0))
                steps_tracking.append(total_steps)
                
                # Log progress
                print(f"Episode {episode} completed - "
                      f"Steps: {total_steps} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Coverage: {info.get('coverage_percentage', 0):.2f}% | "
                      f"Dirt Cleaned: {info.get('dirt_cleaned_percentage', 0):.2f}")
                
                # Reset for next episode
                obs = env.reset()
                episode_reward = 0
                # Reset noise process for next episode
                agent.noise.reset()
            
            # Evaluate policy periodically
            if total_steps % args.eval_interval == 0:
                eval_env = make_env(args, seed=args.seed + 100)
                eval_stats = evaluate_policy(agent, eval_env)
                
                print(f"\nEvaluation at step {total_steps}:")
                print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
                print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
                print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}%\n")
                
                # Save the best model
                if eval_stats['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_stats['mean_reward']
                    model_path = os.path.join(args.model_dir, 'ddpg_best_model.pt')
                    agent.save(model_path)
                    print(f"  New best model saved to {model_path}")
            
            # Save model checkpoint
            if total_steps % args.save_interval == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(args.model_dir, f'ddpg_checkpoint_{total_steps}_{timestamp}.pt')
                agent.save(model_path)
                print(f"Checkpoint saved to {model_path}")
        
        # Close the environment properly
        if args.render:
            env.close()
        
        # Save final model
        model_path = os.path.join(args.model_dir, 'ddpg_final_model.pt')
        agent.save(model_path)
        print(f"Training completed. Final model saved to {model_path}")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        if 'env' in locals() and args.render:
            env.close()


if __name__ == "__main__":
    train_ddpg() 