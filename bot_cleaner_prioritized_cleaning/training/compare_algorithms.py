import os
import sys
import time
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from local modules
from bot_cleaner_prioritized_cleaning.environments.environment import EnhancedVacuumCleanerEnv
from bot_cleaner_prioritized_cleaning.algos.sac import EnhancedSAC
from bot_cleaner_prioritized_cleaning.algos.ddpg import EnhancedDDPG
from bot_cleaner_prioritized_cleaning.algos.ppo import EnhancedPPO


def parse_args():
    # Get the project root directory (bot_cleaner_prioritized_cleaning)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_model_dir = os.path.join(project_root, "models")
    default_log_dir = os.path.join(project_root, "logs")
    
    parser = argparse.ArgumentParser(description="Compare SAC, DDPG, and PPO algorithms for robot cleaner")
    parser.add_argument("--env_size_x", type=int, default=10, help="Environment width")
    parser.add_argument("--env_size_y", type=int, default=10, help="Environment height")
    parser.add_argument("--wall_density", type=float, default=0.1, help="Density of walls (>0 for maze environment)")
    parser.add_argument("--dirt_density", type=float, default=0.3, help="Initial density of dirt")
    parser.add_argument("--dirt_spawn_rate", type=float, default=0.01, help="Rate at which new dirt spawns (not used)")
    parser.add_argument("--prioritize_dirt", type=float, default=2.0, help="Reward multiplier for cleaning dirt")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--coverage_radius", type=float, default=1.0, help="Radius of the vacuum cleaner coverage")
    parser.add_argument("--total_steps", type=int, default=50000, help="Total training steps per algorithm")
    parser.add_argument("--eval_interval", type=int, default=5000, help="Evaluation interval")
    parser.add_argument("--algorithms", type=str, default="sac,ddpg,ppo", help="Comma-separated list of algorithms to compare (sac,ddpg,ppo)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_dir", type=str, default=default_model_dir, help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default=default_log_dir, help="Directory to save logs")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation")
    
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


def evaluate_sac_policy(agent, env, n_episodes=5, render=False):
    """Evaluate the SAC agent performance"""
    total_rewards = []
    coverage_percentages = []
    dirt_cleaned_percentages = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.05)

            action = agent.get_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
        
        total_rewards.append(episode_reward)
        coverage_percentages.append(info.get('coverage_percentage', 0))
        dirt_cleaned_percentages.append(info.get('dirt_cleaned_percentage', 0))
    
    if render:
        env.close()
        
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_coverage': np.mean(coverage_percentages),
        'mean_dirt_cleaned': np.mean(dirt_cleaned_percentages)
    }


def evaluate_ddpg_policy(agent, env, n_episodes=5, render=False):
    """Evaluate the DDPG agent performance"""
    total_rewards = []
    coverage_percentages = []
    dirt_cleaned_percentages = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.05)

            action = agent.get_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
        
        total_rewards.append(episode_reward)
        coverage_percentages.append(info.get('coverage_percentage', 0))
        dirt_cleaned_percentages.append(info.get('dirt_cleaned_percentage', 0))
    
    if render:
        env.close()
        
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_coverage': np.mean(coverage_percentages),
        'mean_dirt_cleaned': np.mean(dirt_cleaned_percentages)
    }


def evaluate_ppo_policy(agent, env, n_episodes=5, render=False):
    """Evaluate the PPO agent performance"""
    total_rewards = []
    coverage_percentages = []
    dirt_cleaned_percentages = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
                time.sleep(0.05)

            action, _, _ = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
        
        total_rewards.append(episode_reward)
        coverage_percentages.append(info.get('coverage_percentage', 0))
        dirt_cleaned_percentages.append(info.get('dirt_cleaned_percentage', 0))
    
    if render:
        env.close()
        
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_coverage': np.mean(coverage_percentages),
        'mean_dirt_cleaned': np.mean(dirt_cleaned_percentages)
    }


def collect_ppo_rollout(agent, env, steps=256):
    """Collect a rollout for PPO training"""
    rollout = []
    obs = env.reset()
    done = False
    episode_reward = 0
    
    for _ in range(steps):
        action, log_prob, value = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        
        # Store transition
        rollout.append({
            "coverage": obs["coverage"],
            "walls": obs["walls"],
            "dirt": obs["dirt"],
            "position": obs["position"],
            "action": action,
            "reward": reward,
            "done": done,
            "log_prob": log_prob,
            "value": value
        })
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            obs = env.reset()
            done = False
            
    return rollout, episode_reward


def plot_comparison(data, filename):
    """Plot and save the comparison results"""
    algorithms = list(data.keys())
    epochs = []
    rewards = []
    coverages = []
    dirt_cleaned = []
    
    # Prepare data for plotting
    for algo in algorithms:
        epochs.append(data[algo]['epochs'])
        rewards.append(data[algo]['rewards'])
        coverages.append(data[algo]['coverage'])
        dirt_cleaned.append(data[algo]['dirt'])
    
    # Create plot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    for i, algo in enumerate(algorithms):
        axs[0, 0].plot(data[algo]['epochs'], data[algo]['rewards'], label=algo.upper())
    axs[0, 0].set_xlabel('Training Epoch')
    axs[0, 0].set_ylabel('Mean Reward')
    axs[0, 0].set_title('Reward Comparison')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot coverage
    for i, algo in enumerate(algorithms):
        axs[0, 1].plot(data[algo]['epochs'], data[algo]['coverage'], label=algo.upper())
    axs[0, 1].set_xlabel('Training Epoch')
    axs[0, 1].set_ylabel('Coverage Percentage')
    axs[0, 1].set_title('Coverage Comparison')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot dirt cleaned
    for i, algo in enumerate(algorithms):
        axs[1, 0].plot(data[algo]['epochs'], data[algo]['dirt'], label=algo.upper())
    axs[1, 0].set_xlabel('Training Epoch')
    axs[1, 0].set_ylabel('Dirt Cleaned Percentage')
    axs[1, 0].set_title('Cleaning Efficiency Comparison')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot sample efficiency (reward per epoch)
    for i, algo in enumerate(algorithms):
        if len(data[algo]['epochs']) > 0:
            epochs_array = np.array(data[algo]['epochs'])
            rewards_array = np.array(data[algo]['rewards'])
            if epochs_array.max() > 0:  # Prevent division by zero
                efficiency = rewards_array / np.maximum(epochs_array, 1)
                axs[1, 1].plot(epochs_array, efficiency, label=algo.upper())
    
    axs[1, 1].set_xlabel('Training Epoch')
    axs[1, 1].set_ylabel('Reward per Epoch')
    axs[1, 1].set_title('Learning Efficiency Comparison')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def compare_algorithms():
    """Main comparison function"""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Parse algorithms to compare
    algorithms = [algo.strip().lower() for algo in args.algorithms.split(',')]
    valid_algorithms = ["sac", "ddpg", "ppo"]
    algorithms = [algo for algo in algorithms if algo in valid_algorithms]
    
    if not algorithms:
        print("No valid algorithms specified. Please choose from: sac, ddpg, ppo")
        return
    
    print(f"Comparing algorithms: {', '.join(algo.upper() for algo in algorithms)}")
    print(f"Training each for {args.total_steps} steps")
    
    # Initialize tracking variables
    comparison_data = {algo: {'steps': [], 'rewards': [], 'coverage': [], 'dirt': [], 'epochs': []} for algo in algorithms}
    
    try:
        # Train and evaluate each algorithm
        for algo in algorithms:
            print(f"\n{'='*50}")
            print(f"Training {algo.upper()} for {args.total_steps} steps")
            print(f"{'='*50}\n")
            
            env = make_env(args, seed=args.seed)
            
            if algo == "sac":
                agent = EnhancedSAC(env)
                train_sac(agent, env, args, comparison_data["sac"])
            elif algo == "ddpg":
                agent = EnhancedDDPG(env)
                train_ddpg(agent, env, args, comparison_data["ddpg"])
            elif algo == "ppo":
                agent = EnhancedPPO(env)
                # Force PPO to start with a smaller buffer for quicker updates
                agent.buffer_size = 512
                train_ppo(agent, env, args, comparison_data["ppo"])
        
        # Generate final comparison plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(args.log_dir, f'algorithm_comparison_{timestamp}.png')
        plot_comparison(comparison_data, plot_filename)
        print(f"\nComparison completed. Results saved to {plot_filename}")
        
        # Final evaluation with rendering if requested
        if args.render:
            print("\nPerforming final evaluation with rendering...")
            for algo in algorithms:
                print(f"\nFinal {algo.upper()} evaluation:")
                eval_env = make_env(args, seed=args.seed + 500)
                model_path = os.path.join(args.model_dir, f'{algo}_final_model.pt')
                
                if algo == "sac":
                    agent = EnhancedSAC(eval_env)
                    agent.load(model_path)
                    evaluate_sac_policy(agent, eval_env, n_episodes=1, render=True)
                elif algo == "ddpg":
                    agent = EnhancedDDPG(eval_env)
                    agent.load(model_path)
                    evaluate_ddpg_policy(agent, eval_env, n_episodes=1, render=True)
                elif algo == "ppo":
                    agent = EnhancedPPO(eval_env)
                    agent.load(model_path)
                    evaluate_ppo_policy(agent, eval_env, n_episodes=1, render=True)
                    
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


def train_sac(agent, env, args, tracking_data):
    """Train the SAC agent and track performance"""
    # Initialize tracking variables
    total_steps = 0
    episode = 0
    episode_reward = 0
    best_eval_reward = -float('inf')
    
    # Initial observation
    obs = env.reset()
    
    # Initial evaluation at step 0
    eval_env = make_env(args, seed=args.seed + 100)
    eval_stats = evaluate_sac_policy(agent, eval_env)
    
    print(f"\nSAC Initial Evaluation:")
    print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
    print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}%\n")
    
    # Track evaluation metrics
    tracking_data['steps'].append(0)
    tracking_data['epochs'].append(0)
    tracking_data['rewards'].append(eval_stats['mean_reward'])
    tracking_data['coverage'].append(eval_stats['mean_coverage'])
    tracking_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
    
    # Main training loop
    while total_steps < args.total_steps:
        # Get action from the agent
        action = agent.get_action(obs)
        
        # Take a step in the environment
        next_obs, reward, done, info = env.step(action)
        total_steps += 1
        episode_reward += reward
        
        # Store experience in replay buffer
        agent.add_to_replay_buffer(obs, action, reward, next_obs, done)
        
        # Update agent
        agent.update()
        
        # Move to next state
        obs = next_obs
        
        # Episode completed
        if done:
            episode += 1
            
            # Log progress
            if episode % 10 == 0:
                print(f"SAC Episode: {episode} | "
                      f"Steps: {total_steps} | "
                      f"Reward: {episode_reward:.2f}")
            
            # Reset for next episode
            obs = env.reset()
            episode_reward = 0
        
        # Evaluate policy periodically
        if total_steps % args.eval_interval == 0:
            eval_env = make_env(args, seed=args.seed + 100)
            eval_stats = evaluate_sac_policy(agent, eval_env)
            
            print(f"\nSAC Evaluation at step {total_steps}:")
            print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
            print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}%\n")
            
            # Track evaluation metrics
            tracking_data['steps'].append(total_steps)
            tracking_data['epochs'].append(episode)
            tracking_data['rewards'].append(eval_stats['mean_reward'])
            tracking_data['coverage'].append(eval_stats['mean_coverage'])
            tracking_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
            
            # Save model
            if eval_stats['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_stats['mean_reward']
                agent.save(os.path.join(args.model_dir, 'sac_best_model.pt'))
    
    # Save final model
    agent.save(os.path.join(args.model_dir, 'sac_final_model.pt'))
    print(f"SAC training completed")


def train_ddpg(agent, env, args, tracking_data):
    """Train the DDPG agent and track performance"""
    # Initialize tracking variables
    total_steps = 0
    episode = 0
    episode_reward = 0
    best_eval_reward = -float('inf')
    
    # Initial observation
    obs = env.reset()
    
    # Initial evaluation at step 0
    eval_env = make_env(args, seed=args.seed + 200)
    eval_stats = evaluate_ddpg_policy(agent, eval_env)
    
    print(f"\nDDPG Initial Evaluation:")
    print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
    print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}%\n")
    
    # Track evaluation metrics
    tracking_data['steps'].append(0)
    tracking_data['epochs'].append(0)
    tracking_data['rewards'].append(eval_stats['mean_reward'])
    tracking_data['coverage'].append(eval_stats['mean_coverage'])
    tracking_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
    
    # Main training loop
    while total_steps < args.total_steps:
        # Get action from the agent
        action = agent.get_action(obs)
        
        # Take a step in the environment
        next_obs, reward, done, info = env.step(action)
        total_steps += 1
        episode_reward += reward
        
        # Store experience in replay buffer
        agent.add_to_replay_buffer(obs, action, reward, next_obs, done)
        
        # Update agent
        agent.update()
        
        # Move to next state
        obs = next_obs
        
        # Episode completed
        if done:
            episode += 1
            
            # Log progress
            if episode % 10 == 0:
                print(f"DDPG Episode: {episode} | "
                      f"Steps: {total_steps} | "
                      f"Reward: {episode_reward:.2f}")
            
            # Reset for next episode
            obs = env.reset()
            episode_reward = 0
            agent.noise.reset()  # Reset exploration noise
        
        # Evaluate policy periodically
        if total_steps % args.eval_interval == 0:
            eval_env = make_env(args, seed=args.seed + 200)
            eval_stats = evaluate_ddpg_policy(agent, eval_env)
            
            print(f"\nDDPG Evaluation at step {total_steps}:")
            print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
            print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}%\n")
            
            # Track evaluation metrics
            tracking_data['steps'].append(total_steps)
            tracking_data['epochs'].append(episode)
            tracking_data['rewards'].append(eval_stats['mean_reward'])
            tracking_data['coverage'].append(eval_stats['mean_coverage'])
            tracking_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
            
            # Save model
            if eval_stats['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_stats['mean_reward']
                agent.save(os.path.join(args.model_dir, 'ddpg_best_model.pt'))
    
    # Save final model
    agent.save(os.path.join(args.model_dir, 'ddpg_final_model.pt'))
    print(f"DDPG training completed")


def train_ppo(agent, env, args, tracking_data):
    """Train the PPO agent and track performance"""
    total_steps = 0
    episode = 0
    best_eval_reward = -float('inf')
    
    # Force evaluation at step 0
    eval_env = make_env(args, seed=args.seed + 300)
    eval_stats = evaluate_ppo_policy(agent, eval_env)
    
    # Track initial metrics
    tracking_data['steps'].append(0)
    tracking_data['epochs'].append(0)
    tracking_data['rewards'].append(eval_stats['mean_reward'])
    tracking_data['coverage'].append(eval_stats['mean_coverage'])
    tracking_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
    
    print(f"\nPPO Initial Evaluation:")
    print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
    print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
    print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}%\n")
            
    while total_steps < args.total_steps:
        # Use much smaller rollouts to ensure frequent updates
        rollout, episode_reward = collect_ppo_rollout(agent, env)
        rollout_steps = len(rollout)
        
        # Check if we'll cross an evaluation threshold with this rollout
        next_eval = ((total_steps // args.eval_interval) + 1) * args.eval_interval
        
        # Update total steps
        prev_steps = total_steps
        total_steps += rollout_steps
        episode += 1
        
        # Update PPO agent
        agent.update(rollout)
        
        # Log progress
        if episode % 10 == 0:
            print(f"PPO Episode: {episode} | "
                  f"Steps: {total_steps} | "
                  f"Rollout Reward: {episode_reward:.2f}")
        
        # If we crossed or hit an exact eval threshold
        if prev_steps < next_eval and total_steps >= next_eval:
            eval_env = make_env(args, seed=args.seed + 300)
            # Force evaluation at exact interval for better plotting
            eval_stats = evaluate_ppo_policy(agent, eval_env)
            
            print(f"\nPPO Evaluation at step {next_eval}:")
            print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
            print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}%\n")
            
            # Track evaluation metrics at exact intervals
            tracking_data['steps'].append(next_eval)
            tracking_data['epochs'].append(episode)
            tracking_data['rewards'].append(eval_stats['mean_reward'])
            tracking_data['coverage'].append(eval_stats['mean_coverage'])
            tracking_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
            
            # Save model
            if eval_stats['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_stats['mean_reward']
                agent.save(os.path.join(args.model_dir, 'ppo_best_model.pt'))
    
    # Final evaluation
    eval_env = make_env(args, seed=args.seed + 300)
    eval_stats = evaluate_ppo_policy(agent, eval_env)
    
    # Save final model
    agent.save(os.path.join(args.model_dir, 'ppo_final_model.pt'))
    print(f"PPO training completed")


if __name__ == "__main__":
    compare_algorithms() 