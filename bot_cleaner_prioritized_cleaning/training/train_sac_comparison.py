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

# Fix import paths
from bot_cleaner_prioritized_cleaning.environments.environment import EnhancedVacuumCleanerEnv
from bot_cleaner_prioritized_cleaning.algos.sac import EnhancedSAC
from bot_cleaner_prioritized_cleaning.algos.ppo import EnhancedPPO
# Alternatively, if the above doesn't work:
# from environments.environment import BotCleanerEnv
# from algos.sac import EnhancedSAC
# from algos.ppo import EnhancedPPO


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAC and PPO for robot cleaner")
    parser.add_argument("--env_size_x", type=int, default=10, help="Environment width")
    parser.add_argument("--env_size_y", type=int, default=10, help="Environment height")
    parser.add_argument("--wall_density", type=float, default=0.1, help="Density of walls")
    parser.add_argument("--dirt_density", type=float, default=0.3, help="Initial density of dirt")
    parser.add_argument("--dirt_spawn_rate", type=float, default=0.01, help="Rate at which new dirt spawns")
    parser.add_argument("--prioritize_dirt", type=float, default=2.0, help="Reward multiplier for cleaning dirt")
    parser.add_argument("--total_steps", type=int, default=500000, help="Total training steps per algorithm")
    parser.add_argument("--eval_interval", type=int, default=10000, help="Evaluation interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_dir", type=str, default="models/comparison", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs/comparison", help="Directory to save logs")
    
    return parser.parse_args()


def make_env(args, seed=None):
    """Create the environment with the specified parameters"""
    env = BotCleanerEnv(
        size_x=args.env_size_x,
        size_y=args.env_size_y,
        wall_density=args.wall_density,
        dirt_density=args.dirt_density,
        dirt_spawn_rate=args.dirt_spawn_rate,
        prioritize_dirt=args.prioritize_dirt
    )
    
    if seed is not None:
        env.seed(seed)
        
    return env


def evaluate_sac_policy(agent, env, n_episodes=5):
    """Evaluate the SAC agent performance"""
    total_rewards = []
    coverage_percentages = []
    dirt_cleaned = []
    
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
        dirt_cleaned.append(info.get('total_dirt_cleaned', 0))
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_coverage': np.mean(coverage_percentages),
        'mean_dirt_cleaned': np.mean(dirt_cleaned)
    }


def evaluate_ppo_policy(agent, env, n_episodes=5):
    """Evaluate the PPO agent performance"""
    total_rewards = []
    coverage_percentages = []
    dirt_cleaned = []
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _, _ = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            episode_reward += reward
            obs = next_obs
        
        total_rewards.append(episode_reward)
        coverage_percentages.append(info.get('coverage_percentage', 0))
        dirt_cleaned.append(info.get('total_dirt_cleaned', 0))
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_coverage': np.mean(coverage_percentages),
        'mean_dirt_cleaned': np.mean(dirt_cleaned)
    }


def collect_ppo_rollout(agent, env, steps=2048):
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


def plot_comparison(sac_data, ppo_data, filename):
    """Plot and save the comparison results"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axs[0, 0].plot(sac_data['steps'], sac_data['rewards'], 'b-', label='SAC')
    axs[0, 0].plot(ppo_data['steps'], ppo_data['rewards'], 'r-', label='PPO')
    axs[0, 0].set_xlabel('Training Steps')
    axs[0, 0].set_ylabel('Mean Reward')
    axs[0, 0].set_title('Reward Comparison')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot coverage
    axs[0, 1].plot(sac_data['steps'], sac_data['coverage'], 'b-', label='SAC')
    axs[0, 1].plot(ppo_data['steps'], ppo_data['coverage'], 'r-', label='PPO')
    axs[0, 1].set_xlabel('Training Steps')
    axs[0, 1].set_ylabel('Coverage Percentage')
    axs[0, 1].set_title('Coverage Comparison')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot dirt cleaned
    axs[1, 0].plot(sac_data['steps'], sac_data['dirt'], 'b-', label='SAC')
    axs[1, 0].plot(ppo_data['steps'], ppo_data['dirt'], 'r-', label='PPO')
    axs[1, 0].set_xlabel('Training Steps')
    axs[1, 0].set_ylabel('Dirt Cleaned')
    axs[1, 0].set_title('Cleaning Efficiency Comparison')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot sample efficiency (reward per 1000 steps)
    steps_per_unit = 1000
    sac_steps = np.array(sac_data['steps']) // steps_per_unit
    ppo_steps = np.array(ppo_data['steps']) // steps_per_unit
    
    if len(sac_steps) > 0 and len(ppo_steps) > 0:
        sac_efficiency = np.array(sac_data['rewards']) / sac_steps
        ppo_efficiency = np.array(ppo_data['rewards']) / ppo_steps
        
        axs[1, 1].plot(sac_data['steps'], sac_efficiency, 'b-', label='SAC')
        axs[1, 1].plot(ppo_data['steps'], ppo_efficiency, 'r-', label='PPO')
        axs[1, 1].set_xlabel('Training Steps')
        axs[1, 1].set_ylabel(f'Reward per {steps_per_unit} Steps')
        axs[1, 1].set_title('Sample Efficiency Comparison')
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
    
    # Initialize tracking variables
    sac_data = {'steps': [], 'rewards': [], 'coverage': [], 'dirt': []}
    ppo_data = {'steps': [], 'rewards': [], 'coverage': [], 'dirt': []}
    
    # Train and evaluate SAC
    print("Starting SAC training...")
    env_sac = make_env(args, seed=args.seed)
    agent_sac = EnhancedSAC(env_sac)
    
    start_time = time.time()
    total_steps = 0
    episode = 0
    episode_reward = 0
    obs = env_sac.reset()
    
    while total_steps < args.total_steps:
        # Get action from SAC agent
        action = agent_sac.get_action(obs)
        
        # Take step in environment
        next_obs, reward, done, info = env_sac.step(action)
        total_steps += 1
        episode_reward += reward
        
        # Store experience in replay buffer
        agent_sac.add_to_replay_buffer(obs, action, reward, next_obs, done)
        
        # Update agent
        agent_sac.update()
        
        # Move to next state
        obs = next_obs
        
        # Episode completed
        if done:
            episode += 1
            
            # Log progress
            if episode % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"SAC Episode: {episode} | "
                      f"Steps: {total_steps} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Elapsed: {elapsed_time:.2f}s")
            
            # Reset for next episode
            obs = env_sac.reset()
            episode_reward = 0
        
        # Evaluate policy periodically
        if total_steps % args.eval_interval == 0:
            eval_env = make_env(args, seed=args.seed + 100)
            eval_stats = evaluate_sac_policy(agent_sac, eval_env)
            
            print(f"\nSAC Evaluation at step {total_steps}:")
            print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
            print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}\n")
            
            # Track evaluation metrics
            sac_data['steps'].append(total_steps)
            sac_data['rewards'].append(eval_stats['mean_reward'])
            sac_data['coverage'].append(eval_stats['mean_coverage'])
            sac_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
            
            # Save model
            agent_sac.save(os.path.join(args.model_dir, f'sac_checkpoint_{total_steps}.pt'))
    
    # Save final SAC model
    agent_sac.save(os.path.join(args.model_dir, 'sac_final.pt'))
    print(f"SAC training completed")
    
    # Train and evaluate PPO
    print("\nStarting PPO training...")
    env_ppo = make_env(args, seed=args.seed)
    agent_ppo = EnhancedPPO(env_ppo)
    
    start_time = time.time()
    total_steps = 0
    episode = 0
    
    while total_steps < args.total_steps:
        # Collect rollout for PPO
        rollout, episode_reward = collect_ppo_rollout(agent_ppo, env_ppo)
        total_steps += len(rollout)
        episode += 1
        
        # Update PPO agent
        agent_ppo.update(rollout)
        
        # Log progress
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"PPO Episode: {episode} | "
                  f"Steps: {total_steps} | "
                  f"Rollout Reward: {episode_reward:.2f} | "
                  f"Elapsed: {elapsed_time:.2f}s")
        
        # Evaluate policy periodically
        if total_steps >= args.eval_interval and total_steps % args.eval_interval < 2048:
            eval_env = make_env(args, seed=args.seed + 200)
            eval_step = (total_steps // args.eval_interval) * args.eval_interval
            eval_stats = evaluate_ppo_policy(agent_ppo, eval_env)
            
            print(f"\nPPO Evaluation at step {eval_step}:")
            print(f"  Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Mean Coverage: {eval_stats['mean_coverage']:.2f}%")
            print(f"  Mean Dirt Cleaned: {eval_stats['mean_dirt_cleaned']:.2f}\n")
            
            # Track evaluation metrics
            ppo_data['steps'].append(eval_step)
            ppo_data['rewards'].append(eval_stats['mean_reward'])
            ppo_data['coverage'].append(eval_stats['mean_coverage'])
            ppo_data['dirt'].append(eval_stats['mean_dirt_cleaned'])
            
            # Save model
            agent_ppo.save(os.path.join(args.model_dir, f'ppo_checkpoint_{eval_step}.pt'))
            
            # Plot comparison if we have data from both algorithms
            if len(sac_data['steps']) > 0 and len(ppo_data['steps']) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = os.path.join(args.log_dir, f'comparison_{timestamp}.png')
                plot_comparison(sac_data, ppo_data, plot_filename)
    
    # Save final PPO model
    agent_ppo.save(os.path.join(args.model_dir, 'ppo_final.pt'))
    print(f"PPO training completed")
    
    # Final comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(args.log_dir, f'final_comparison_{timestamp}.png')
    plot_comparison(sac_data, ppo_data, plot_filename)
    print(f"Comparison completed. Results saved to {plot_filename}")


if __name__ == "__main__":
    compare_algorithms() 