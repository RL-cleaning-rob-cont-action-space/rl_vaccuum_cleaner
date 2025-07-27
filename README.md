## RL Vacuum Cleaner: Reinforcement Learning for Autonomous Cleaning

A comprehensive reinforcement learning project implementing multiple deep RL algorithms to train autonomous vacuum cleaner agents for efficient floor coverage in various environments[1]. The project explores different environmental complexities and compares the performance of DDPG, PPO, and SAC algorithms.

### Project Structure

The repository is organized into distinct modules focusing on different cleaning scenarios[1]:

**Core Components:**

- **`bot_cleaner_basic_coverage/`** - Basic area coverage without obstacles[1]
- **`bot_cleaner_static_obstacles/`** - Navigation with fixed maze-like obstacles[1]
- **`bot_cleaner_dynamic_environment/`** - Adaptive environments with dynamic wall generation[1]
- **`bot_cleaner_prioritized_cleaning/`** - Advanced cleaning with dirt prioritization[1]
- **`bot_cleaner_vision_navigation/`** - Vision-based navigation (future development)[1]

**Shared Infrastructure:**

- **`docs/`** - Project documentation and final report[1]
- **`images/`** - Training results, plots, and demonstration videos[1]
- **`utils/`** - Utility functions (currently empty)[1]

### foundation environment implements a continuous 2D space where the vacuum cleaner learns to maximize floor coverage[2]. Key features include:

- **Grid-based coverage tracking** with 50×50 resolution[2]
- **Continuous action space** with linear velocity (-0.5 to 0.5) and angular velocity (-π/2 to π/2)[2]
- **Reward system** that encourages exploration of uncovered areas while penalizing redundant cleaning[2]
- **Episode termination** at 95% coverage or maximum steps (2000)[2]

#### Static Obstacles Environment

Extends the basic environment with maze-like obstacles[3]:

- **Grid-based maze layout** with predefined wall patterns[3]
- **Collision detection** preventing movement through walls[3]
- **Modified reward structure** with penalties for invalid moves[3]
- **Start and exit positions** for navigation objectives[3]

#### Dynamic Environment

Features procedurally generated obstacles for varied training scenarios[4]:

- **Dynamic wall generation** using configurable density parameters[4]
- **Connectivity validation** ensuring all areas remain reachable[4]
- **Collision-based termination** with negative rewards for wall contact[4]
- **Scalable grid system** supporting different resolutions[4]

#### Prioritized Cleaning Environment

Advanced scenario incorporating dirt management and prioritization[5]:

- **Dirt density configuration** with customizable spawn rates[5]
- **Priority-based rewards** with higher values for dirt cleaning[5]
- **Wall density control** for maze complexity adjustment[5]
- **Comprehensive command-line interface** for training configuration[5]

### Algorithms

#### Deep Deterministic Policy Gradient (DDPG)

Implements an actor-critic approach for continuous control[6]:

**Architecture:**

- **Convolutional actor network** processing 50×50 coverage grids[6]
- **Separate position encoder** for 3D position/orientation data[6]
- **Ornstein-Uhlenbeck noise** for exploration[6]
- **Target networks** with soft updates (τ=0.001)[6]

**Training Features:**

- **Experience replay buffer** (capacity: 100,000)[6]
- **Curriculum learning** with progressive environment size increases[6]
- **Model checkpointing** based on coverage performance[6]

#### Proximal Policy Optimization (PPO)

On-policy algorithm for stable policy learning[7]:

- **Clipped objective function** preventing large policy updates[7]
- **Rollout-based training** with configurable update intervals[7]
- **Combined actor-critic network** architecture[7]
- **Deterministic evaluation** using policy mean values[7]

#### Soft Actor-Critic (SAC)

Maximum entropy reinforcement learning for robust policies[8]:

**Key Components:**

- **Dual Q-networks** for value function approximation[8]
- **Squashed Gaussian policy** with learnable temperature parameter[8]
- **Automatic entropy tuning** for exploration-exploitation balance[8]
- **Target network soft updates** for stability[8]

**Implementation Details:**

- **Replay buffer integration** for off-policy learning[8]
- **State preprocessing** combining coverage and position data[8]
- **Action rescaling** to environment-specific ranges[8]

### Training and Evaluation

#### Training Infrastructure

Each algorithm includes comprehensive training scripts with:

- **Configurable hyperparameters** for environment and learning settings[9]
- **Progress monitoring** with coverage percentage and reward tracking[9]
- **Model persistence** with automatic best-model saving[9]
- **Rendering options** for training visualization[9]

#### Evaluation Framework

Systematic model evaluation includes[10]:

- **Deterministic policy execution** for consistent performance assessment[10]
- **Multi-episode testing** with statistical analysis[10]
- **Coverage percentage metrics** and reward accumulation tracking[10]
- **Visual demonstration** capabilities with environment rendering[10]

### Dependencies and Requirements

The project requires the following key dependencies[11]:

**Core Libraries:**

- `torch==2.6.0` - Deep learning framework
- `torchvision==0.21.0` - Computer vision utilities
- `gym==0.26.2` - Reinforcement learning environment interface
- `numpy==2.2.4` - Numerical computing
- `matplotlib==3.10.1` - Visualization and plotting
- `opencv-python==4.11.0.86` - Computer vision processing

**Additional Dependencies:**

- `cloudpickle==3.1.1` - Serialization utilities
- `scipy` (implied by dynamic wall generation) - Scientific computing

### Getting Started

#### Installation

```bash
git clone https://github.com/RL-cleaning-rob-cont-action-space/rl_vaccuum_cleaner.git
cd rl_vaccuum_cleaner
pip install -r requirements.txt
```

#### Basic Training

```bash
# Train DDPG on basic coverage
python bot_cleaner_basic_coverage/training/ddpg/training.py

# Train SAC with custom parameters
python bot_cleaner_basic_coverage/training/sac/training.py

# Train PPO for obstacle navigation
python bot_cleaner_static_obstacles/training/ppo/training.py
```

#### Model Evaluation

```bash
# Evaluate trained DDPG model
python bot_cleaner_basic_coverage/testing/ddpg/model_evaluation.py

# Test SAC performance
python bot_cleaner_basic_coverage/testing/sac/model_evaluation.py
```

#### Advanced Training (Prioritized Cleaning)

```bash
# Quick test run
python bot_cleaner_prioritized_cleaning/training/sac/training.py --render

# Production training
python bot_cleaner_prioritized_cleaning/training/sac/training.py \
    --total_steps 1000000 \
    --eval_interval 10000 \
    --save_interval 50000 \
    --max_steps 1000 \
    --env_size_x 15 \
    --env_size_y 15 \
    --dirt_density 0.4 \
    --no_render
```

### Project Results

The project demonstrates successful implementation of multiple RL algorithms for autonomous cleaning tasks[12]. Training results show progressive learning curves with agents achieving over 95% coverage efficiency in basic scenarios[12]. The modular architecture enables systematic comparison of algorithm performance across different environmental complexities[12].

**Key Achievements:**

- **Multi-algorithm implementation** with DDPG, PPO, and SAC variants
- **Scalable environment complexity** from basic coverage to dynamic obstacle navigation
- **Comprehensive evaluation framework** with quantitative performance metrics
- **Extensible architecture** supporting future enhancements like vision-based navigation

The project serves as a practical demonstration of deep reinforcement learning applications in robotics, providing a foundation for autonomous cleaning system development and algorithm comparison in continuous control domains.
