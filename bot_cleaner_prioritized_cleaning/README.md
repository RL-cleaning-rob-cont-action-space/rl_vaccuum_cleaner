# Vacuum Cleaner Reinforcement Learning

This project implements a Soft Actor-Critic (SAC) reinforcement learning algorithm for a vacuum cleaner agent that learns to efficiently clean a space while prioritizing dirt-prone areas.

## Training the SAC Agent

The training script is located at `bot_cleaner_prioritized_cleaning/training/sac/training.py` and can be run with various command-line arguments to configure the training process.

### Basic Usage

```bash
python bot_cleaner_prioritized_cleaning/training/sac/training.py
```

By default, this will run a 100-step test training with rendering enabled.

### Command-Line Arguments

#### Environment Configuration
- `--env_size_x`: Width of the environment (default: 10)
- `--env_size_y`: Height of the environment (default: 10)
- `--wall_density`: Density of walls - values > 0 create a maze environment (default: 0.1)
- `--dirt_density`: Initial density of dirt in the environment (default: 0.3)
- `--dirt_spawn_rate`: Rate at which new dirt spawns (not used in current implementation) (default: 0.01)
- `--prioritize_dirt`: Reward multiplier for cleaning dirt (default: 2.0)
- `--coverage_radius`: Radius of the vacuum cleaner coverage area (default: 1.0)
- `--max_steps`: Maximum steps per episode (default: 100)

#### Training Control
- `--total_steps`: Total number of training steps (default: 100)
- `--eval_interval`: Interval between evaluations (default: 50)
- `--save_interval`: Interval between model checkpoints (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)

#### Output Options
- `--model_dir`: Directory to save models (default: project_root/models)
- `--log_dir`: Directory to save logs (default: project_root/logs)
- `--render`: Enable environment rendering during training
- `--no_render`: Disable rendering (overrides --render)

### Training Examples

#### Quick Test Run (100 steps with rendering)
```bash
python bot_cleaner_prioritized_cleaning/training/sac/training.py --env_size_x 15 --env_size_y 15 --dirt_density 0.4
```

#### Short Training (1,000 steps)
```bash
python bot_cleaner_prioritized_cleaning/training/sac/training.py \
  --total_steps 1000 \
  --eval_interval 200 \
  --save_interval 500 \
  --max_steps 200 \
  --env_size_x 15 \
  --env_size_y 15 \
  --dirt_density 0.4 \
  --no_render
```

#### Medium Training (50,000 steps)
```bash
python bot_cleaner_prioritized_cleaning/training/sac/training.py \
  --total_steps 50000 \
  --eval_interval 2000 \
  --save_interval 10000 \
  --max_steps 500 \
  --env_size_x 15 \
  --env_size_y 15 \
  --dirt_density 0.4 \
  --no_render
```

#### Long Training (1,000,000 steps)
```bash
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

### Tips for Efficient Training

1. **Disable Rendering for Long Runs**: Always use `--no_render` for anything beyond test runs, as rendering significantly slows down training.

2. **Adjust Evaluation Frequency**: For longer runs, use larger `--eval_interval` values to reduce overhead from frequent evaluations.

3. **Optimize Episode Length**: Set `--max_steps` according to the complexity of your environment. Larger environments may need longer episodes.

4. **Prioritize Dirt Cleaning**: Increase `--prioritize_dirt` value (e.g., 3.0 or higher) if you want the agent to focus more on cleaning dirt rather than exploration.

5. **Environment Size**: Larger environments (`--env_size_x` and `--env_size_y`) are more challenging and require longer training.

## Saved Models

Models are saved to the `bot_cleaner_prioritized_cleaning/models/` directory with the following naming convention:

- `sac_best_model.pt`: The model with the highest evaluation reward
- `sac_checkpoint_X_TIMESTAMP.pt`: Periodic checkpoints at step X
- `sac_final_model.pt`: The final model after training completes

## Viewing Results

During training, the script will print:
- Episode rewards
- Coverage percentage
- Dirt cleaned percentage
- Evaluation results

For longer runs, monitor the model saving messages to see when new checkpoints are created. 