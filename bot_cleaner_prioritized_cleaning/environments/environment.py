import math
import time

import gym
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces


class EnhancedVacuumCleanerEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        size_x=10,
        size_y=10,
        coverage_radius=1.0,
        max_steps=1000,
        env_type="empty",
        dirt_percentage=0.1,
        dirt_reward_multiplier=3.0,
        random_seed=None,
    ):
        """
        Enhanced Vacuum Cleaner Environment with dirt and obstacle support.

        Args:
            size_x: Width of the environment
            size_y: Height of the environment
            coverage_radius: Radius of the vacuum cleaner coverage
            max_steps: Maximum steps per episode
            env_type: Type of environment - "empty" or "maze"
            dirt_percentage: Percentage of non-obstacle cells that contain dirt
            dirt_reward_multiplier: Multiplier for reward when cleaning dirt
            random_seed: Seed for random number generation
        """
        super().__init__()
        self.size_x = size_x
        self.size_y = size_y
        self.coverage_radius = coverage_radius
        self.max_steps = max_steps
        self.env_type = env_type
        self.dirt_percentage = dirt_percentage
        self.dirt_reward_multiplier = dirt_reward_multiplier

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Define maze layout based on env_type
        self._define_layout()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-0.5, -np.pi / 2]),
            high=np.array([0.5, np.pi / 2]),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=np.array([0, 0, -np.pi]),
                    high=np.array([size_x, size_y, np.pi]),
                    dtype=np.float32,
                ),
                "coverage": spaces.Box(
                    low=0, high=1, shape=(size_x * size_y,), dtype=np.float32
                ),
                "walls": spaces.Box(
                    low=0, high=1, shape=(size_x * size_y,), dtype=np.float32
                ),
                "dirt": spaces.Box(
                    low=0, high=1, shape=(size_x * size_y,), dtype=np.float32
                ),
            }
        )

        # For rendering
        self.fig = None
        self.ax = None
        self.coverage_path = []
        self.coverage_patches = []

    def _define_layout(self):
        """Define the environment layout based on env_type"""
        # Create wall grid
        self.wall_grid = np.zeros((self.size_y, self.size_x), dtype=np.int32)

        # Set up walls for maze environment
        if self.env_type == "maze":
            # Add vertical walls with gaps
            wall_positions = [int(self.size_x * 0.3), int(self.size_x * 0.7)]
            gap_start = self.size_y // 2 - 2
            gap_height = 4
            for x in wall_positions:
                for y in range(self.size_y):
                    if y < gap_start or y >= gap_start + gap_height:
                        self.wall_grid[y, x] = 1

            # Add a horizontal wall with gap
            horizontal_wall_y = self.size_y // 2
            gap_start = self.size_x // 2 - 2
            gap_width = 4
            for x in range(self.size_x):
                if x < gap_start or x >= gap_start + gap_width:
                    self.wall_grid[horizontal_wall_y, x] = 1

        # Start and exit positions
        self.start_pos = np.array([1.0, 1.0])
        self.exit_pos = np.array([self.size_x - 2.0, self.size_y - 2.0])

    def _create_dirt(self):
        """Generate dirt in the environment"""
        self.dirt_grid = np.zeros((self.size_y, self.size_x), dtype=np.int32)

        # Count valid (non-wall) cells
        valid_cells = np.argwhere(self.wall_grid == 0)
        num_valid_cells = len(valid_cells)

        # Calculate number of dirt cells
        num_dirt_cells = int(num_valid_cells * self.dirt_percentage)

        # Randomly select dirt cells from valid cells
        if num_dirt_cells > 0:
            dirt_indices = np.random.choice(
                num_valid_cells, num_dirt_cells, replace=False
            )
            for idx in dirt_indices:
                y, x = valid_cells[idx]
                self.dirt_grid[y, x] = 1

        # Track how much dirt has been cleaned
        self.total_dirt = num_dirt_cells
        self.cleaned_dirt = 0

    def reset(self):
        """Reset the environment"""
        self.agent_position = self.start_pos.copy()
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)
        self.coverage_grid = np.zeros((self.size_y, self.size_x), dtype=np.float32)
        self.coverage_path = []
        self.coverage_patches = []
        self.steps = 0
        self.coverage_percentage = 0.0

        # Create new dirt configuration
        self._create_dirt()
        self.dirt_cleaned_percentage = 0.0

        return self._get_observation()

    def _is_valid_move(self, position):
        """Check if the proposed position is valid (within bounds and not in a wall)"""
        # First check exact boundary conditions (continuous space)
        if (position[0] < 0 or position[0] >= self.size_x or 
            position[1] < 0 or position[1] >= self.size_y):
            return False
            
        # Now check for walls using grid coordinates
        x, y = int(position[0]), int(position[1])
        
        # Double-check to be safe
        if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
            return False

        # Check wall collision
        if self.wall_grid[y, x] == 1:
            return False

        return True

    def step(self, action):
        """Take a step in the environment"""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        lin_vel, ang_vel = action

        # Update orientation
        self.agent_orientation = (
            (self.agent_orientation + ang_vel * 0.1 + np.pi) % (2 * np.pi)
        ) - np.pi

        # Calculate potential new position
        delta_x = lin_vel * math.cos(self.agent_orientation) * 0.1
        delta_y = lin_vel * math.sin(self.agent_orientation) * 0.1
        new_pos = self.agent_position + np.array([delta_x, delta_y])
        
        # Clip the position to be within boundaries before checking validity
        new_pos[0] = np.clip(new_pos[0], 0, self.size_x - 0.001)
        new_pos[1] = np.clip(new_pos[1], 0, self.size_y - 0.001)

        # Validate move
        if self._is_valid_move(new_pos):
            self.agent_position = new_pos
            reward = 0.1  # Small positive reward for valid movement
        else:
            reward = -0.5  # Penalty for invalid move

        # Update coverage and dirt
        newly_cleaned_dirt = self._update_coverage()

        # Add reward for cleaning dirt
        if newly_cleaned_dirt > 0:
            reward += newly_cleaned_dirt * self.dirt_reward_multiplier

        self.steps += 1

        # Check if the episode is done
        done = (
            (self.coverage_percentage >= 0.95 and self.dirt_cleaned_percentage >= 0.95)
            or self.steps >= self.max_steps
            or np.linalg.norm(self.agent_position - self.exit_pos) < 1.0
        )

        return (
            self._get_observation(),
            reward,
            done,
            {
                "coverage_percentage": self.coverage_percentage,
                "dirt_cleaned_percentage": self.dirt_cleaned_percentage,
                "steps": self.steps,
            },
        )

    def _update_coverage(self):
        """Update coverage grid and clean dirt"""
        newly_cleaned_dirt = 0

        # Add current position to coverage path
        self.coverage_path.append(
            (self.agent_position[0], self.agent_position[1], self.coverage_radius)
        )

        # Create coverage patch for rendering
        coverage_circle = patches.Circle(
            (self.agent_position[0], self.agent_position[1]),
            radius=self.coverage_radius,
            color="blue",
            alpha=0.3,
        )
        self.coverage_patches.append(coverage_circle)

        # Calculate coverage area
        x_center, y_center = int(self.agent_position[0]), int(self.agent_position[1])
        radius_cells = int(self.coverage_radius + 0.5)  # Round up radius to cells

        # Update coverage grid and clean dirt in radius
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                nx, ny = x_center + dx, y_center + dy

                # Check if within bounds and in radius
                if (
                    0 <= nx < self.size_x
                    and 0 <= ny < self.size_y
                    and self.wall_grid[ny, nx] == 0
                    and dx**2 + dy**2 <= radius_cells**2
                ):

                    # Update coverage
                    self.coverage_grid[ny, nx] = 1

                    # Clean dirt if present
                    if self.dirt_grid[ny, nx] == 1:
                        self.dirt_grid[ny, nx] = 0
                        newly_cleaned_dirt += 1
                        self.cleaned_dirt += 1

        # Calculate coverage and dirt percentages
        non_wall_cells = np.sum(self.wall_grid == 0)
        self.coverage_percentage = np.sum(self.coverage_grid) / non_wall_cells

        if self.total_dirt > 0:
            self.dirt_cleaned_percentage = self.cleaned_dirt / self.total_dirt
        else:
            self.dirt_cleaned_percentage = 1.0

        return newly_cleaned_dirt

    def render(self, mode="human"):
        """Render the environment"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            plt.ion()

        self.ax.clear()

        # Set background to white
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")

        # Draw walls as gray rectangles
        for y in range(self.size_y):
            for x in range(self.size_x):
                if self.wall_grid[y, x] == 1:
                    wall = patches.Rectangle(
                        (x, y), 1, 1, facecolor=[0.5, 0.5, 0.5], edgecolor="none"
                    )
                    self.ax.add_patch(wall)

        # Draw dirt as brown squares
        for y in range(self.size_y):
            for x in range(self.size_x):
                if self.dirt_grid[y, x] == 1:
                    dirt = patches.Rectangle(
                        (x + 0.2, y + 0.2),
                        0.6,
                        0.6,
                        facecolor="brown",
                        edgecolor="none",
                        alpha=0.7,
                    )
                    self.ax.add_patch(dirt)

        # Draw coverage
        for patch in self.coverage_patches:
            self.ax.add_patch(patch)

        # Draw agent
        agent_circle = patches.Circle(
            self.agent_position,
            radius=self.coverage_radius,
            facecolor="red",
            alpha=0.5,
            edgecolor="black",
        )
        self.ax.add_patch(agent_circle)

        # Draw orientation line
        end_x = self.agent_position[0] + self.coverage_radius * 2 * math.cos(
            self.agent_orientation
        )
        end_y = self.agent_position[1] + self.coverage_radius * 2 * math.sin(
            self.agent_orientation
        )
        self.ax.plot(
            [self.agent_position[0], end_x],
            [self.agent_position[1], end_y],
            color="black",
            linewidth=2,
        )

        # Display info text
        self.ax.text(
            0.05,
            0.95,
            f"Coverage: {self.coverage_percentage:.2%}\n"
            f"Dirt Cleaned: {self.dirt_cleaned_percentage:.2%}\n"
            f"Steps: {self.steps}/{self.max_steps}",
            transform=self.ax.transAxes,
            fontsize=10,
            color="black",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7),
        )

        self.ax.set_xlim(0, self.size_x)
        self.ax.set_ylim(0, self.size_y)
        self.ax.set_aspect("equal")
        plt.pause(0.01)

    def _get_observation(self):
        """Get the current observation"""
        return {
            "position": np.array(
                [
                    self.agent_position[0],
                    self.agent_position[1],
                    self.agent_orientation,
                ],
                dtype=np.float32,
            ),
            "coverage": self.coverage_grid.flatten(),
            "walls": self.wall_grid.flatten(),
            "dirt": self.dirt_grid.flatten(),
        }

    def close(self):
        """Close the environment"""
        if self.fig:
            plt.close(self.fig)
            plt.ioff()


def main():
    """Test the environment with random actions"""
    # Test maze environment with dirt
    env = EnhancedVacuumCleanerEnv(
        size_x=15,
        size_y=10,
        coverage_radius=1.0,
        max_steps=500,
        env_type="maze",
        dirt_percentage=0.2,
        dirt_reward_multiplier=3.0,
    )

    obs = env.reset()
    for _ in range(500):  # Run for 500 steps max
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.05)  # Add delay to visualize better

        if done:
            print(
                f"Episode finished!\n"
                f"Coverage: {info['coverage_percentage']:.2%}\n"
                f"Dirt Cleaned: {info['dirt_cleaned_percentage']:.2%}\n"
                f"Steps: {info['steps']}"
            )
            break

    plt.show()  # Keep the final render visible
    env.close()


if __name__ == "__main__":
    main()
