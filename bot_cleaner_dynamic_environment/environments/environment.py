import math
import gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import random
from bot_cleaner_dynamic_environment.environments.dynamic_walls import get_walls,visualize_walls

class ContinuousVacuumCleanerEnv(gym.Env):
    
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, size=10.0, resolution=50, coverage_radius=0.5, max_steps=500):
        super().__init__()
        self.size = size
        self.resolution = resolution
        self.coverage_radius = coverage_radius
        self.max_steps = max_steps
        self.cell_size = size / resolution
        self.max_linear_velocity = 0.5
        self.max_angular_velocity = np.pi/2
        self.dt = 0.1
        self.wall_width = 1

        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_velocity, -self.max_angular_velocity]),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity]),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([0, 0, -np.pi]),
                high=np.array([size, size, np.pi]),
                dtype=np.float32
            ),
            "coverage": spaces.Box(
                low=0, high=1, shape=(resolution * resolution,), dtype=np.float32
            )
        })

        self.fig = None
        self.ax = None
        self.reset()

    def reset(self):
        # self.coverage_grid = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        self.coverage_grid= get_walls(main_grid_size=self.resolution, wall_size=3)

        # Keep trying random positions until a free spot is found
        while True:
            x = np.random.uniform(0, self.size)
            y = np.random.uniform(0, self.size)
            grid_x = int(x / self.cell_size)
            grid_y = int(y / self.cell_size)
            if self.coverage_grid[grid_y, grid_x] == 0:
                break

        self.agent_position = np.array([x, y], dtype=np.float32)
        self.agent_orientation = np.random.uniform(-np.pi, np.pi)

        self.steps = 0
        self.coverage_percentage = 0.0
        self._update_coverage()
        return self._get_observation()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        lin_vel, ang_vel = action
        # print(action)
        new_theta = ((self.agent_orientation + ang_vel * self.dt + np.pi) % (2 * np.pi)) - np.pi

        delta_x = lin_vel * math.cos(new_theta) * self.dt
        delta_y = lin_vel * math.sin(new_theta) * self.dt
        new_pos = self.agent_position + np.array([delta_x, delta_y])

        # Convert to grid cell
        grid_x = int(new_pos[0] / self.cell_size)
        grid_y = int(new_pos[1] / self.cell_size)

        # Check wall collision
        collision = (
            grid_x < 0 or grid_x >= self.resolution or
            grid_y < 0 or grid_y >= self.resolution or
            self.coverage_grid[grid_y, grid_x] == -1
        )

        if not collision:
            self.agent_position = np.clip(new_pos, [0, 0], [self.size, self.size])
            self.agent_orientation = new_theta
            newly_covered, _ = self._update_coverage()
            reward = self._calculate_reward(newly_covered)
        else:
            reward = -10.0  # Negative reward for bumping into a wall
            done = True

        self.steps += 1
        done = self.coverage_percentage >= 0.95 or self.steps >= self.max_steps
        # plt = visualize_walls(self.coverage_grid)
        # plt.pause(0.2)
        # plt.close()
        return self._get_observation(), reward, done, {
            "coverage_percentage": self.coverage_percentage,
            "steps": self.steps
        }

    def _update_coverage(self):
        agent_x, agent_y = self.agent_position
        min_x = max(0, int((agent_x - self.coverage_radius) / self.cell_size))
        max_x = min(self.resolution - 1, int((agent_x + self.coverage_radius) / self.cell_size))
        min_y = max(0, int((agent_y - self.coverage_radius) / self.cell_size))
        max_y = min(self.resolution - 1, int((agent_y + self.coverage_radius) / self.cell_size))

        newly_covered = 0
        for i in range(min_x, max_x + 1):
            for j in range(min_y, max_y + 1):
                cell_x = (i + 0.5) * self.cell_size
                cell_y = (j + 0.5) * self.cell_size
                distance = (cell_x - agent_x)**2 + (cell_y - agent_y)**2

                if distance <= self.coverage_radius**2:
                    if self.coverage_grid[j, i] == 0:
                        self.coverage_grid[j, i] = 1  # Mark as cleaned
                        newly_covered += 1

        # Exclude wall cells (-1) when computing coverage percentage
        cleanable_cells = np.sum(self.coverage_grid != -1)
        cleaned_cells = np.sum(self.coverage_grid == 1)
        self.coverage_percentage = cleaned_cells / cleanable_cells if cleanable_cells > 0 else 0

        return newly_covered, (max_x - min_x + 1) * (max_y - min_y + 1)


    def _calculate_reward(self, newly_covered):
        center = np.array([self.size/2, self.size/2])
        dist_from_center = np.linalg.norm(self.agent_position - center)
        recent_coverage = np.mean(self.coverage_grid[
            max(0, int((self.agent_position[1]-1)/self.cell_size)) :
            min(self.resolution, int((self.agent_position[1]+1)/self.cell_size)),
            max(0, int((self.agent_position[0]-1)/self.cell_size)) :
            min(self.resolution, int((self.agent_position[0]+1)/self.cell_size))
        ])
        
        # Determine current grid cell
        cell_x = np.clip(int(self.agent_position[0] / self.cell_size), 0, self.resolution - 1)
        cell_y = np.clip(int(self.agent_position[1] / self.cell_size), 0, self.resolution - 1)
        # Check if agent is staying in a covered cell (not wall)
        is_in_covered_cell = self.coverage_grid[cell_y, cell_x] == 1

        

        return (
            newly_covered * 2.0
            # dist_from_center * 0.05 +
            # +(1 - recent_coverage) * 0.1 
            -0.1 * (newly_covered == 0) 
            - 0.5 * is_in_covered_cell
            -0.005 
            +(100 if self.coverage_percentage >= 0.95 else 0)
        )

    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()
        self.ax.clear()

        # Ensure discrete integer values for rendering
        display_grid = np.full_like(self.coverage_grid, 0)  # Start with all uncleaned = 0 (white)
        display_grid[self.coverage_grid == 1] = 1            # Cleaned = 1 (blue)
        display_grid[self.coverage_grid == -1] = 2           # Wall = 2 (black)

        from matplotlib import colors
        cmap = colors.ListedColormap(["white", "blue", "black"])  # 0, 1, 2
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        self.ax.imshow(
            display_grid,
            extent=[0, self.size, 0, self.size],
            origin="lower",
            cmap=cmap,
            norm=norm
        )

        # Draw agent
        agent_circle = Circle(
            self.agent_position,
            radius=self.coverage_radius,
            facecolor="red",
            alpha=0.5,
            edgecolor="black"
        )
        self.ax.add_patch(agent_circle)

        # Reference points (optional)
        # self.ax.add_patch(Circle((0, 0), radius=0.1))
        # self.ax.add_patch(Circle((5, 5), radius=0.1))
        # self.ax.add_patch(Circle((10, 10), radius=0.1))

        # Draw orientation
        end_x = self.agent_position[0] + self.coverage_radius * 1.5 * math.cos(self.agent_orientation)
        end_y = self.agent_position[1] + self.coverage_radius * 1.5 * math.sin(self.agent_orientation)
        self.ax.plot([self.agent_position[0], end_x], [self.agent_position[1], end_y], 'k-')

        # Display text
        self.ax.text(0.05, 0.95,
                    f"Coverage: {self.coverage_percentage:.2%}\nSteps: {self.steps}/{self.max_steps}",
                    transform=self.ax.transAxes,
                    fontsize=12,
                    verticalalignment='top')

        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        plt.pause(0.01)


    def close(self):
        if self.fig:
            plt.close(self.fig)
            plt.ioff()


    def _get_observation(self):
        return {
            "position": np.array([
                self.agent_position[0],
                self.agent_position[1],
                self.agent_orientation
            ], dtype=np.float32),
            "coverage": self.coverage_grid.flatten()
        }


class SimpleWallEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, size=5.0, resolution=50, coverage_radius=0.5, max_steps=2000):
        super().__init__()
        self.size = size
        self.resolution = resolution
        self.coverage_radius = coverage_radius
        self.max_steps = max_steps
        self.cell_size = size / resolution
        self.max_linear_velocity = 0.5
        self.max_angular_velocity = np.pi / 2
        self.dt = 0.1
        self.steps = 0

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_velocity, -self.max_angular_velocity]),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity]),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            "position": spaces.Box(
                low=np.array([0, 0, -np.pi]),
                high=np.array([size, size, np.pi]),
                dtype=np.float32
            ),
            "coverage": spaces.Box(
                low=0, high=1, shape=(resolution * resolution,), dtype=np.float32
            )
        })

        # Define walls (list of line segments [(x1, y1, x2, y2)])
        self.walls = [
            (1.0, 1.0, 4.0, 1.0),  # Horizontal wall
            (1.0, 3.0, 4.0, 3.0),  # Another horizontal wall
            (2.0, 1.0, 2.0, 3.0)   # Vertical wall
        ]

        self.fig = None
        self.ax = None
        self.reset()

    def reset(self):
        self.position = np.array([self.size / 2, self.size / 2, 0])  # Start in the center
        self.coverage = np.zeros((self.resolution * self.resolution,))
        self.steps = 0
        return self.get_observation()

    def step(self, action):
        linear_velocity, angular_velocity = action
        new_x = self.position[0] + linear_velocity * np.cos(self.position[2]) * self.dt
        new_y = self.position[1] + linear_velocity * np.sin(self.position[2]) * self.dt
        new_theta = self.position[2] + angular_velocity * self.dt

        # Check for wall collision
        if not self.check_collision(self.position[:2], (new_x, new_y)):
            self.position = np.array([new_x, new_y, new_theta])

        self.steps += 1
        done = self.steps >= self.max_steps
        reward = self.calculate_reward()
        return self.get_observation(), reward, done, {}

    def check_collision(self, old_pos, new_pos):
        x1, y1 = old_pos
        x2, y2 = new_pos

        for (wx1, wy1, wx2, wy2) in self.walls:
            if self.line_intersection((x1, y1, x2, y2), (wx1, wy1, wx2, wy2)):
                return True
        return False

    def line_intersection(self, line1, line2):
        """ Check if two line segments intersect. """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and \
               ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4))

    def calculate_reward(self):
        return np.sum(self.coverage) / len(self.coverage)

    def get_observation(self):
        return {"position": self.position, "coverage": self.coverage}

    def render(self, mode="human"):
        if self.fig is None:
            plt.ion()  # Turn on interactive mode
            self.fig, self.ax = plt.subplots(figsize=(5, 5))

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)

        # Draw walls
        for (x1, y1, x2, y2) in self.walls:
            self.ax.plot([x1, x2], [y1, y2], "k-", linewidth=3, label="Wall")

        # Draw cleaner
        self.ax.plot(self.position[0], self.position[1], "ro", markersize=8, label="Cleaner")

        # Add labels
        self.ax.set_title("Vacuum Cleaner Environment")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to update


    def close(self):
        if self.fig:
            plt.close(self.fig)