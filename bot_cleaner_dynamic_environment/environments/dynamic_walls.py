import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import zoom

def generate_dynamic_walls(size=1, wall_density=1):
    grid = np.zeros((size, size), dtype=int)  # 0 = Walkable, -1 = Wall
    num_walls = int(size * size * wall_density)

    # Randomly place walls
    for _ in range(num_walls):
        x, y = random.randint(0, size-1), random.randint(0, size-1)
        grid[y, x] = -1  # Set as a wall

    def is_reachable(grid):
        visited = np.zeros_like(grid)
        stack = [(0, 0)]  # Start from (0,0) assuming it's walkable
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= size or y >= size:
                continue
            if visited[y, x] or grid[y, x] == -1:
                continue
            visited[y, x] = 1
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])  # Move in 4 directions

        return np.sum((visited == 1) & (grid == 0)) == np.sum(grid == 0)  # All empty spaces reachable

    # Keep modifying grid until it's fully reachable
    while not is_reachable(grid):
        grid = np.zeros((size, size), dtype=int)
        for _ in range(num_walls):
            x, y = random.randint(0, size-1), random.randint(0, size-1)
            grid[y, x] = -1

    return grid


# Visualization
def visualize_walls(grid):
    size = len(grid)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xticks([])
    ax.set_yticks([])
    
    for y in range(size):
        for x in range(size):
            if grid[y, x] == 1:
                ax.add_patch(Rectangle((x, size-y-1), 1, 1, color="black"))  # Draw wall
            elif grid[y,x] == -1:
                ax.add_patch(Rectangle((x, size-y-1), 1, 1, color="blue"))

    plt.xlim(0, size)
    plt.ylim(0, size)
    plt.gca().set_aspect("equal")
    plt.show()
    return plt

def scale_grid(grid, new_resolution):
    """Scale the grid to a new resolution without interpolation (just resizing)."""
    current_resolution = grid.shape[0]
    # Create a new grid of the new resolution (initialize with zeros or the default value)
    new_grid = np.zeros((new_resolution, new_resolution), dtype=grid.dtype)
    
    # Calculate the scaling factor for rows and columns
    row_scale = new_resolution / current_resolution
    col_scale = new_resolution / current_resolution
    
    for i in range(new_resolution):
        for j in range(new_resolution):
            # Map the indices from the new grid to the old grid
            old_i = int(i / row_scale)
            old_j = int(j / col_scale)
            
            # Ensure we're within bounds of the old grid
            old_i = min(old_i, current_resolution - 1)
            old_j = min(old_j, current_resolution - 1)
            
            # Copy the value from the original grid to the new grid
            new_grid[i, j] = grid[old_i, old_j]
    
    return new_grid

def get_walls(main_grid_size, wall_size):
    walls = generate_dynamic_walls(size=wall_size, wall_density=0.3)
    # visualize_walls(walls)
    scaled_walls = scale_grid(walls, int(main_grid_size))
    # visualize_walls(scaled_walls)
    # print(f"grid shape : {walls.shape}")
    # print(f"grid shape : {scaled_walls.shape}")
    return scaled_walls

if __name__ == "__main__":
    get_walls(5,1)