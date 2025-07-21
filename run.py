import math
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from math import atan2, degrees
from fractions import Fraction

matplotlib.use('Agg')  # Non-interactive backend for saving images

def dot_coords(n):
    """Map dot numbers to grid coordinates."""
    return {i * n + j + 1: (i, j) for i in range(n) for j in range(n)}

def dots_between(a, b, grid_size):
    """Return intermediate dots between a and b (exclusive) in straight line."""
    (r1, c1), (r2, c2) = a, b
    dr, dc = r2 - r1, c2 - c1
    gcd = math.gcd(dr, dc)
    if gcd <= 1:
        return []
    step_r, step_c = dr // gcd, dc // gcd
    return [(r1 + step_r * k, c1 + step_c * k) for k in range(1, gcd)]

def dfs(visited, current, path, all_paths, grid_size):
    if len(path) >= 4:
        all_paths.append(path[:])
    if len(path) == grid_size ** 2:
        return  # Full grid

    for next_dot in visited.keys():
        if not visited[next_dot]:
            r1, c1 = dot_coords(grid_size)[current]
            r2, c2 = dot_coords(grid_size)[next_dot]
            between = dots_between((r1, c1), (r2, c2), grid_size)
            if all(visited[coord_to_dot(pos, grid_size)] for pos in between):
                visited[next_dot] = True
                path.append(next_dot)
                dfs(visited, next_dot, path, all_paths, grid_size)
                path.pop()
                visited[next_dot] = False

def coord_to_dot(coord, grid_size):
    """Convert (row, col) to dot number."""
    return coord[0] * grid_size + coord[1] + 1

def generate_passwords(grid_size=3):
    coords_map = dot_coords(grid_size)
    all_paths = []
    visited = {dot: False for dot in coords_map}
    for start in coords_map:
        visited[start] = True
        dfs(visited, start, [start], all_paths, grid_size)
        visited[start] = False
    # Convert dot numbers to grid coordinates
    grid_paths = [[coords_map[dot] for dot in path] for path in all_paths]
    return grid_paths

# Complexity scoring: turns + long jumps
def compute_complexity(path):
    score = 0
    for i in range(1, len(path) - 1):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        x3, y3 = path[i + 1]

        # Angle change
        angle1 = atan2(y2 - y1, x2 - x1)
        angle2 = atan2(y3 - y2, x3 - x2)
        if degrees(angle1 - angle2) % 360 not in [0, 180]:
            score += 1  # Change in direction

        # Long move detection
        if abs(x1 - x2) > 1 or abs(y1 - y2) > 1:
            score += 2  # Long step = more complex

    return score

def get_most_complex_passwords(passwords, top_n=5):
    scored = [(path, compute_complexity(path)) for path in passwords]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [path for path, _ in scored[:top_n]]

def save_paths_as_images(passwords, grid_size=3, folder="output"):
    timestamp = str(int(time.time()))  # Timestamp folder
    output_folder = os.path.join(folder, timestamp)
    os.makedirs(output_folder, exist_ok=True)
    
    for i, path in enumerate(passwords):
        x_coords = [point[1] for point in path]
        y_coords = [grid_size - 1 - point[0] for point in path]  # Flip y for plotting

        plt.figure(figsize=(6, 6))
        plt.scatter(x_coords, y_coords, color='black', zorder=5)
        plt.plot(x_coords, y_coords, color='black', marker='o', zorder=4)

        # Labels
        for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.text(x + 0.1, y + 0.1, f'{idx+1}', fontsize=9, color='black')

        plt.xlim(-0.5, grid_size - 0.5)
        plt.ylim(-0.5, grid_size - 0.5)
        plt.axis('off')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Password {i+1} Visualization", fontsize=10)
        image_filename = os.path.join(output_folder, f"password_{i+1}.png")
        plt.savefig(image_filename, bbox_inches='tight')
        plt.close()
        print(f"Saved image: {image_filename}")

# Main
if __name__ == "__main__":
    grid_size = int(input("What size grid do you want to calculate passwords for?: "))  
    print(f"Generating patterns for {grid_size}x{grid_size} grid...")
    passwords = generate_passwords(grid_size)
    print(f"Generated {len(passwords)} total patterns.")

    most_complex_passwords = get_most_complex_passwords(passwords, top_n=5)
    print(f"Top {len(most_complex_passwords)} most complex patterns:")
    for password in most_complex_passwords:
        print(password)

    save_paths_as_images(most_complex_passwords, grid_size)
    print(f"Total patterns considered: {len(passwords)}")
