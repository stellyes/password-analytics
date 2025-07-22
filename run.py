import math
import os
import time
from math import atan2, degrees, sqrt
from multiprocessing import Pool, cpu_count, current_process
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# ---------------------------
# GRID UTILS
# ---------------------------
def dot_coords(n):
    return {i * n + j + 1: (i, j) for i in range(n) for j in range(n)}

def coord_to_dot(coord, grid_size):
    return coord[0] * grid_size + coord[1] + 1

def dots_between(a, b, grid_size):
    (r1, c1), (r2, c2) = a, b
    dr, dc = r2 - r1, c2 - c1
    gcd = math.gcd(dr, dc)
    if gcd <= 1:
        return []
    step_r, step_c = dr // gcd, dc // gcd
    return [(r1 + step_r * k, c1 + step_c * k) for k in range(1, gcd)]

# ---------------------------
# COMPLEXITY SCORING
# ---------------------------
def compute_complexity(path):
    score = 0.0
    for i in range(1, len(path) - 1):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        x3, y3 = path[i + 1]

        # Angle sharpness (larger angles = more complex)
        angle1 = atan2(y2 - y1, x2 - x1)
        angle2 = atan2(y3 - y2, x3 - x2)
        angle_diff = abs(degrees(angle1 - angle2)) % 360
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        score += angle_diff / 45  # Normalize (every 45° adds 1 point)

        # Line length factor (longer lines = more complex)
        dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        score += dist
    return score

# ---------------------------
# DFS WORKER
# ---------------------------
def dfs_worker(args):
    start_dot, grid_size, coords_map, current_max_complexity, progress_interval = args
    visited = {dot: False for dot in coords_map}
    visited[start_dot] = True
    local_top = []
    local_max_complexity = current_max_complexity
    path_count = 0
    total_permutations = math.factorial(grid_size ** 2 - 1)

    def dfs(current, path):
        nonlocal path_count, local_max_complexity, local_top
        if len(path) == grid_size ** 2:
            coords_path = [coords_map[dot] for dot in path]
            complexity = compute_complexity(coords_path)

            if complexity > local_max_complexity:
                local_max_complexity = complexity
                local_top = [(complexity, coords_path)]  # Clear and store only new max
            elif complexity == local_max_complexity:
                local_top.append((complexity, coords_path))

            path_count += 1
            if path_count % progress_interval == 0:
                percent = (path_count / total_permutations) * 100
                os.system('cls' if os.name == 'nt' else 'clear')  # Clear console
                print(f"[{current_process().name}] Progress: {percent:.2f}% | "
                      f"Patterns Checked: {path_count:,}")
            return

        for next_dot in visited:
            if not visited[next_dot]:
                r1, c1 = coords_map[current]
                r2, c2 = coords_map[next_dot]
                between = dots_between((r1, c1), (r2, c2), grid_size)
                if all(visited[coord_to_dot(p, grid_size)] for p in between):
                    visited[next_dot] = True
                    path.append(next_dot)
                    dfs(next_dot, path)
                    path.pop()
                    visited[next_dot] = False

    dfs(start_dot, [start_dot])
    print(f"[{current_process().name}] Finished. {path_count} full-grid paths explored.")
    return local_top, local_max_complexity

# ---------------------------
# GENERATE PASSWORDS
# ---------------------------
def generate_passwords(grid_size=3, progress_interval=100000):
    coords_map = dot_coords(grid_size)
    dots = list(coords_map.keys())
    current_max_complexity = -float('inf')
    args_list = [(dot, grid_size, coords_map, current_max_complexity, progress_interval) for dot in dots]

    with Pool(processes=min(cpu_count(), len(dots))) as pool:
        worker_results = pool.map(dfs_worker, args_list)

    global_top = []
    global_max_complexity = -float('inf')
    for worker_top, worker_max in worker_results:
        if worker_max > global_max_complexity:
            global_max_complexity = worker_max
            global_top = worker_top[:]
        elif worker_max == global_max_complexity:
            global_top.extend(worker_top)

    global_top.sort(reverse=True)
    return [path for _, path in global_top], global_max_complexity

# ---------------------------
# RENDER IMAGES
# ---------------------------
def save_paths_as_images(passwords, grid_size=3, folder="output"):
    timestamp = str(int(time.time()))
    output_folder = os.path.join(folder, timestamp)
    os.makedirs(output_folder, exist_ok=True)

    for i, path in enumerate(passwords):
        x_coords = [p[1] for p in path]
        y_coords = [grid_size - 1 - p[0] for p in path]

        plt.figure(figsize=(6, 6))
        plt.scatter(x_coords, y_coords, color='black', zorder=5)
        plt.plot(x_coords, y_coords, color='black', marker='o', zorder=4)

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

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    grid_size = 4
    progress_interval = 100000  # Print progress every N patterns

    print(f"Generating most complex full-grid patterns for {grid_size}x{grid_size}...\n")
    start_time = time.time()
    passwords, max_complexity = generate_passwords(grid_size, progress_interval)
    duration = time.time() - start_time

    print(f"\nGeneration complete in {duration:.2f} seconds.")
    print(f"\nMost complex patterns (complexity = {max_complexity:.2f}):")
    for path in passwords:
        print(path)

    save_paths_as_images(passwords, grid_size)
    print(f"\nTotal patterns saved: {len(passwords)}")
