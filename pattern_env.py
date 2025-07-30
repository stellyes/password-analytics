import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import atan2, degrees, sqrt


class PatternEnv(gym.Env):
    def __init__(self, grid_size=4):
        super(PatternEnv, self).__init__()
        self.grid_size = grid_size
        self.total_dots = grid_size ** 2
        self.coords_map = dot_coords(grid_size)

        self.action_space = spaces.Discrete(self.total_dots)
        self.observation_space = spaces.MultiBinary(self.total_dots)

        self.max_steps = self.total_dots * 2  # Optional safeguard

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.path = []
        self.visited = np.zeros(self.total_dots, dtype=np.int8)
        self.steps = 0

        start = self.np_random.integers(self.total_dots)
        self.visited[start] = 1
        self.path.append(start)
        self.current_pos = start

        return self.visited.copy(), {}

    def step(self, action):
        self.steps += 1
        done = False
        reward = 0.0
        truncated = False

        if self.visited[action]:
            reward = -1.0
            done = True
        elif action == self.current_pos:
            reward = -0.5
            done = True
        else:
            intermediates = self.get_intermediate_dots(self.current_pos, action)
            invalid = any(self.visited[i] for i in intermediates)

            if invalid:
                reward = -1.0
                done = True
            else:
                for i in intermediates:
                    self.visited[i] = 1
                    self.path.append(i)

                self.visited[action] = 1
                self.path.append(action)
                self.current_pos = action

                current_length = len(self.path)

                # Encourage longer paths with a small step bonus
                reward = 0.05 + 0.025 * current_length

                if current_length == self.total_dots:
                    coords_path = [self.coords_map[dot + 1] for dot in self.path]
                    reward = compute_complexity(coords_path)
                    done = True
                    print("Pattern complete:", self.path)
                    print("Complexity score:", reward)

                elif self.steps >= self.max_steps:
                    reward = -1.0
                    done = True
                    print("Max steps reached. Path length:", current_length)

                # Discard short patterns explicitly
                elif current_length < self.grid_size ** 2 - 2:
                    reward -= 0.5
                    done = True
                    print("Short pattern discarded:", [int(p) for p in self.path])

        if done:
            print("Episode ended. Path length:", len(self.path), "Reward:", reward)

        return self.visited.copy(), reward, done, truncated, {}


    def get_intermediate_dots(self, start, end):
        x1, y1 = self.coords_map[start + 1]
        x2, y2 = self.coords_map[end + 1]

        dx = x2 - x1
        dy = y2 - y1

        steps = max(abs(dx), abs(dy))
        if steps <= 1:
            return []

        intermediates = []
        for i in range(1, steps):
            x = x1 + i * dx // steps
            y = y1 + i * dy // steps
            for dot_idx, coord in self.coords_map.items():
                if coord == (x, y):
                    intermediates.append(dot_idx - 1)
                    break

        return intermediates

    def render(self, mode='human'):
        print("🧩 Current path:", self.path)


def dot_coords(n):
    return {i * n + j + 1: (i, j) for i in range(n) for j in range(n)}


# -------- COMPLEXITY SCORING --------
def angle_score(p1, p2, p3):
    angle1 = atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = atan2(p3[1] - p2[1], p3[0] - p2[0])
    delta = abs(degrees(angle2 - angle1)) % 360
    if delta > 180:
        delta = 360 - delta
    if delta % 45 == 0:
        return delta / 180 * 0.5
    return delta / 180


def direction_category(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dx == 0:
        return 'vertical'
    elif dy == 0:
        return 'horizontal'
    else:
        return 'diagonal'


def count_direction_changes(path):
    categories = [direction_category(path[i], path[i + 1]) for i in range(len(path) - 1)]
    changes = sum(1 for i in range(1, len(categories)) if categories[i] != categories[i - 1])
    return changes / max(1, len(categories) - 1)


def angle_variance(path):
    if len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path) - 1):
        a1 = atan2(path[i][1] - path[i - 1][1], path[i][0] - path[i - 1][0])
        a2 = atan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
        delta = abs(degrees(a2 - a1)) % 360
        if delta > 180:
            delta = 360 - delta
        angles.append(delta)
    return np.std(angles) / 180


def avg_center_distance(path, grid_size):
    if not path:
        return 0.0
    center = (grid_size - 1) / 2
    dists = [sqrt((x - center) ** 2 + (y - center) ** 2) for x, y in path]
    return sum(dists) / len(dists)


def is_standard_angle_segment(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    return dx == 0 or dy == 0 or dx == dy


def standard_angle_penalty(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dx == 0 or dy == 0:
        return -1.0
    elif dx == dy:
        return -0.75
    return 0.5


def compute_complexity(path):
    if len(path) < 2:
        return 0.0

    total_score = 0.0
    total_length = 0.0
    direction_changes = 0
    angles = []
    irregular_segments = 0

    prev_dx, prev_dy = None, None

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        total_length += dist

        angle_bonus = standard_angle_penalty((x1, y1), (x2, y2))
        if angle_bonus > 0:
            irregular_segments += 1

        total_score += dist + angle_bonus

        if prev_dx is not None and prev_dy is not None:
            if (dx, dy) != (prev_dx, prev_dy):
                direction_changes += 1

            dot = prev_dx * dx + prev_dy * dy
            mag1 = math.hypot(prev_dx, prev_dy)
            mag2 = math.hypot(dx, dy)
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle = math.acos(cos_angle)
                angles.append(angle)

        prev_dx, prev_dy = dx, dy

    decay_factor = 0.98
    step_bonus = sum((decay_factor ** i) for i in range(len(path)))
    total_score += 0.3 * step_bonus

    if len(path) >= 13:
        total_score += 2.0

    if len(path) > 1:
        irregular_ratio = irregular_segments / (len(path) - 1)
        total_score += 2.0 * irregular_ratio

    if len(angles) > 1:
        angle_var = np.var(angles)
        total_score += 0.5 * angle_var

    return round(max(0.0, min(total_score, 10.0)), 4)
