import gymnasium as gym
from gymnasium import spaces
import numpy
from math import atan2, degrees, sqrt


class PatternEnv(gym.Env):
    def __init__(self, grid_size = 4):
        super(PatternEnv, self).__init__()
        self.grid_size = grid_size
        self.total_dots = grid_size ** 2
        self.coords_map = dot_coords(grid_size)

        self.action_space = spaces.Discrete(self.total_dots)
        self.observation_space = spaces.MultiBinary(self.total_dots)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.path = []
        self.visited = numpy.zeros(self.total_dots, dtype=numpy.int8)

        start = self.np_random.integers(self.total_dots)
        self.visited[start] = 1
        self.path.append(start)
        self.current_pos = start

        return self.visited.copy(), {}


    def step(self, action):
        done = False
        reward = 0.0
        truncated = False  # No time limit logic yet

        if self.visited[action]:
            reward = -1.0
            done = True
        elif self.current_pos is not None and action == self.current_pos:
            reward = -0.5  # discourage staying in place
            done = True
        else:
            self.visited[action] = 1
            self.path.append(action)
            self.current_pos = action

            if len(self.path) == self.total_dots:
                coords_path = [self.coords_map[dot + 1] for dot in self.path]
                reward = compute_complexity(coords_path)
                done = True
            else:
                # reward shaping
                reward = 0.1 + len(self.path) * 0.05

        return self.visited.copy(), reward, done, truncated, {}



    def render(self, mode='human'):
        print("Current path:", self.path)

def dot_coords(n):
    '''Generates grid layout for n x n plot'''
    return {i * n + j + 1: (i, j) for i in range(n) for j in range(n)}

# ---------------------------
# COMPLEXITY SCORING
# ---------------------------
def angle_score(p1, p2, p3):
    angle1 = atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = atan2(p3[1] - p2[1], p3[0] - p2[0])
    delta = abs(degrees(angle2 - angle1)) % 360
    if delta > 180:
        delta = 360 - delta
    return (180 - delta) / 180  # sharper = closer to 1


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
    return changes / max(1, len(categories) - 1)  # normalize to [0,1]


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
    return numpy.std(angles) / 180  # normalized to [0, 1]


def avg_center_distance(path, grid_size):
    if not path:
        return 0.0
    center = (grid_size - 1) / 2
    dists = [sqrt((x - center) ** 2 + (y - center) ** 2) for x, y in path]
    return sum(dists) / len(dists)


def compute_complexity(path, grid_size=4):
    angle_weight = 2.0
    length_weight = 1.0
    direction_change_weight = 2.0
    angle_var_weight = 1.0
    center_dist_weight = -0.5  # negative to discourage hugging the edges

    score = 0.0

    max_segment_length = sqrt(2) * (grid_size - 1)

    # Angle + Length
    for i in range(1, len(path) - 1):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        x3, y3 = path[i + 1]

        score += angle_weight * angle_score((x1, y1), (x2, y2), (x3, y3))

        dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        score += length_weight * dist

    # Final segment
    if len(path) >= 2:
        x1, y1 = path[-2]
        x2, y2 = path[-1]
        dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        score += length_weight * dist

    # Direction change reward
    score += direction_change_weight * count_direction_changes(path)

    # Angle variance
    score += angle_var_weight * angle_variance(path)

    # Center distance
    score += center_dist_weight * avg_center_distance(path, grid_size)

    # Normalize
    max_possible_score = (
        (len(path) - 2) * (angle_weight * 1 + length_weight * max_segment_length) +
        length_weight * max_segment_length +
        direction_change_weight * 1.0 +
        angle_var_weight * 1.0 +
        abs(center_dist_weight) * sqrt(2) * (grid_size - 1)
    )

    normalized_score = score / max_possible_score if max_possible_score > 0 else 0.0
    return max(0.0, normalized_score)  # Clamp to [0, 1]
