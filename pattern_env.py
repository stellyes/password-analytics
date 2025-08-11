import math
import numpy
import gymnasium


class PatternEnv(gymnasium.Env):
    
    def __init__(self, grid_size=4):
        super(PatternEnv, self).__init__()

        # Generating the grid space 
        self.size = grid_size
        self.coordinates = grid_size ** 2
        self.map = generate_coordinate_map(grid_size)

        # Generate RL space
        self.action_space = gymnasium.spaces.Discrete(self.coordinates)
        self.observation_space = gymnasium.spaces.MultiBinary(self.coordinates)

        self.last_full_path = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        # Generate a new seed for current episode
        # or, the "player" starts the game
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(seed)

        # To keep track of the player's path
        self.path = []
        self.visited = numpy.zeros(self.coordinates, dtype=numpy.int8)
        self.current_position = None
        self.last_full_path = None

        # Resetting the step count
        self.step_count = 0
        return self.visited.copy(), {}

    def step(self, action):
        done = False
        reward = 0.0

        # End episode early only if invalid move AND max steps reached
        if self.visited[action] == 1 or \
            (self.current_position is not None and action == self.current_position):
            # Strongly discouraging staying in place and revisiting points
            reward = -1
            done = True
        else:
            # Random move 2% of the time
            if numpy.random.rand() < 0.02:
                unvisited = [i for i in range(self.coordinates) if not self.visited[i]]
                if unvisited:
                    action = numpy.random.choice(unvisited)

            intermediates = self.get_intermediate_points(self.current_position, action)

            # Mark intermediate points visited
            for point in intermediates:
                if not self.visited[point]:
                    self.visited[point] = 1
                    self.path.append(point)
                    reward -= 0.25

            # Mark final chosen action
            self.visited[action] = 1
            self.path.append(action)
            self.current_position = action

            # Multiplier for longer paths
            reward += 0.1 + (len(self.path) * 0.025)

        # Increment step counter regardless of intermediates
        self.step_count += 1

        # End after exactly N steps
        if self.step_count >= self.coordinates:
            # Calculate complexity if all dots visited
            if numpy.all(self.visited):
                path_coordinates = [self.map[point + 1] for point in self.path]
                reward += compute_complexity(path_coordinates)
                reward += 1  # bonus for completion
            done = True

        # Debug quick fix from your original code
        if reward > 0.8:
            reward = -1     

        return self.visited.copy(), reward, done, False, {}



    def get_intermediate_points(self, start, end):
        """Return all intermediate points passed through in a straight line."""

        if start is None:
            return [] #no intermediates for first point

        x1, y1 = self.map[start + 1]
        x2, y2 = self.map[end + 1]

        dx = x2 - x1
        dy = y2 - y1

        # Only consider straight lines or diagonals
        steps = max(abs(dx), abs(dy))
        if steps <= 1:
            return []  # no intermediates needed

        intermediates = []
        for i in range(1, steps):
            x = x1 + i * dx // steps
            y = y1 + i * dy // steps
            # Find the dot index from (x, y)
            for dot_idx, coord in self.map.items():
                if coord == (x, y):
                    intermediates.append(dot_idx - 1)  # back to 0-indexed
                    break

        return intermediates

def generate_coordinate_map(n):
    '''Generates grid layout for n x n plot'''
    return {i * n + j + 1: (i, j) for i in range(n) for j in range(n)}


def angle_score(p1, p2, p3, penalty):
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]

    def slope(dx, dy):
        if dx == 0:
            return float('inf')  # vertical
        return dy / dx

    slope1 = slope(dx1, dy1)
    slope2 = slope(dx2, dy2)

    # Penalize forbidden slopes for either segment
    forbidden = {0, float('inf'), 1, -1}
    if slope1 in forbidden or slope2 in forbidden:
        return -0.8

    # Otherwise reward sharper turns
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    delta = abs((angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi)
    return 0.6 * (1 - delta / math.pi)  

def direction_score(p1, p2, p3):
    # 1 represents up and right
    # -1 represents down and left    
    v1_direction = 1 if p1[1] < p2[1] else -1
    v2_direction = 1 if p2[1] < p3[1] else -1
    h1_direction = 1 if p1[0] < p2[0] else -1
    h2_direction = 1 if p2[0] < p3[0] else -1

    # Reward sharper angles
    direction_change = [
        bool(v1_direction != v2_direction),
        bool(h1_direction != h2_direction)
    ]
    return 0.15 * direction_change.count(True)


def compute_complexity(path, dot_coords):
    """
    Compute complexity score for an Android-style unlock pattern.
    Higher is more complex, lower is simpler.

    Parameters:
    - path: list of dot indices in the pattern
    - dot_coords: dict mapping index -> (x, y) coordinates
    """
    if len(path) < 2:
        return -1.0  # no complexity

    total_score = 0.0
    penalties = 0.0
    bonuses = 0.0

    prev_angle = None
    prev_dir_cat = None
    direction_streak = 0

    for i in range(1, len(path)):
        x1, y1 = dot_coords[path[i - 1]]
        x2, y2 = dot_coords[path[i]]

        dx, dy = x2 - x1, y2 - y1
        segment_length = math.hypot(dx, dy)

        # Calculate angle in degrees
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        # --- 1. Penalize perfect multiples of 45° ---
        angle_mod = angle % 45
        angle_diff = min(angle_mod, 45 - angle_mod)
        if angle_diff < 1e-6:  # perfect multiple of 45°
            penalties += 2.0
        else:
            # Reward for being away from perfect angles (max at 22.5° offset)
            irregularity_bonus = (angle_diff / 22.5)  # normalized 0–1
            bonuses += irregularity_bonus * 0.5

        # --- 2. Penalize repeated direction categories ---
        # Categories: H(0), V(1), D(2)
        if abs(dx) < 1e-6:  # vertical
            dir_cat = 1
        elif abs(dy) < 1e-6:  # horizontal
            dir_cat = 0
        elif abs(abs(dx) - abs(dy)) < 1e-6:  # perfect diagonal
            dir_cat = 2
        else:  # irregular
            dir_cat = 3

        if prev_dir_cat is not None:
            if dir_cat == prev_dir_cat:
                direction_streak += 1
                penalties += 0.5 * direction_streak
            else:
                direction_streak = 0
        prev_dir_cat = dir_cat

        # --- 3. Penalize long straight jumps ---
        if dir_cat in (0, 1, 2) and segment_length > 1.5:
            penalties += 0.8 * (segment_length - 1.5)

        # --- 4. Reward sharper turns ---
        if prev_angle is not None:
            turn_angle = abs(angle - prev_angle)
            if turn_angle > 180:
                turn_angle = 360 - turn_angle
            if 30 <= turn_angle <= 150:
                bonuses += 0.3
        prev_angle = angle

        # --- Base score: encourage movement ---
        total_score += segment_length * 0.1

    # Final complexity score
    raw_score = total_score + bonuses - penalties

    # Normalize (rough scaling so RL rewards aren't too extreme)
    normalized = max(-5.0, min(5.0, raw_score)) / 5.0

    return normalized