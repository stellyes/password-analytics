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
        reward = 0.0
        done = False

        # First move → always allowed
        if self.current_position is None:
            self.visited[action] = 1
            self.path.append(action)
            self.current_position = action
            # No big reward for just starting
            return self.visited.copy(), 0.0, done, False, {}

        # Random move 7% of the time
        #if numpy.random.rand() < 0.07:
        #    unvisited = [i for i in range(self.coordinates) if not self.visited[i]]
        #    if unvisited:
        #       action = numpy.random.choice(unvisited)

        # Invalid move → episode ends
        if self.visited[action] == 1 or action == self.current_position:
            reward = -1.0
            done = True
            return self.visited.copy(), reward, done, False, {}

        # Intermediate point check
        intermediates, discourage_factor, dx, dy = self.get_intermediate_points(self.current_position, action)
        if any(self.visited[i] for i in intermediates):
            reward = -1.0
            done = True
            return self.visited.copy(), reward, done, False, {}

        # Valid move → mark visited
        self.visited[action] = 1
        self.path.append(action)
        self.current_position = action

        norm_length = len(self.path) / self.coordinates
        norm_dist = (abs(dx) + abs(dy)) / (2 * self.coordinates)
        discourage_factor = 0.5 + norm_dist ** 2

        # Base incremental reward scaled by path length and discourage factor
        reward = 0.05 + 0.05 * discourage_factor
        reward *= norm_length

        # If all dots visited → bigger final reward
        if len(self.path) == self.coordinates:
            coord_path = [self.map[p] for p in self.path]
            reward += 2.0 + compute_complexity(coord_path)
            done = True
        else:
            # If episode ends prematurely, harsh penalty
            if done:
                reward -= 1.0
            # Living penalty to encourage continuing
            elif len(self.path) < 3:
                reward -= 0.05

        if done and len(self.path) >= 10:
            reward = -1.0

        return self.visited.copy(), reward, done, False, {}


    def get_intermediate_points(self, start, end):
        """
        Returns:
            intermediates: list of points passed over between start and end
            discourage_factor: multiplier to discourage boring/short moves
        """
        intermediates = []

        start_x, start_y = self.map[start]
        end_x, end_y = self.map[end]

        dx = end_x - start_x
        dy = end_y - start_y

        # Calculate number of steps in the path
        steps = max(abs(dx), abs(dy))
        if steps > 1:
            step_x = dx // steps
            step_y = dy // steps
            for k in range(1, steps):
                intermediates.append((start_x + step_x * k, start_y + step_y * k))

        # Map back to indices
        intermediates_idx = [
            idx for idx, coord in self.map.items() if coord in intermediates
        ]

        # Discourage factor based on Manhattan distance (shorter moves get smaller factors)
        manhattan_dist = abs(dx) + abs(dy)
        discourage_factor = 0.5 + (manhattan_dist / (2 * len(set(self.map.values()))))

        return intermediates_idx, discourage_factor, dx, dy


def generate_coordinate_map(n):
    """Generates grid layout for n x n plot with 0-based keys."""
    return {i * n + j: (i, j) for i in range(n) for j in range(n)}


def compute_complexity(path, dot_coords=None):
    """
    Compute complexity score for an Android-style unlock pattern.
    Higher is more complex, lower is simpler.

    Parameters:
    - path: list of dot indices OR list of (x, y) coordinates.
    - dot_coords: optional dict mapping index -> (x, y) coordinates.
                  If None, 'path' is assumed to already be coordinate tuples.
    """
    # If dot_coords is provided, convert indices to coordinates
    if dot_coords is not None and path and not isinstance(path[0], tuple):
        coords = [dot_coords[p] for p in path]
    else:
        coords = path

    if len(coords) < 2:
        return -1.0

    total_score = 0.0
    penalties = 0.0
    bonuses = 0.0

    prev_angle = None
    prev_dir_cat = None
    direction_streak = 0

    # Define standard angles in degrees (0°, 45°, 90°, 135°, 180°)
    standard_angles = [0, 45, 90, 135, 180]
    angle_penalty_strength = 3.0  # stronger penalty on exact matches

    for i in range(1, len(coords)):
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]

        dx, dy = x2 - x1, y2 - y1
        segment_length = math.hypot(dx, dy)

        # Compute angle in degrees normalized to [0, 180]
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        if angle > 180:
            angle = 360 - angle

        # Calculate closest standard angle difference
        angle_diffs = [abs(angle - sa) for sa in standard_angles]
        min_diff = min(angle_diffs)

        # Penalize angles near standard angles with a smooth penalty
        # Strong penalty if within 5 degrees, tapering off up to 15 degrees
        if min_diff < 5:
            penalty = angle_penalty_strength * (1 - min_diff / 5)
            penalties += penalty
        elif min_diff < 15:
            penalty = angle_penalty_strength * (1 - (min_diff - 5) / 10) * 0.5
            penalties += penalty
        else:
            # Reward irregular angles more strongly
            irregularity_bonus = min_diff / 180  # normalized 0–1
            bonuses += irregularity_bonus * 0.7

        # Penalize repeated direction categories (vertical, horizontal, diagonal, irregular)
        if abs(dx) < 1e-6:
            dir_cat = 1  # vertical
        elif abs(dy) < 1e-6:
            dir_cat = 0  # horizontal
        elif abs(abs(dx) - abs(dy)) < 1e-6:
            dir_cat = 2  # perfect diagonal
        else:
            dir_cat = 3  # irregular

        if prev_dir_cat is not None:
            if dir_cat == prev_dir_cat:
                direction_streak += 1
                penalties += 0.5 * direction_streak
            else:
                direction_streak = 0
        prev_dir_cat = dir_cat

        # Penalize very long straight jumps (> 1.5 units)
        if dir_cat in (0, 1, 2) and segment_length > 1.5:
            penalties += 0.8 * (segment_length - 1.5)

        # Reward sharper turns between segments
        if prev_angle is not None:
            turn_angle = abs(angle - prev_angle)
            if turn_angle > 180:
                turn_angle = 360 - turn_angle
            if 30 <= turn_angle <= 150:
                bonuses += 0.3
        prev_angle = angle

        # Reward longer moves earlier in the path
        # Normalize by position to weight earlier segments more
        position_weight = 1.5 - (i / len(coords))  # decreases linearly
        total_score += segment_length * 0.1 * position_weight

    raw_score = total_score + bonuses - penalties
    normalized = max(-5.0, min(5.0, raw_score)) / 5.0

    return normalized