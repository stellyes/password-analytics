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

        # Cap max episode length
        self.step_count += 1
        if self.step_count >= self.coordinates * 2:
            done = True

        if self.visited[action] or \
            self.current_position is not None and \
                action == self.current_position:
            # Strongly discouraging staying in  
            # place and revisiting points
            reward -= 0.5
        else:
            intermediates = []
            if self.current_position is not None:
                intermediates = self.get_intermediate_points(self.current_position, action)

            if any(self.visited[i] for i in intermediates):
                # Discourage using intermediates that have been visited
                reward -= 0.2
            else:
                # We want to reward the intersections between
                # visited points and intermediates.
                # Increases complexity with "harmless overlapping".
                intersections = list(set(intermediates) & set(self.visited))
                for i in intersections:
                    reward += 0.2

                for i in intermediates:
                    self.visited[i] = 1
                    self.path.append(i)
                    # Discourage wasteful moves
                    reward -= 0.2
                
                self.visited[action] = 1
                self.path.append(action)
                self.current_position = action

                # Calculate total path complexity and close
                if len(self.path) == self.coordinates:
                    path_coordinates = [self.map[point + 1] for point in self.path]
                    reward += compute_complexity(path_coordinates, self.size) + 2
                    done = True
                else:
                    # Multiplier for continuing the path
                    reward += 0.2


        # If the player quits early
        # HEAVY penalty for not completing the path
        if done and len(self.path) < self.coordinates:
            reward -= 0.5 * (1 - len(self.path) / self.coordinates)

        return self.visited.copy(), reward, done, False, {}


    def get_intermediate_points(self, start, end):
        """Return all intermediate points passed through in a straight line."""
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
    angle1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    delta = abs(math.degrees(angle2 - angle1) - 180) % 360

    # Discourage standard angles
    # More loss if angles occur earlier in path
    if (delta in [0, 45, math.inf]):
        return -0.6 * (1 - (penalty[0]/penalty[1]))

    # Reward sharper angles
    return 0.3 + (0.3 * (delta - 180)/180)  

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


def compute_complexity(path, grid_size=4):
    score = 0

    # Calculate angle and length complexity
    for i in range(1, len(path) - 1):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        x3, y3 = path[i + 1]

        score += angle_score((x1, y1), (x2, y2), (x3, y3), (i, len(path) - 2))
        score += direction_score((x1, y1), (x2, y2), (x3, y3))

        # Rewards earlier, greedier 
        # utilization of empty board
        distance_score = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance_score *= (0.25 - (0.25 * (i / len(path) - 1)))
        score += distance_score
        
    return max(0.0, score)