import math
import os
import time
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Use a non-interactive backend for saving images

# Define all 8 possible lines in the grid
def get_all_possible_lines():
    """Return a set of all 8 valid lines between grid points."""
    base_lines = {
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (1, 2)),
        ((0, 0), (1, 1)),
        ((0, 0), (2, 1)),
        ((2, 0), (1, 0)),
        ((1, 0), (0, 2)),
        ((0, 1), (1, 0)),
    }

    def generate_all_transformations():
        """Generate all 8 transformations (rotations and reflections) for a 3x3 grid."""
        def identity(p):
            return p

        def rotate90(p):
            x, y = p
            return (y, 2 - x)

        def rotate180(p):
            x, y = p
            return (2 - x, 2 - y)

        def rotate270(p):
            x, y = p
            return (2 - y, x)

        def reflect_horizontal(p):
            x, y = p
            return (2 - x, y)

        def reflect_vertical(p):
            x, y = p
            return (x, 2 - y)

        def reflect_main_diagonal(p):
            x, y = p
            return (y, x)

        def reflect_anti_diagonal(p):
            x, y = p
            return (2 - y, 2 - x)

        return [
            identity,
            rotate90,
            rotate180,
            rotate270,
            reflect_horizontal,
            reflect_vertical,
            reflect_main_diagonal,
            reflect_anti_diagonal
        ]

    # Add all rotations/reflections (by treating grid as 3x3)
    all_lines = set()
    for line in base_lines:
        a, b = line
        # Generate symmetric versions by rotating and reflecting
        for transform in generate_all_transformations():
            a_t = transform(a)
            b_t = transform(b)
            all_lines.add(tuple(sorted((a_t, b_t))))
    return all_lines

# Function to check if a line is valid (sorted tuple to avoid order mismatch)
def check_line_covered(line, used_lines):
    """Check if a given line has already been covered (order-agnostic)."""
    return tuple(sorted(line)) in used_lines

def save_paths_as_images(passwords, folder="output"):
    """Save each password's path as an image file in the specified folder."""
    
    # Create the output folder with a subfolder named by the UNIX timestamp
    timestamp = str(int(time.time()))  # Get UNIX timestamp
    output_folder = os.path.join(folder, timestamp)
    os.makedirs(output_folder, exist_ok=True)
    
    for i, path in enumerate(passwords):
        # Extract x and y coordinates from the path
        x_coords = [point[0] for point in path]
        y_coords = [point[1] for point in path]
        
        # Create a plot for the current password's path
        plt.figure(figsize=(6, 6))
        
        # Plot the points
        plt.scatter(x_coords, y_coords, color='black', zorder=5)
        
        # Plot the lines between points
        plt.plot(x_coords, y_coords, color='black', marker='o', zorder=4)
        
        # Add labels to the points
        for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.text(x + 0.05, y + 0.05, f'({x},{y})', fontsize=9, color='black')

        # Set grid limits based on the grid size (assuming square grid)
        grid_size = 3  # Change as needed, or dynamically use the max grid size
        plt.xlim(-0.5, grid_size - 0.5)
        plt.ylim(-0.5, grid_size - 0.5)
        
        # Remove axis and grid
        plt.axis('off')  # Hide axis
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Set aspect ratio to be equal so the grid is not distorted
        plt.gca().set_aspect('equal', adjustable='box')

        # Title of the plot (for clarity)
        plt.title(f"Password {i+1}: Path Visualization", fontsize=10)
        
        # Save the plot as an image in the specified folder
        image_filename = os.path.join(output_folder, f"password_{i+1}.png")
        plt.savefig(image_filename, bbox_inches='tight')
        plt.close()  # Close the plot to avoid overlap of images
        
        print(f"Saved image: {image_filename}")

def is_between(a, b, c):
    """Check if point c is between points a and b horizontally or vertically."""
    # Horizontal check
    if a[0] == b[0] and a[0] == c[0]:
        return min(a[1], b[1]) < c[1] < max(a[1], b[1])
    # Vertical check
    if a[1] == b[1] and a[1] == c[1]:
        return min(a[0], b[0]) < c[0] < max(a[0], b[0])
    return False

def calculate_distance(a, b):
    """Calculate the Euclidean distance between points a and b."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def generate_passwords(grid_size=3):
    """Generate all possible valid passwords for a 3x3 grid."""
    grid = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    all_lines = get_all_possible_lines()  # Get all 8 valid lines
    
    # Result list to store all valid passwords
    passwords = []
    
    def backtrack(path, used, used_lines):
        # If we have a full path (at least 4 dots), add it to the result
        if len(path) >= 4 and all(line in used_lines for line in all_lines):
            passwords.append(path)
            return
        
        if len(path) == len(grid):  # Full path with all 9 dots
            # Check if all 8 lines are used at this stage
            if all(line in used_lines for line in all_lines):
                passwords.append(path)
            return
        
        # Prioritize the closest available point to the last point
        last_point = path[-1]
        remaining_points = [point for point in grid if point not in used]
        remaining_points.sort(key=lambda p: calculate_distance(last_point, p))  # Sort by distance
        
        for point in remaining_points:
            # Check that no intermediate dots are skipped
            valid_move = True
            if path:
                # Check if we need to pass through intermediate dots
                x1, y1 = last_point
                x2, y2 = point
                if x1 != x2 and y1 != y2:  # Diagonal movement
                    # Check that all points in between are visited
                    for ix in range(min(x1, x2) + 1, max(x1, x2)):
                        if (ix, y1) not in used:
                            valid_move = False
                            break
                    for iy in range(min(y1, y2) + 1, max(y1, y2)):
                        if (x1, iy) not in used:
                            valid_move = False
                            break

            if valid_move:
                # Add the line to the used_lines set
                new_used_lines = set(used_lines)
                line = tuple(sorted((last_point, point)))  # Create a line from previous to current point
                new_used_lines.add(line)
                
                # Mark the current point as used and backtrack
                backtrack(path + [point], used | {point}, new_used_lines)

    # Start the backtracking from each possible point in the grid
    for start_point in grid:
        backtrack([start_point], {start_point}, set())  # Empty used_lines initially

    return passwords


def remove_rotational_and_reverse_duplicates(passwords):
    """Remove passwords that are rotations or reversals of each other."""
    unique_passwords = []
    seen = set()  # Set to store patterns (including reversed versions)

    for password in passwords:
        # Convert the password to a tuple (to make it hashable)
        rotation_tuple = tuple(password)
        reverse_tuple = tuple(reversed(password))  # Reverse of the current password

        # Check if the reversed password has already been seen
        if reverse_tuple not in seen:
            # Add the current password's tuple and its reverse to the set
            unique_passwords.append(password)
            seen.add(rotation_tuple)  # Add the original pattern
            seen.add(reverse_tuple)   # Add the reversed pattern

    return unique_passwords

def calculate_path_length(path):
    """Calculate the total length of the path (sum of Euclidean distances)."""
    length = 0
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return length

def calculate_slope_sum(path):
    """Calculate the sum of the absolute values of the slopes between consecutive points."""
    slope_sum = 0
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        # Avoid division by zero (vertical line)
        if x2 != x1:
            slope_sum += abs((y2 - y1) / (x2 - x1))
        else:
            # If it's a vertical line, add an arbitrarily large value (for complexity)
            slope_sum += float('inf')
    return slope_sum

def calculate_complexity(path):
    """Calculate the complexity of the password based on path length and slope sum."""
    path_length = calculate_path_length(path)
    slope_sum = calculate_slope_sum(path)
    # Here we assume both factors are equally important for simplicity, but you can adjust the weight
    return path_length + slope_sum

def get_most_complex_passwords(passwords, top_n=5):
    """Return the top_n most complex passwords based on path length and slope sum."""
    # Calculate complexity for each password
    password_complexities = [(password, calculate_complexity(password)) for password in passwords]
    
    # Sort passwords by complexity in descending order
    password_complexities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top_n most complex passwords
    return [password for password, _ in password_complexities[:top_n]]

# Generate all passwords
passwords = generate_passwords()

# Remove rotational and reverse duplicates
unique_passwords = remove_rotational_and_reverse_duplicates(passwords)

# Get the most complex passwords
most_complex_passwords = get_most_complex_passwords(unique_passwords, top_n=5)

# Print the most complex passwords
print(f"The top {len(most_complex_passwords)} most complex passwords are:")
for password in most_complex_passwords:
    print(password)

# Save the most complex passwords as images
save_paths_as_images(most_complex_passwords)

print(f"Total unique passwords considered: {len(unique_passwords)}")
