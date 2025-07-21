# Dot-Grid Password Analytics

**For a grid of dots of size *n* by *n*, what is/are the most complex passwords that can be generated?**

# Overview

This Python script generates all valid unlock patterns on an `n x n` grid (similar to Android's unlock screen), evaluates their complexity, and saves images of the most complex patterns. It does so by:

1. Mapping grid coordinates to dot numbers.
2. Generating all valid sequences of connected dots based on movement rules.
3. Scoring patterns based on angular turns and step lengths.
4. Visualizing and saving the top patterns as images.
---
## Script Breakdown

### 1. `dot_coords(n)`
**Purpose:**  
Returns a mapping from dot numbers (1-based) to their `(row, col)` coordinates on an `n x n` grid where `n => 3 && n <= 7`. Constraints on `n` are implemented to prevent e

**Example:**
```python
dot_coords(3)
# Output: {1: (0, 0), 2: (0, 1), ..., 9: (2, 2)}
```
### 2. `coord_to_dot(coord, grid_size)`
**Purpose:**  
Converts a coordinate `(row, col)` back into a dot number.

### 3. `dots_between(a, b, grid_size)`
**Purpose:**
Finds all intermediate dots between two coordinates `a` and `b` that lie on a straight line. This is important for enforcing the rule that you can't skip unvisited dots.
For example, if we have a 3x3 size grid, connecting from `(0, 0)` to `(0, 2)` would subsequently mark `(0, 1)` as "visited".
**Logic:**
-   Uses the GCD of row and column differences to find step increments.
   
-   Only returns intermediate points if they exist along the line segment.

### 4. `dfs(visited, current, path, all_paths, grid_size)`

**Purpose:**  
Recursive depth-first search to explore all valid patterns from a starting dot.

**Key Rules Enforced:**

-   No dot is visited twice.
    
-   If a move skips intermediate dots, those dots must already be visited.
    
-   A valid pattern must be at least 4 dots long.
    

**Result:**  
Stores all valid paths in `all_paths`.
### 5. `generate_passwords(grid_size=3)`

**Purpose:**  
Generates all valid unlock patterns on the grid.

**Process:**

-   For each starting dot, it runs `dfs`.
    
-   Converts final dot-number-based paths into coordinate-based ones.
    

**Output:**  
A list of paths represented as `(row, col)` coordinates.

### 6. `compute_complexity(path)`

**Purpose:**  
Calculates a complexity score for each pattern based on:

-   **Directional changes** (turns): +1 point.
    
-   **Long moves** (≥2 cell steps in any direction): +2 points.
    

**Logic:**  
Uses `atan2` to detect angular change between three consecutive points.
### 7. `get_most_complex_passwords(passwords, top_n=5)`

**Purpose:**  
Ranks all generated paths by complexity and returns the top `n`.
### 8. `save_paths_as_images(passwords, grid_size=3, folder="output")`

**Purpose:**  
Visualizes and saves given password patterns as `.png` images.

**Features:**

-   Flips Y-axis for intuitive plotting.
    
-   Uses matplotlib with a non-interactive backend.
    
-   Adds dot labels and timestamps output folders for uniqueness.
- ---
## Example Usage
```bash
$ python pattern_generator.py
What size grid do you want to calculate passwords for?: 3
Generating patterns for 3x3 grid...
Generated 389112 total patterns.
Top 5 most complex patterns:
[(0, 0), (1, 1), (2, 2), ...]
...
Saved image: output/1721590123/password_1.png
...
```
---
## Requirements

-   Python 3.6+
    
-   matplotlib
    

You can install required packages with:
```bash
pip install matplotlib
```
---
## Notes

-   The complexity metric is heuristic; it prioritizes jagged and long-step patterns.
    
-   Larger grid sizes will lead to exponential growth in generated patterns and memory usage.
---
## Future releases
- Multi-threading to minimize generation time
