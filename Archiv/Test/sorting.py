import numpy as np

def grid_sorting(elements):
    """
    Fits and sorts found points into a 3x3 grid structure. Interpolates missing points.
    
    Args:
        elements (list): List of dictionaries with 'area_focus_point' as [x, y].
        
    Returns:
        list: Sorted and completed 3x3 grid structure with valid points.
    """
    # Step 1: Filter valid points within a 3x3 grid area
    valid_elements = [
        elem for elem in elements
        if 0 <= elem["area_focus_point"][0] < 3 and 0 <= elem["area_focus_point"][1] < 3
    ]
    
    # Step 2: Sort valid points by row (y-coordinate) and then by column (x-coordinate)
    valid_elements.sort(key=lambda e: (e["area_focus_point"][1], e["area_focus_point"][0]))
    
    # Step 3: Create a 3x3 grid and map the valid points
    grid = [[None for _ in range(3)] for _ in range(3)]
    for elem in valid_elements:
        x, y = int(round(elem["area_focus_point"][0])), int(round(elem["area_focus_point"][1]))
        if grid[y][x] is None:  # If slot is not yet occupied
            grid[y][x] = elem
    
    # Step 4: Interpolate missing points
    for y in range(3):
        for x in range(3):
            if grid[y][x] is None:
                grid[y][x] = {
                    "area_focus_point": [x + 0.5, y + 0.5],  # Interpolated position
                    "bbox": None,
                    "color": None,
                    "grid_position": None,
                    "avg_hue": None,
                }
    
    # Step 5: Flatten grid into a list and assign grid positions
    flattened_grid = []
    for i, row in enumerate(grid):
        for j, elem in enumerate(row):
            elem["grid_position"] = i * 3 + j + 1
            flattened_grid.append(elem)
    
    return flattened_grid

# Example usage
dictionary = [
    {"area_focus_point": [1, 1]},
    {"area_focus_point": [2.1, 1]},
    {"area_focus_point": [1, 2]},
    {"area_focus_point": [3, 3]},  # Out of bounds
    {"area_focus_point": [0, 0]},
]

sorted_dict = grid_sorting(dictionary)
for entry in sorted_dict:
    print(entry)
