# Projekt:      Rubiks Cube
# Participants: Lukas Gerstlauer, 205293, lgerstla@stud.hs-heilbronn.de
#               Tim Söns, 204453, tsoens@stud.hs-heilbronn.de
# Lecture:      Computer & Robot Vision
# Program:      MAS
# Date:         31.01.2025

import numpy as np

def group_coordinates(coords, threshold, required_groups=3):
    """
    Groups coordinates based on a threshold and returns the mean of each group.
    Args:
        coords (list of float): A list of coordinates to be grouped.
        threshold (float): The maximum difference between consecutive coordinates 
                           to be considered in the same group.
        required_groups (int, optional): The number of groups required. Defaults to 3.
    Returns:
        list of float: A sorted list of the mean values of each group if the number 
                       of groups matches required_groups, otherwise None.
    """
    
    coords = sorted(coords) 
    groups = [] 
    current_group = [coords[0]] 

    for i in range(1, len(coords)):
        if coords[i] - coords[i - 1] <= threshold:  # Check if the current coordinate is within the threshold
            current_group.append(coords[i]) 
        else:
            groups.append(current_group) 
            current_group = [coords[i]] 
    groups.append(current_group) 
     
    # Check if the number of groups matches the required number of groups
    if len(groups) != required_groups: 
        return None

    group_means = [np.mean(group) for group in groups]  

    return sorted(group_means) 

def assign_grid_positions(entries):
    """
    Assigns grid positions to a list of entries based on their area focus points.
    Parameters:
    entries (list of dict): A list of dictionaries, where each dictionary represents an entry with the following keys:
        - 'area_focus_point' (tuple): A tuple (x, y) representing the focus point of the entry.
        - 'bbox' (tuple): A tuple (x, y, width, height) representing the bounding box of the entry.
    Returns:
    tuple: A tuple containing:
        - entries (list of dict): The updated list of entries with an additional key 'grid_position' indicating the assigned grid position.
        - x_groups (list of float): The x-coordinates of the grid groups.
        - y_groups (list of float): The y-coordinates of the grid groups.
    """
    # Check if the entries list is empty
    if not entries:  
        return None, None, None
    
    # Extract coordinates
    x_coords = [entry['area_focus_point'][0] for entry in entries]  
    y_coords = [entry['area_focus_point'][1] for entry in entries] 

    total_width, total_height = 0, 0 

    for entry in entries:
        bbox = entry['bbox']
        total_width += bbox[2]
        total_height += bbox[3] 

    average_width = total_width / len(entries)  # Calculate average width
    average_height = total_height / len(entries)  # Calculate average height
    threshold = (average_width + average_height) / 4  # Calculate threshold

    x_groups = group_coordinates(x_coords, threshold, required_groups=3) 
    y_groups = group_coordinates(y_coords, threshold, required_groups=3)  

    if x_groups is None or y_groups is None:  # Check if grouping was successful
        return entries, None, None
    else:
        x_groups.sort() 
        y_groups.sort() 

        for entry in entries:
            x, y = entry['area_focus_point']
            
            # Find the closest group
            col = np.argmin([abs(x - group) for group in x_groups])  
            row = np.argmin([abs(y - group) for group in y_groups])
            
            # Calculate grid position and assign grid position to entry
            grid_position = row * 3 + col + 1  
            entry['grid_position'] = grid_position 

    return entries, x_groups, y_groups

def merge_groups(group1, group2, faktor1=2, faktor2=1):
    """
    Merges two groups of numerical values by averaging corresponding elements
    with given factors.
    Parameters:
        group1 (list of float or int): The first group of numerical values.
        group2 (list of float or int): The second group of numerical values.
        faktor1 (float or int, optional): The factor to multiply elements of group1. Default is 2.
        faktor2 (float or int, optional): The factor to multiply elements of group2. Default is 1.
    Returns:
        list of float: A new list containing the merged and averaged values of the input groups.
                    If both groups are None, returns None. If one group is None, returns the other group.
    """
    # Check if groups are None
    if group1 is None and group2 is None:  
        return None
    if group1 is None: 
        return group2
    if group2 is None: 
        return group1
    
    merged_group = [
        (faktor1 * g1 + faktor2 * g2) / (faktor1 + faktor2)  # Merge and average corresponding elements
        for g1, g2 in zip(sorted(group1), sorted(group2))]
    
    return merged_group

def merge_dictionaries(dict_color_sorted, dict_edge_sorted, x_groups1, y_groups1, x_groups2, y_groups2, faktor_color_dict=2, faktor_edge_dict=1):
    """
    Merges two dictionaries containing sorted color and edge data, respectively, and their associated groupings.
    Parameters:
        dict_color_sorted (list of dict): List of dictionaries sorted by color, each containing 'grid_position', 'bbox', 'area_focus_point', 'color', and 'avg_hue'.
        dict_edge_sorted (list of dict): List of dictionaries sorted by edge, each containing 'grid_position', 'bbox', and 'area_focus_point'.
        x_groups1 (list): Groupings associated with dict_color_sorted along the x-axis.
        y_groups1 (list): Groupings associated with dict_color_sorted along the y-axis.
        x_groups2 (list): Groupings associated with dict_edge_sorted along the x-axis.
        y_groups2 (list): Groupings associated with dict_edge_sorted along the y-axis.
        faktor_color_dict (int, optional): Weighting factor for color dictionary entries. Default is 2.
        faktor_edge_dict (int, optional): Weighting factor for edge dictionary entries. Default is 1.
    Returns:
    tuple: A tuple containing:
        - merged_dict (dict): Merged dictionary with combined entries from both input dictionaries.
        - merged_x_group (list): Merged groupings along the x-axis.
        - merged_y_group (list): Merged groupings along the y-axis.
    """

    if dict_color_sorted is None and dict_edge_sorted is None:  # Check if both dictionaries are None
        return None, None, None

    if dict_color_sorted is None or any(entry['grid_position'] is None for entry in dict_color_sorted):
        merged_dict = {entry['grid_position']: entry for entry in dict_edge_sorted}  # Use edge dictionary if color dictionary is None or invalid
        if dict_edge_sorted is None:
            return None, None, None
        for entry in merged_dict.values():
            entry['detected'] = 'edge'  # Mark as detected by edge
        return merged_dict, x_groups2, y_groups2

    if dict_edge_sorted is None:  # Use color dictionary if edge dictionary is None
        merged_dict = {entry['grid_position']: entry for entry in dict_color_sorted}
        for entry in merged_dict.values():
            entry['detected'] = 'color'  # Mark as detected by color
        return merged_dict, x_groups1, y_groups1

    merged_dict = {}
    for entry in dict_color_sorted + dict_edge_sorted:
        pos = entry['grid_position']
        if pos not in merged_dict:
            merged_dict[pos] = entry  # Add entry to merged dictionary
            merged_dict[pos]['detected'] = 'color' if entry in dict_color_sorted else 'edge'  # Mark detection source
        else:
            merged_entry = merged_dict[pos]
            merged_entry['bbox'] = [
                (faktor_color_dict * merged_entry['bbox'][i] + faktor_edge_dict * entry['bbox'][i]) /
                (faktor_color_dict + faktor_edge_dict)  # Merge bounding boxes
                for i in range(4)
            ]
            merged_entry['area_focus_point'] = [
                int((faktor_color_dict * merged_entry['area_focus_point'][i] + faktor_edge_dict * entry['area_focus_point'][i]) /
                (faktor_color_dict + faktor_edge_dict))  # Merge area focus points
                for i in range(2)
            ]
            # Preserve color and hue values if present
            merged_entry['color'] = merged_entry.get('color') if merged_entry.get('color') is not None else entry.get('color')
            merged_entry['avg_hue'] = merged_entry.get('avg_hue') if merged_entry.get('avg_hue') is not None else entry.get('avg_hue')

            merged_entry['detected'] = 'both'  # Mark as detected by both

    # Merge groups
    merged_x_group = merge_groups(x_groups1, x_groups2, faktor_color_dict, faktor_edge_dict)  
    merged_y_group = merge_groups(y_groups1, y_groups2, faktor_color_dict, faktor_edge_dict) 

    return merged_dict, merged_x_group, merged_y_group  

def interpolate_missing_entries(merged_dict, merged_group_x, merged_group_y):
    """
    Interpolates missing entries in a grid dictionary.
    This function takes a dictionary representing a grid and interpolates missing entries
    based on the provided x and y coordinate groups. If either of the coordinate groups
    is None, the original dictionary is returned without any modifications.
    Args:
        merged_dict (dict): The dictionary containing grid entries. Each key is a grid position (1-9),
                            and the value is a dictionary with details about the grid cell.
        merged_group_x (list or None): A list of x-coordinates for the grid cells. If None, no interpolation
                                       is performed for x-coordinates.
        merged_group_y (list or None): A list of y-coordinates for the grid cells. If None, no interpolation
                                       is performed for y-coordinates.
    Returns:
        dict: The updated dictionary with interpolated entries for missing grid positions.
              Each interpolated entry contains:
                - "bbox": None
                - "area_focus_point": A tuple with interpolated x and y coordinates
                - "color": None
                - "grid_position": The grid position (1-9)
                - "detected": "interpolated"
                - "avg_hue": None
    """
    if merged_group_x is None or merged_group_y is None: 
        return merged_dict
    
    # Calculate mean coordinates
    x_mean = np.mean(merged_group_x) if merged_group_x else None  
    y_mean = np.mean(merged_group_y) if merged_group_y else None 

    for pos in range(1, 10):
        if pos not in merged_dict:  # Check if position is missing
            row, col = divmod(pos - 1, 3)
            interpolated_entry = {
                "bbox": None,
                "area_focus_point": (int(merged_group_x[col] if col < len(merged_group_x) else x_mean),
                                     int(merged_group_y[row] if row < len(merged_group_y) else y_mean)),  # Interpolate coordinates
                "color": None,
                "grid_position": pos,
                "detected": "interpolated", 
                "avg_hue": None
            }
            # Add interpolated entry to dictionary
            merged_dict[pos] = interpolated_entry  

    return merged_dict 
