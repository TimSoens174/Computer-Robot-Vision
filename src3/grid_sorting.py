import numpy as np

def group_coordinates(coords, threshold=50, required_groups=3):
    """Gruppiert Koordinaten entlang einer Achse, erzwingt eine Mindestanzahl von Gruppen."""
    if len(coords) < required_groups:
        return None

    coords = sorted(coords)
    groups = []
    current_group = [coords[0]]

    for i in range(1, len(coords)):
        if coords[i] - coords[i - 1] <= threshold:
            current_group.append(coords[i])
        else:
            groups.append(current_group)
            current_group = [coords[i]]
    groups.append(current_group)

    group_means = [np.mean(group) for group in groups]
    return sorted(group_means)

def assign_grid_positions(entries):
    """
    Ordnet Einträgen basierend auf ihren Koordinaten Grid-Positionen zu
    und gibt die Gruppen zurück.
    """
    if not entries:
        return None, None, None
    
    x_coords = [entry['area_focus_point'][0] for entry in entries]
    y_coords = [entry['area_focus_point'][1] for entry in entries]

    total_width, total_height = 0, 0

    for entry in entries:
        bbox = entry['bbox']
        total_width += bbox[2]
        total_height += bbox[3]

    average_width = total_width / len(entries)
    average_height = total_height / len(entries)
    threshold = (average_width + average_height) / 4

    x_groups = group_coordinates(x_coords, threshold, required_groups=3)
    y_groups = group_coordinates(y_coords, threshold, required_groups=3)

    if x_groups is None or y_groups is None:
        return None, None, None

    x_groups.sort()
    y_groups.sort()

    for entry in entries:
        x, y = entry['area_focus_point']
        col = np.argmin([abs(x - group) for group in x_groups])
        row = np.argmin([abs(y - group) for group in y_groups])
        grid_position = row * 3 + col + 1
        entry['grid_position'] = grid_position

    return entries, x_groups, y_groups

def merge_groups(group1, group2, faktor1=2, faktor2=1):
    """
    Kombiniert zwei Gruppen und gibt die kombinierte Gruppe zurück.
    - Falls eine der Gruppen None ist, wird die andere zurückgegeben.
    - Falls beide Gruppen None sind, wird None zurückgegeben.
    - Falls beide Gruppen existieren, wird ein gewichteter Durchschnitt berechnet.
    """
    if group1 is None and group2 is None:
        return None
    if group1 is None:
        return group2
    if group2 is None:
        return group1
    
    merged_group = [
        (faktor1 * g1 + faktor2 * g2) / (faktor1 + faktor2)
        for g1, g2 in zip(sorted(group1), sorted(group2))
    ]
    return merged_group

def merge_dictionaries(dict_color_sorted, dict_edge_sorted, x_groups1, y_groups1, x_groups2, y_groups2, faktor_color_dict=2, faktor_edge_dict=1):
    """
    Kombiniert zwei Dictionaries und die zugehörigen Gruppen, gibt das kombinierte Ergebnis zurück.
    """
    if dict_color_sorted is None and dict_edge_sorted is None:
        return None, None, None

    if dict_color_sorted is None:
        merged_dict = {entry['grid_position']: entry for entry in dict_edge_sorted}
        for entry in merged_dict.values():
            entry['detected'] = 'edge'
        return merged_dict, x_groups2, y_groups2

    if dict_edge_sorted is None:
        merged_dict = {entry['grid_position']: entry for entry in dict_color_sorted}
        for entry in merged_dict.values():
            entry['detected'] = 'color'
        return merged_dict, x_groups1, y_groups1

    merged_dict = {}
    for entry in dict_color_sorted + dict_edge_sorted:
        pos = entry['grid_position']
        if pos not in merged_dict:
            merged_dict[pos] = entry
            merged_dict[pos]['detected'] = 'color' if entry in dict_color_sorted else 'edge'
        else:
            merged_entry = merged_dict[pos]
            merged_entry['bbox'] = [
                (faktor_color_dict * merged_entry['bbox'][i] + faktor_edge_dict * entry['bbox'][i]) /
                (faktor_color_dict + faktor_edge_dict)
                for i in range(4)
            ]
            merged_entry['area_focus_point'] = [
                (faktor_color_dict * merged_entry['area_focus_point'][i] + faktor_edge_dict * entry['area_focus_point'][i]) /
                (faktor_color_dict + faktor_edge_dict)
                for i in range(2)
            ]
            # Sichern der Farb- und Hue-Werte (nur wenn sie vorhanden sind)
            merged_entry['color'] = merged_entry.get('color') if merged_entry.get('color') is not None else entry.get('color')
            merged_entry['avg_hue'] = merged_entry.get('avg_hue') if merged_entry.get('avg_hue') is not None else entry.get('avg_hue')

            
            merged_entry['detected'] = 'both'


    merged_x_group = merge_groups(x_groups1, x_groups2, faktor_color_dict, faktor_edge_dict)
    merged_y_group = merge_groups(y_groups1, y_groups2, faktor_color_dict, faktor_edge_dict)

    return merged_dict, merged_x_group, merged_y_group

def interpolate_missing_entries(merged_dict, merged_group_x, merged_group_y):
    """
    Interpoliert fehlende Einträge in einem kombinierten Dictionary und nutzt die gemergten Gruppen.
    """
    if merged_group_x is None or merged_group_y is None:
        return merged_dict

    x_mean = np.mean(merged_group_x) if merged_group_x else None
    y_mean = np.mean(merged_group_y) if merged_group_y else None

    for pos in range(1, 10):
        if pos not in merged_dict:
            row, col = divmod(pos - 1, 3)
            interpolated_entry = {
                "bbox": None,
                "area_focus_point": [merged_group_x[col] if col < len(merged_group_x) else x_mean,
                                     merged_group_y[row] if row < len(merged_group_y) else y_mean],
                "color": 'None',
                "grid_position": pos,
                "detected": "interpolated",
                "avg_hue": None
            }
            merged_dict[pos] = interpolated_entry

    return merged_dict
