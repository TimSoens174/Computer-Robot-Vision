import numpy as np

def group_coordinates(coords, threshold=50, max_groups=3):
    """
    Gruppiert ähnliche Koordinaten basierend auf einem Schwellenwert.
    Stellt sicher, dass nur maximal `max_groups` Gruppen zurückgegeben werden.
    """
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
    
    # Berechne die Mittelwerte der Gruppen
    group_means = [np.mean(group) for group in groups]
    
    # Reduziere auf max_groups Gruppen
    if len(group_means) > max_groups:
        group_means = sorted(group_means)[:max_groups]
    
    return sorted(group_means)

def assign_grid_positions(entries, threshold=50):
    """
    Ordnet jedem Element eine spezifische Grid-Position (Zeile, Spalte, 1-9) zu.
    """
    x_coords = [entry['area_focus_point'][0] for entry in entries]
    y_coords = [entry['area_focus_point'][1] for entry in entries]
    
    # Gruppiere ähnliche Koordinaten und beschränke auf maximal 3 Gruppen
    x_groups = group_coordinates(x_coords, threshold, max_groups=3)
    y_groups = group_coordinates(y_coords, threshold, max_groups=3)
    
    # Sortiere die Gruppen
    x_groups.sort()
    y_groups.sort()
    
    # Ordne jedem Eintrag eine Grid-Position zu
    for entry in entries:
        x, y = entry['area_focus_point']
        col = np.argmin([abs(x - group) for group in x_groups])
        row = np.argmin([abs(y - group) for group in y_groups])
        grid_position = row * 3 + col + 1  # Position 1-9 berechnen
        entry['grid_position'] = (row, col)
        entry['position_1_to_9'] = grid_position
    
    return entries, x_groups, y_groups

# Beispiel-Eingabe
entries = [
    {'bbox': [81, 335, 96, 96], 'area_focus_point': [129, 383], 'color': 'Rot', 'average_hue': 33.1},
    {'bbox': [199, 334, 96, 95], 'area_focus_point': [247, 381], 'color': 'Rot', 'average_hue': 34.4},
    {'bbox': [77, 97, 94, 92], 'area_focus_point': [124, 143], 'color': 'Gelb', 'average_hue': 29.5},
    {'bbox': [78, 217, 92, 94], 'area_focus_point': [124, 264], 'color': 'Grün', 'average_hue': 82.3},
    {'bbox': [315, 221, 93, 92], 'area_focus_point': [361, 267], 'color': 'Orange', 'average_hue': 8.2},
    {'bbox': [197, 219, 91, 91], 'area_focus_point': [242, 264], 'color': 'Weiß', 'average_hue': 3.15},
    {'bbox': [198, 102, 90, 91], 'area_focus_point': [243, 147], 'color': 'Weiß', 'average_hue': 2.57},
]

# Finde Grid-Positionen
assigned_entries, x_groups, y_groups = assign_grid_positions(entries, threshold=50)

# Ausgabe der Ergebnisse
for entry in assigned_entries:
    print(f"Element {entry['color']} hat die Grid-Position: {entry['grid_position']} "
          f"und Position 1-9: {entry['position_1_to_9']}")
