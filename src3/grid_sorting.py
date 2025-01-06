import numpy as np

def group_coordinates(coords, threshold=50, max_groups=3):
    """Gruppiert Koordinaten entlang einer Achse und gibt Gruppendurchschnitte zurück."""
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

    # Gruppendurchschnitt berechnen und auf max_groups begrenzen
    group_means = [np.mean(group) for group in groups]
    if len(group_means) > max_groups:
        group_means = sorted(group_means)[:max_groups]

    return sorted(group_means)


def assign_grid_positions(entries, threshold=50):
    """
    Ordnet Einträgen basierend auf ihren Koordinaten Grid-Positionen zu
    und gibt die Gruppen zurück.
    """
    x_coords = [entry['area_focus_point'][0] for entry in entries]
    y_coords = [entry['area_focus_point'][1] for entry in entries]

    # Gruppiere Koordinaten und finde Mittelwerte der Gruppen
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
        entry['grid_position'] = grid_position

    return entries, x_groups, y_groups

def merge_and_interpolate(dict_color_sorted, x_groups1, y_groups1, dict_edge_sorted, x_groups2, y_groups2, threshold=50):
    """
    Kombiniert zwei Dictionaries, berücksichtigt Gruppeninformationen und interpoliert fehlende Werte.
    """
    # Zusammenführen der Dictionaries
    merged_dict = {}
    for entry in dict_color_sorted + dict_edge_sorted:
        pos = entry['grid_position']
        if pos not in merged_dict:
            merged_dict[pos] = entry
            merged_dict[pos]['detected'] = 'color' if entry in dict_color_sorted else 'edge'
        else:
            # Kombiniere Informationen, falls doppelt vorhanden
            merged_dict[pos] = {
                key: entry[key] if entry[key] is not None else merged_dict[pos].get(key, None)
                for key in entry
            }
            merged_dict[pos]['detected'] = 'both'

    # Fehlende Positionen interpolieren
    for pos in range(1, 10):
        if pos not in merged_dict:
            row, col = divmod(pos - 1, 3)
            x_mean = np.mean(x_groups1 + x_groups2) if x_groups1 + x_groups2 else None
            y_mean = np.mean(y_groups1 + y_groups2) if y_groups1 + y_groups2 else None

            interpolated_entry = {
                "bbox": None,
                "area_focus_point": [x_mean, y_mean],
                "color": 'Unknown',
                "grid_position": pos,
                "detected": "interpolated",
                "avg_hue": None
            }
            merged_dict[pos] = interpolated_entry

    return merged_dict
