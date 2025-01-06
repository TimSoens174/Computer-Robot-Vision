import cv2
def draw_grid_positions(image, merged_dict, outer_grey_frame):
    # Kopie des Originalbildes für die Bearbeitung
    debug_image = image.copy()

    # Zeichne die grid_position als Zahl (nur die Position) an den area_focus_point
    for pos, entry in merged_dict.items():
        grid_position = entry['grid_position']
        area_focus_point = entry['area_focus_point']
        color = entry['color']
        detected = entry['detected']

        # Korrektur der Koordinaten, um den Versatz (outer_grey_frame) zu berücksichtigen
        corrected_x = area_focus_point[0] + outer_grey_frame[0]
        corrected_y = area_focus_point[1] + outer_grey_frame[1]

        # Text mit der Positionsnummer anstatt der kompletten Info
        text = f"{grid_position}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_color = (0, 0, 0)  # Rot
        thickness = 3

        # Position für den Text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = corrected_x - text_size[0] // 2
        text_y = corrected_y + text_size[1] // 2

        # Text auf das Bild zeichnen (nur die Position)
        cv2.putText(debug_image, text, (text_x, text_y), font, font_scale, font_color, thickness)

    # Erstelle eine Legende unten rechts
    legend_x = image.shape[1] - 400  # Startpunkt der Legende
    legend_y = image.shape[0] - 400  # Startpunkt der Legende
    legend_font = cv2.FONT_HERSHEY_SIMPLEX
    legend_font_scale = 0.8
    legend_font_color = (0, 0, 0)  # Rot
    legend_thickness = 2
    line_height = 40  # Abstand zwischen den Zeilen der Legende

    # Zeige die Legende für jede Position im Grid
    for i in range(1, 10):
        # Suche den entsprechenden Eintrag
        entry = next((item for item in merged_dict.values() if item['grid_position'] == i), None)
        if entry:
            # Hier geben wir die vollständige Legende aus (Position, Farbe, Detected)
            legend_text = f"Pos {i}: {entry['color']}, {entry['detected']}"
            # Text auf das Bild zeichnen
            cv2.putText(debug_image, legend_text, (legend_x, legend_y), legend_font, legend_font_scale, legend_font_color, legend_thickness)
            legend_y += line_height  # Erhöhe die Y-Position für die nächste Zeile

    # Zeige das Bild an
    cv2.imshow("Debug Image with Grid Info", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # Beispielaufruf der Funktion
# if __name__ == "__main__":
#     image_path = os.path.join("Pictures2", "Picture 15.jpg")
#     image = cv2.imread(image_path)
    
#     # Angenommen, merged_dict wurde vorher schon erstellt
#     merged_dict = {
#         1: {'grid_position': 1, 'area_focus_point': (121, 143), 'color': 'Gelb', 'detected': 'both'},
#         2: {'grid_position': 2, 'area_focus_point': (242, 146), 'color': 'Gelb', 'detected': 'both'},
#         3: {'grid_position': 3, 'area_focus_point': (365, 148), 'color': 'Grün', 'detected': 'both'},
#         4: {'grid_position': 4, 'area_focus_point': (119, 264), 'color': None, 'detected': 'edge'},
#         5: {'grid_position': 5, 'area_focus_point': (240, 264), 'color': 'Gelb', 'detected': 'both'},
#         6: {'grid_position': 6, 'area_focus_point': (362, 268), 'color': 'Gelb', 'detected': 'both'},
#         7: {'grid_position': 7, 'area_focus_point': (126, 381), 'color': 'Rot', 'detected': 'both'},
#         8: {'grid_position': 8, 'area_focus_point': (245, 383), 'color': 'Rot', 'detected': 'both'},
#         9: {'grid_position': 9, 'area_focus_point': (361, 381), 'color': 'Gelb', 'detected': 'both'}
#     }
#     outer_grey_frame = [100, 100, 500, 500]  # Beispielwert für den Versatz

#     draw_grid_positions(image, merged_dict, outer_grey_frame)
