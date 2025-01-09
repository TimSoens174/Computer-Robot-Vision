import cv2
import matplotlib.pyplot as plt

def draw_images(image, corrected_image, edge_mask, color_mask, outer_grey_frame, reduced_outer_grey_frame, inner_grey_frame, final_dict):
    # Stelle sicher, dass alle Bilder die gleiche Größe haben
    height, width = image.shape[:2]
    
    # Erstelle ein neues Bild für die Anzeige
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [2, 1]})
    
    # Originalbild mit Bounding Box und Beschriftung
    ax1 = axes[0, 0]
    
    
    # Zeichne alle Bounding Boxen auf das Originalbild (adjusted coordinates)
    cv2.rectangle(image, (outer_grey_frame[0], outer_grey_frame[1]), (outer_grey_frame[0] + outer_grey_frame[2], outer_grey_frame[1] + outer_grey_frame[3]), (0, 255, 0), 2)
    cv2.rectangle(image, (reduced_outer_grey_frame[0], reduced_outer_grey_frame[1]), (reduced_outer_grey_frame[0] + reduced_outer_grey_frame[2], reduced_outer_grey_frame[1] + reduced_outer_grey_frame[3]), (255, 0, 0), 2)
    cv2.rectangle(image, (inner_grey_frame[0], inner_grey_frame[1]), (inner_grey_frame[0] + inner_grey_frame[2], inner_grey_frame[1] + inner_grey_frame[3]), (255, 0, 0), 2)
    
    # Zeichne die Positionsnummern und die Legende basierend auf dem final_dict
    for pos, entry in final_dict.items():
        area_focus_point = entry['area_focus_point']
        grid_position = entry['grid_position']
        
        # Berechne den Textort und zeichne die Positionsnummer
        text = f"{grid_position}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (0, 0, 0)  # Schwarz
        thickness = 2

        # Text Position anpassen, um die Position in der Mitte des Focus-Punkts zu setzen
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int(area_focus_point[0] - text_size[0] // 2)  # Stelle sicher, dass text_x eine ganze Zahl ist
        text_y = int(area_focus_point[1] + text_size[1] // 2)  # Stelle sicher, dass text_y eine ganze Zahl ist

        # Text auf das Bild zeichnen
        cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)

    # Zeige die Legende
    legend_x = int(image.shape[1] * 0.8)  # Startpunkt der Legende
    legend_y = image.shape[0] - 50  # Startpunkt der Legende
    legend_font = cv2.FONT_HERSHEY_SIMPLEX
    legend_font_scale = 0.8
    legend_font_color = (0, 0, 0)  # Schwarz
    legend_thickness = 2
    line_height = 30  # Abstand zwischen den Zeilen der Legende
    
    # Generiere die Legende
    for pos, entry in final_dict.items():
        legend_text = f"Pos {entry['grid_position']}: {entry['color']}, {entry['detected']}"
        cv2.putText(image, legend_text, (legend_x, legend_y), legend_font, legend_font_scale, legend_font_color, legend_thickness)
        legend_y += line_height

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # BGR zu RGB umwandeln
    ax1.set_title("Original Image with Bounding Boxes")
    ax1.axis("off")

    # Korrigiertes Bild
    ax2 = axes[0, 1]
    ax2.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Corrected Image")
    ax2.axis("off")
    
    # Edge-Maske
    ax3 = axes[1, 0]
    ax3.imshow(cv2.cvtColor(edge_mask, cv2.COLOR_BGR2RGB))
    ax3.set_title("Edge Mask")
    ax3.axis("off")
    
    # Farbige Maske
    ax4 = axes[1, 1]
    ax4.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    ax4.set_title("Colored Mask")
    ax4.axis("off")
    
    # Zeige das Bild
    plt.tight_layout()
    plt.show()
