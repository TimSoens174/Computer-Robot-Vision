import cv2
import matplotlib.pyplot as plt


def draw_images(image, corrected_image, edge_mask, color_mask, outer_grey_frame, reduced_outer_grey_frame, inner_grey_frame, final_dict):
    # Erstelle ein neues Bild für die Anzeige
    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])
    
    # Spalte 2: Bearbeitetes Originalbild mit Bounding Boxes und Legenden (nimmt gesamte Spalte ein)
    ax1 = fig.add_subplot(grid[:, 1])
    # Zeichne die Bounding Boxen und Legenden
    cv2.rectangle(image, (outer_grey_frame[0], outer_grey_frame[1]), 
                  (outer_grey_frame[0] + outer_grey_frame[2], outer_grey_frame[1] + outer_grey_frame[3]), 
                  (0, 255, 0), 2)
    cv2.rectangle(image, (reduced_outer_grey_frame[0], reduced_outer_grey_frame[1]), 
                  (reduced_outer_grey_frame[0] + reduced_outer_grey_frame[2], reduced_outer_grey_frame[1] + reduced_outer_grey_frame[3]), 
                  (255, 0, 0), 2)
    cv2.rectangle(image, (inner_grey_frame[0], inner_grey_frame[1]), 
                  (inner_grey_frame[0] + inner_grey_frame[2], inner_grey_frame[1] + inner_grey_frame[3]), 
                  (255, 0, 0), 2)

    # Positionsnummern und Legenden zeichnen
    for pos, entry in final_dict.items():
        area_focus_point = entry['area_focus_point']
        grid_position = entry['grid_position']
        text = f"{grid_position}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (0, 0, 0)
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int(area_focus_point[0] + outer_grey_frame[0] - text_size[0] // 2)
        text_y = int(area_focus_point[1] + outer_grey_frame[1] + text_size[1] // 2)
        cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, thickness)

    # Legende zeichnen
    legend_x = int(image.shape[1] * 0.7)
    legend_y = int(image.shape[0] * 0.5)
    legend_font = cv2.FONT_HERSHEY_SIMPLEX
    legend_font_scale = image.shape[0] * 0.0015
    legend_font_color = (0, 0, 0)
    legend_thickness = 2
    line_height = int(image.shape[0] * 0.05)
    for pos, entry in sorted(final_dict.items(), key=lambda x: x[1]['grid_position']):
        legend_text = f"{entry['grid_position']}: {entry['color']}, {entry['detected']}"
        cv2.putText(image, legend_text, (legend_x, legend_y), legend_font, legend_font_scale, legend_font_color, legend_thickness)
        legend_y += line_height

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image with Bounding Boxes")
    ax1.axis("off")

    # Spalte 1, Reihe 1: Korrigiertes Bild
    ax2 = fig.add_subplot(grid[0, 0])
    ax2.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Corrected Image")
    ax2.axis("off")

    # Spalte 1, Reihe 2: Edge-Maske
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.imshow(cv2.cvtColor(edge_mask, cv2.COLOR_BGR2RGB))
    ax3.set_title("Edge Mask")
    ax3.axis("off")

    # Spalte 1, Reihe 3: Farbige Maske
    ax4 = fig.add_subplot(grid[2, 0])
    ax4.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    ax4.set_title("Colored Mask")
    ax4.axis("off")

    # Add the keypress event
    #fig.canvas.mpl_connect('key_press_event', close_plot)

    # Zeige das Layout
    plt.tight_layout()
    plt.show()

def close_plot(event):
    """Closes the plot when the 'q' key is pressed."""
    if event.key == 'q':  # Customize the key if needed
        plt.close(event.canvas.figure)
