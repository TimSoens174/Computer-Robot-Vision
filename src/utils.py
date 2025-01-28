import cv2
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_images(image, corrected_image, edge_mask, color_mask, outer_grey_frame, reduced_outer_grey_frame, inner_grey_frame, final_dict, correction_values):
    # Erstelle ein neues Bild für die Anzeige
    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])
    
    # Spalte 2, obere 2/3: Bearbeitetes Originalbild
    ax1 = fig.add_subplot(grid[0:2, 1])
    # Zeichne die Bounding Boxen
    cv2.rectangle(image, (outer_grey_frame[0], outer_grey_frame[1]), 
                  (outer_grey_frame[0] + outer_grey_frame[2], outer_grey_frame[1] + outer_grey_frame[3]), 
                  (0, 255, 0), 2)
    cv2.rectangle(image, (reduced_outer_grey_frame[0], reduced_outer_grey_frame[1]), 
                  (reduced_outer_grey_frame[0] + reduced_outer_grey_frame[2], reduced_outer_grey_frame[1] + reduced_outer_grey_frame[3]), 
                  (255, 0, 0), 2)
    cv2.rectangle(image, (inner_grey_frame[0], inner_grey_frame[1]), 
                  (inner_grey_frame[0] + inner_grey_frame[2], inner_grey_frame[1] + inner_grey_frame[3]), 
                  (255, 0, 0), 2)

    # Zeichne die Positionsnummern
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

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Processed Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(grid[2, 1])
    legend_data = {
        "Position": [],
        "Color": [],
        "Focus Point": [],
        "Detected": []
    }
    for pos, entry in final_dict.items():
        legend_data["Position"].append(pos)
        legend_data["Color"].append(entry.get('color', 'None') if entry.get('color') is not None else 'None')
        legend_data["Focus Point"].append(entry['area_focus_point'])
        legend_data["Detected"].append(entry['detected'])

    df = pd.DataFrame(legend_data)
    df = df.sort_values(by="Position")  # Sortiere die Tabelle nach der Position
    ax2.axis("off")
    table = ax2.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Begrenze die Breite der Tabelle
    for key, cell in table.get_celld().items():
        cell.set_width(0.2)  # Setze die Breite jeder Zelle auf 0.2
        cell.set_height(0.12)  # Setze die Höhe jeder Zelle auf 0.2

    # Create a new axis for the second table within the same grid cell
    divider = make_axes_locatable(ax2)
    ax3 = divider.append_axes("bottom", size="70%", pad=0.05)  # Reduced pad value to decrease the distance
    ax3.axis("off")
    rounded_correction_values = [round(value, 2) for value in correction_values]
    col_labels = ['ĉ$_0$', 'ĉ$_r$', 'ĉ$_g$', 'ĉ$_b$']
    table2 = ax3.table(cellText=[rounded_correction_values], colLabels=col_labels, cellLoc='center', loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.2)

    # Begrenze die Breite der Tabelle
    for key, cell in table2.get_celld().items():
        cell.set_width(0.2)  # Setze die Breite jeder Zelle auf 0.2
        cell.set_height(0.2)  # Setze die Höhe jeder Zelle auf 0.2

  

    # Spalte 1, Reihe 1: Korrigiertes Bild
    ax3 = fig.add_subplot(grid[0, 0])
    ax3.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    ax3.set_title("Corrected Image")
    ax3.axis("off")

    # Spalte 1, Reihe 2: Edge-Maske
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.imshow(cv2.cvtColor(edge_mask, cv2.COLOR_BGR2RGB))
    ax4.set_title("Edge Mask")
    ax4.axis("off")

    # Spalte 1, Reihe 3: Farbige Maske
    ax5 = fig.add_subplot(grid[2, 0])
    ax5.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    ax5.set_title("Colored Mask")
    ax5.axis("off")

    # Zeige das Layout
    plt.tight_layout()
    plt.show()

def close_plot(event):
    """Closes the plot when the 'q' key is pressed."""
    if event.key == 'q':  # Customize the key if needed
        plt.close(event.canvas.figure)