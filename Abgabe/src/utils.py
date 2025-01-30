# Projekt:      Rubiks Cube
# Participants: Lukas Gerstlauer, 205293, lgerstla@stud.hs-heilbronn.de
#               Tim Söns, 204453, tsoens@stud.hs-heilbronn.de
# Lecture:      Computer & Robot Vision
# Program:      MAS
# Date:         31.01.2025

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_images(image, corrected_image, edge_mask, color_mask, outer_grey_frame, reduced_outer_grey_frame, inner_grey_frame, final_dict, correction_values):
    """
    Draws and displays various images and information related to image processing.
    Parameters:
        image (numpy.ndarray): The original image.
        corrected_image (numpy.ndarray): The corrected version of the original image.
        edge_mask (numpy.ndarray): The edge mask of the image.
        color_mask (numpy.ndarray): The color mask of the image.
        outer_grey_frame (tuple): Coordinates and dimensions of the outer grey frame (x, y, width, height).
        reduced_outer_grey_frame (tuple): Coordinates and dimensions of the reduced outer grey frame (x, y, width, height).
        inner_grey_frame (tuple): Coordinates and dimensions of the inner grey frame (x, y, width, height).
        final_dict (dict): Dictionary containing information about detected areas, including their positions, colors, focus points, and detection status.
        correction_values (list): List of correction values for the image.
    Returns:
        None
    """
    # Create a new figure for display
    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])
    
    # Column 2, top 2/3: Processed original image
    ax1 = fig.add_subplot(grid[0:2, 1])
    # Draw the bounding boxes
    cv2.rectangle(image, (outer_grey_frame[0], outer_grey_frame[1]), 
                  (outer_grey_frame[0] + outer_grey_frame[2], outer_grey_frame[1] + outer_grey_frame[3]), 
                  (0, 255, 0), 2)
    cv2.rectangle(image, (reduced_outer_grey_frame[0], reduced_outer_grey_frame[1]), 
                  (reduced_outer_grey_frame[0] + reduced_outer_grey_frame[2], reduced_outer_grey_frame[1] + reduced_outer_grey_frame[3]), 
                  (255, 0, 0), 2)
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_images(image, corrected_image, edge_mask, color_mask, outer_grey_frame, reduced_outer_grey_frame, inner_grey_frame, final_dict, correction_values):
    """
    Draws and displays various images and information related to image processing.
    Parameters:
        image (numpy.ndarray): The original image.
        corrected_image (numpy.ndarray): The corrected version of the original image.
        edge_mask (numpy.ndarray): The edge mask of the image.
        color_mask (numpy.ndarray): The color mask of the image.
        outer_grey_frame (tuple): Coordinates and dimensions of the outer grey frame (x, y, width, height).
        reduced_outer_grey_frame (tuple): Coordinates and dimensions of the reduced outer grey frame (x, y, width, height).
        inner_grey_frame (tuple): Coordinates and dimensions of the inner grey frame (x, y, width, height).
        final_dict (dict): Dictionary containing information about detected areas, including their positions, colors, focus points, and detection status.
        correction_values (list): List of correction values for the image.
    Returns:
        None
    """
    # Create a new figure for display
    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])
    
    # Column 2, top 2/3: Processed original image
    ax1 = fig.add_subplot(grid[0:2, 1])
    # Draw the bounding boxes
    cv2.rectangle(image, (outer_grey_frame[0], outer_grey_frame[1]), 
                  (outer_grey_frame[0] + outer_grey_frame[2], outer_grey_frame[1] + outer_grey_frame[3]), 
                  (0, 255, 0), 2)
    cv2.rectangle(image, (reduced_outer_grey_frame[0], reduced_outer_grey_frame[1]), 
                  (reduced_outer_grey_frame[0] + reduced_outer_grey_frame[2], reduced_outer_grey_frame[1] + reduced_outer_grey_frame[3]), 
                  (255, 0, 0), 2)
    cv2.rectangle(image, (inner_grey_frame[0], inner_grey_frame[1]), 
                  (inner_grey_frame[0] + inner_grey_frame[2], inner_grey_frame[1] + inner_grey_frame[3]), 
                  (255, 0, 0), 2)

    # Draw the position numbers
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

    # Column 2, bottom 1/3: Legend table
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
    df = df.sort_values(by="Position")  # Sort the table by position
    ax2.axis("off")
    table = ax2.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Limit the width of the table
    for key, cell in table.get_celld().items():
        cell.set_width(0.2)  # Set the width of each cell to 0.2
        cell.set_height(0.12)  # Set the height of each cell to 0.12

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

    # Limit the width of the table
    for key, cell in table2.get_celld().items():
        cell.set_width(0.2)  # Set the width of each cell to 0.2
        cell.set_height(0.2)  # Set the height of each cell to 0.2

    # Column 1, row 1: Corrected image
    ax3 = fig.add_subplot(grid[0, 0])
    ax3.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
    ax3.set_title("Corrected Image")
    ax3.axis("off")

    # Column 1, row 2: Edge mask
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.imshow(cv2.cvtColor(edge_mask, cv2.COLOR_BGR2RGB))
    ax4.set_title("Edge Mask")
    ax4.axis("off")

    # Column 1, row 3: Colored mask
    ax5 = fig.add_subplot(grid[2, 0])
    ax5.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    ax5.set_title("Colored Mask")
    ax5.axis("off")

    # Show the layout
    plt.tight_layout()
    plt.show()

def close_plot(event):
    """
    Closes the plot window when a specific key is pressed.

    Parameters:
    event (matplotlib.backend_bases.KeyEvent): The key event that triggers the function. 
                                               The plot window will close if the key specified 
                                               in the function (default is 'q') is pressed.
    """
    if event.key == 'q':  # Customize the key if needed
        plt.close(event.canvas.figure)
