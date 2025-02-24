# Projekt:      Rubiks Cube
# Participants: Lukas Gerstlauer, 205293, lgerstla@stud.hs-heilbronn.de
#               Tim Söns, 204453, tsoens@stud.hs-heilbronn.de
# Lecture:      Computer & Robot Vision
# Program:      MAS
# Date:         31.01.2025

import cv2
import os
import image_preprocessing as ip
import edge_detection as edge
import color_filtering as color
import grid_sorting as sorting
import utils
import time

# Parameters for image processing
LIVECAM = False
CAMERA = 0
IMAGE_PATH = os.path.join("Pictures/Picture1.jpg")

# Color correction parameters
COLOR_CORRECTION = True
SIGMA_B = 18.84
SIGMA_G = 18.92
SIGMA_R = 18.23
MAX_B = 158
MAX_G = 140
MAX_R = 130
INNER_GREY_FRAME_FAKTOR = 0.9
OUTER_GREY_FRAME_FAKTOR = 0.95

COLOR_CORRECTION_GROUND_TRUTH = [SIGMA_B, SIGMA_G, SIGMA_R, MAX_B, MAX_G, MAX_R]

# Main function to process an image
def main(image):
    """
    Main function to process an image through various stages including detection, cropping, color correction, 
    edge detection, sorting, merging, interpolation, and final visualization.
    Args:
        image (numpy.ndarray): The input image to be processed.
    Returns:
        None
    The function performs the following steps:
    1. Checks if the image is loaded correctly.
    2. Detects the outer gray frame and reduces its bounding box.
    3. Crops the image based on the outer gray frame.
    4. Corrects the image colors using a predefined ground truth.
    5. Filters colors and detects edges in the corrected image.
    6. Groups and sorts the detected colors and edges.
    7. Merges the color and edge dictionaries and interpolates missing entries.
    8. Updates the color for entries with None values.
    9. Removes entries with None grid positions and sorts the final dictionary.
    10. Draws and displays the processed images and results.
    Raises:
        Exception: If any step in the image processing pipeline fails.
    """

    # Check if the image is loaded correctly
    if image is None:
        print("Error: The image could not be loaded. Please check the image path.")
        return

    # Detect the outer gray frame and reduce its bounding box
    outer_grey_frame = ip.detect_outer_gray_frame(image)
    reduced_outer_grey_frame = ip.reduce_boundingbox(outer_grey_frame, OUTER_GREY_FRAME_FAKTOR)
    inner_grey_frame = ip.reduce_boundingbox(reduced_outer_grey_frame, INNER_GREY_FRAME_FAKTOR)

    # Crop the image based on the outer gray frame
    cropped_image = ip.cut_image(image, outer_grey_frame)

    # Relocate Boundingbox in cropped frame
    cropped_outer_grey_frame = [0, 0, outer_grey_frame[2], outer_grey_frame[3]]

    # Reduce bounding boxes in the cropped frame
    croped_reduced_outer_grey_frame = ip.reduce_boundingbox(cropped_outer_grey_frame, OUTER_GREY_FRAME_FAKTOR)
    croped_inner_grey_frame = ip.reduce_boundingbox(croped_reduced_outer_grey_frame, INNER_GREY_FRAME_FAKTOR)

    # Correct the image colors using a predefined ground truth
    if COLOR_CORRECTION:
        corrected_image, correction_values = color.correctImage(cropped_image, COLOR_CORRECTION_GROUND_TRUTH, croped_reduced_outer_grey_frame, croped_inner_grey_frame)
        # color.showHistogram(cropped_image, corrected_image, croped_reduced_outer_grey_frame, croped_inner_grey_frame)
        if corrected_image is None:
            print("Error: Image correction failed.")
            return
    else:
        corrected_image = cropped_image
        correction_values = None
    
    # Filter colors in the corrected image
    dict_color, color_mask = color.color_filter(corrected_image)
    print("Color Dictionary:")
    for element in dict_color:
        print(element)

    # Detect edges in the corrected image
    dict_edge, edge_mask = edge.edge_detection(corrected_image)
    print("Edge Dictionary:")
    for element in dict_edge:
        print(element)

    # Group and sort the detected colors
    dict_color_sorted, x_groups1, y_groups1 = sorting.assign_grid_positions(dict_color)
    if x_groups1 is None or y_groups1 is None:
        print("Error: No color groups found.")
    else:
        print("Dict Color Groups:")
        print("X-Groups:", x_groups1)
        print("Y-Groups:", y_groups1)

    # Group and sort the detected edges
    dict_edge_sorted, x_groups2, y_groups2 = sorting.assign_grid_positions(dict_edge)
    if x_groups2 is None or y_groups2 is None:
        print("Error: No edge groups found.")
    else:
        print("\nDict Edge Groups:")
        print("X-Groups:", x_groups2)
        print("Y-Groups:", y_groups2)

    # Merge the dictionaries and groups
    merged_dict, merged_x_group, merged_y_group = sorting.merge_dictionaries(dict_color_sorted, dict_edge_sorted, x_groups1, y_groups1, x_groups2, y_groups2)

    # Interpolate missing entries
    interpolated_dict = sorting.interpolate_missing_entries(merged_dict, merged_x_group, merged_y_group)

    # Update color for entries with None values
    final_dict = color.update_color_for_none_entries(interpolated_dict, corrected_image)

    if final_dict is None:
        print("Error: Updating color for None entries failed.")
        return

    # Remove entries with None values for grid_position
    final_dict = {k: v for k, v in final_dict.items() if v['grid_position'] is not None}

    # Sort the dictionary based on grid_position
    sorted_final_dict = sorted(final_dict.items(), key=lambda x: x[1]['grid_position'])

    # Output the sorted entries
    print("\nCompensated Dictionary:")
    for pos, entry in sorted_final_dict:
        print(f"Position: {pos}, Entry: {entry}")

    # Draw and display the processed images and results
    try: 
        utils.draw_images(image, corrected_image, edge_mask, color_mask, outer_grey_frame, reduced_outer_grey_frame, inner_grey_frame, final_dict, correction_values)
    except Exception as e:
        print("Error: Drawing images failed.")
        print(e)
        return
    

if __name__ == "__main__":
    if LIVECAM:
        cap = cv2.VideoCapture(CAMERA)

        # Check if the camera could be opened
        if not cap.isOpened():
            print("Error: Camera could not be opened!")

        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            
            if not ret:
                print("Error reading the camera image!")
                break

            main(frame)
            time.sleep(0.3)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
    else: 
        image = cv2.imread(IMAGE_PATH)
        main(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
