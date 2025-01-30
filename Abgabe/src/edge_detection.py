# Projekt:      Rubiks Cube
# Participants: Lukas Gerstlauer, 205293, lgerstla@stud.hs-heilbronn.de
#               Tim Söns, 204453, tsoens@stud.hs-heilbronn.de
# Lecture:      Computer & Robot Vision
# Program:      MAS
# Date:         31.01.2025

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def edge_detection(image):
    """
    Perform edge detection on the given image and return a list of detected elements with their attributes.
    Parameters:
    image (numpy.ndarray): Input image in BGR format.
    Returns:
    list_dict_edge (list): A list of dictionaries, each containing attributes of detected elements:
        - bbox (tuple): Bounding box of the detected element (x, y, width, height).
        - area_focus_point (list): Coordinates of the focus point of the detected area [cx, cy].
        - color (None): Placeholder for color attribute (not used in this function).
        - grid_position (None): Placeholder for grid position attribute (not used in this function).
        - avg_hue (None): Placeholder for average hue attribute (not used in this function).
    clean_mask (numpy.ndarray): The cleaned and sharpened mask used for contour detection.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Threshold for Black Regions
    # Black pixels will have low intensity in grayscale
    _, black_mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # Display the Black Mask
    # plt.figure(figsize=(6, 6))
    # plt.imshow(black_mask, cmap='gray')
    # plt.title("Black Mask")
    # plt.axis("off")
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 2: Morphological Operations to Clean the Mask (with stronger effect)
    # Create a larger kernel to increase intensity
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Increase kernel size for more intense effect

    # Apply MORPH_CLOSE to fill gaps and remove small noise
    clean_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Increase iterations for stronger effect

    # Apply a morphological gradient to sharpen edges
    gradient_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_GRADIENT, kernel)

    # Combine the cleaned mask with the gradient mask to highlight edges
    clean_mask = cv2.bitwise_or(clean_mask, gradient_mask)

    # Convert the cleaned mask to BGR for colored drawing
    clean_mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

    # Display the cleaned and sharpened mask
    # plt.figure(figsize=(6, 6))
    # plt.imshow(clean_mask, cmap='gray')
    # plt.title("Cleaned and Sharpened Mask")
    # plt.axis("off")
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Step 3: Find Contours in the Cleaned Mask
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Filter for Rectangular Contours
    valid_rectangles = []
    list_dict_edge = []

    for contour in contours:
        # Approximate the contour with a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Calculate area to filter small or large regions
        area = cv2.contourArea(approx)
        if (math.sqrt(area) > math.sqrt(width * height)/6) & (math.sqrt(area) < math.sqrt(width * height)/3):
            # Calculate centres of gravity
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                area_focus_point = [cx, cy]
            else:
                area_focus_point = [None, None]

            # Save contour
            valid_rectangles.append(approx)

            # Create entry in the attribute dictionary
            valid_element = {
                "bbox": cv2.boundingRect(approx),
                "area_focus_point": area_focus_point,
                "color": None,
                "grid_position": None,
                "avg_hue": None,
            }
            list_dict_edge.append(valid_element)

    # # Display the Result
    # plt.figure(figsize=(6, 6))
    # plt.imshow(cv2.cvtColor(clean_mask_bgr, cv2.COLOR_BGR2RGB))
    # plt.title("Detected Elements with Bounding Boxes and Focus Points")
    # plt.axis("off")
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        
    return list_dict_edge, clean_mask