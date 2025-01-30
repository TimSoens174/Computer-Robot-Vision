# Projekt:      Rubiks Cube
# Participants: Lukas Gerstlauer, 205293, lgerstla@stud.hs-heilbronn.de
#               Tim Söns, 204453, tsoens@stud.hs-heilbronn.de
# Lecture:      Computer & Robot Vision
# Program:      MAS
# Date:         31.01.2025

import cv2
import numpy as np

def cut_image(image, boundingbox):
    """
    Cuts a sub-image from the given image using the specified bounding box.

    Parameters:
    image (numpy.ndarray): The input image from which to cut the sub-image.
    boundingbox (tuple): A tuple of four integers (x, y, w, h) representing the 
                         top-left corner (x, y) and the width (w) and height (h) 
                         of the bounding box.

    Returns:
    numpy.ndarray: The sub-image defined by the bounding box.
    """
    x = boundingbox[0]  # Extract the x-coordinate of the top-left corner
    y = boundingbox[1]  # Extract the y-coordinate of the top-left corner
    w = boundingbox[2]  # Extract the width of the bounding box
    h = boundingbox[3]  # Extract the height of the bounding box
    img = image.copy() 

    return img[y:y+h, x:x+w]

def detect_outer_gray_frame(image):
    """
    Detects the outer gray frame in an image and returns the bounding box coordinates.
    Parameters:
    image (numpy.ndarray): The input image in which to detect the outer gray frame.
    Returns:
    list: A list containing the x, y coordinates of the top-left corner, and the width and height of the bounding box 
            of the detected outer gray frame. If no frame is detected, returns [0, 0, 0, 0].
    """
    bild = image.copy()
    x,y,w,h = 0,0,0,0
    gray = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)

    # Smooth the image
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detect edges (Canny) with adjusted thresholds
    edges = cv2.Canny(blurred, 10, 20, L2gradient=True)

    # Close gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest rectangular contour
    largest_contour = None
    max_area = 0

    for contour in contours:
        # Compute contour approximation
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # Rectangles have 4 points
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:  # Only store the largest contour
                max_area = area
                largest_contour = approx

    if largest_contour is not None:
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
    
    return [x,y,w,h]

def reduce_boundingbox(boundingbox, Faktor):
    """
    Reduces the size of a bounding box by a given factor.
    Parameters:
    boundingbox (list or tuple): A list or tuple containing four elements [x, y, w, h] where
                                    x and y are the coordinates of the top-left corner,
                                    w is the width, and h is the height of the bounding box.
    Faktor (float): The factor by which to reduce the width and height of the bounding box.
    Returns:
    list: A list containing four elements [small_x, small_y, small_w, small_h] where
            small_x and small_y are the coordinates of the top-left corner of the reduced bounding box,
            small_w is the reduced width, and small_h is the reduced height.
    """
    x = boundingbox[0]
    y = boundingbox[1]
    w = boundingbox[2]
    h = boundingbox[3]
    center_x, center_y = x + w // 2, y + h // 2  # Center of the box
    small_w, small_h = int(w * Faktor), int(h * Faktor)    # 15% smaller width and height
    small_x, small_y = center_x - small_w // 2, center_y - small_h // 2  # Top-left corner

    return [small_x, small_y, small_w, small_h]
