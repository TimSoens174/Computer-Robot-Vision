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

def getMaskPixels(image, outer_correction_frame, inner_correction_frame):
    """
    Extracts the pixels from an image that lie within a specified region defined by two rectangular frames.
    Parameters:
    image (numpy.ndarray): The input image from which to extract pixels.
    outer_correction_frame (tuple): A tuple (x, y, width, height) defining the outer rectangular frame.
    inner_correction_frame (tuple): A tuple (x, y, width, height) defining the inner rectangular frame.
    Returns:
    numpy.ndarray: An array of pixels that lie in the region between the outer and inner rectangular frames.
    """
    
    height, width = image.shape[:2]

    # Mask for the large box
    mask_large = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask_large, (outer_correction_frame[0], outer_correction_frame[1]), (outer_correction_frame[0] + outer_correction_frame[2], outer_correction_frame[1] + outer_correction_frame[3]), 255, -1)

    # Mask for the small box
    mask_small = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask_small, (inner_correction_frame[0], inner_correction_frame[1]), (inner_correction_frame[0] + inner_correction_frame[2], inner_correction_frame[1] + inner_correction_frame[3]), 255, -1)

    # Area between the boxes: large mask minus small mask
    mask_between = cv2.subtract(mask_large, mask_small)

    # Extract pixels in the area
    all_pixels = image[mask_between == 255]
    pixels = all_pixels[~np.all(all_pixels[:,:] == 0, axis=1)]

    return pixels


def getImageValues(input_image, Box, smallBox):
    """
    Calculate the standard deviations and peak histogram values for the blue, green, and red channels 
    of the pixels within a specified region of an image.
    Parameters:
    input_image (numpy.ndarray): The input image from which to extract pixel values.
    Box (tuple): The coordinates defining the larger region of interest in the format (x, y, width, height).
    smallBox (tuple): The coordinates defining the smaller region of interest within the larger region in the format (x, y, width, height).
    Returns:
    list: A list containing the standard deviations and peak histogram values for the blue, green, and red channels 
          in the following order: [sigma_b_s, sigma_g_s, sigma_r_s, peak_b_s, peak_g_s, peak_r_s].
    """

    image = input_image.copy()
    
    pixels = getMaskPixels(image, Box, smallBox)
    
    # Calculate the standard deviation for each color channel
    sigma_b_s = np.std(pixels[:, 0])  # Blue channel
    sigma_g_s = np.std(pixels[:, 1])  # Green channel
    sigma_r_s = np.std(pixels[:, 2])  # Red channel

    # Calculate the peak histogram values for each color channel
    peak_b_s = np.argmax(np.histogram(pixels[:,0], bins=256, range=[0, 256])[0])
    peak_g_s = np.argmax(np.histogram(pixels[:,1], bins=256, range=[0, 256])[0])
    peak_r_s = np.argmax(np.histogram(pixels[:,2], bins=256, range=[0, 256])[0])

    return [sigma_b_s, sigma_g_s, sigma_r_s, peak_b_s, peak_g_s, peak_r_s]


def correctImage(image, ground_thruth, outer_correction_frame, inner_correction_frame):
    """
    Corrects the color of an image based on provided ground truth and correction frames.
    Parameters:
    image (numpy.ndarray): The input image to be corrected.
    ground_thruth (list or numpy.ndarray): The ground truth values for color correction.
    outer_correction_frame (numpy.ndarray): The outer frame used for calculating correction values.
    inner_correction_frame (numpy.ndarray): The inner frame used for calculating correction values.
    Returns:
    numpy.ndarray: The color-corrected image.
    """
    
    corrected_image = image.copy()
    
    # Calculate correction values
    correction_values = getImageValues(image, outer_correction_frame, inner_correction_frame)
    
    # Calculate correction constants
    c_0 = (correction_values[2]/ground_thruth[2] + correction_values[1]/ground_thruth[1] + correction_values[0]/ground_thruth[3])/3
    c_r = correction_values[5]/c_0 - ground_thruth[5]
    c_g = correction_values[4]/c_0 - ground_thruth[4]
    c_b = correction_values[3]/c_0 - ground_thruth[3]

    # Recalibrate image
    corrected_image[:, :, 0] = np.clip((image[:, :, 0]/c_0 - c_b), 0, 255)  # B
    corrected_image[:, :, 1] = np.clip((image[:, :, 1]/c_0 - c_g), 0, 255)  # G
    corrected_image[:, :, 2] = np.clip((image[:, :, 2]/c_0 - c_r), 0, 255)  # R

    return corrected_image, [c_0, c_r, c_g, c_b]

def showHistogram(original_image, corrected_image, outer_correction_frame, inner_correction_frame):
    """
    Displays histograms of the color channels for the original and corrected images.
    Parameters:
    original_image (numpy.ndarray): The original image.
    corrected_image (numpy.ndarray): The corrected image.
    outer_correction_frame (tuple): The outer frame for masking pixels (x, y, width, height).
    inner_correction_frame (tuple): The inner frame for masking pixels (x, y, width, height).
    Returns:
    None
    """

    original_pixels = getMaskPixels(original_image, outer_correction_frame, inner_correction_frame)
    corrected_pixels = getMaskPixels(corrected_image, outer_correction_frame, inner_correction_frame)
    
    fig, (ax1, ax2)= plt.subplots(1,2)
    fig.suptitle('Histogramm der Farbkanäle')
    ax1.hist(original_pixels[:,0], bins=256, range=[0, 256], color='b')
    ax1.hist(original_pixels[:,1], bins=256, range=[0, 256], color='g')
    ax1.hist(original_pixels[:,2], bins=256, range=[0, 256], color='r')
    ax1.set_title('Originalbild')

    ax2.hist(corrected_pixels[:,0], bins=256, range=[0, 256], color='b')
    ax2.hist(corrected_pixels[:,1], bins=256, range=[0, 256], color='g')
    ax2.hist(corrected_pixels[:,2], bins=256, range=[0, 256], color='r')
    ax2.set_title('Korrigiertes Bild')
    plt.show(block=False)
    plt.pause(0.5)



def color_filter(image):
    """
    Filters an image to detect specific colors and their regions.
    Args:
        image (numpy.ndarray): The input image in BGR format.
    Returns:
        list: A list of dictionaries, each containing information about detected color regions:
            - bbox (list): Bounding box coordinates [x, y, width, height].
            - area_focus_point (list): Center point of the bounding box [x_center, y_center].
            - color (str): Detected color name.
            - grid_position (None): Placeholder for grid position (currently None).
            - average_hue (float): Average hue value of the detected region.
    """

    result_picture = image.copy()
    height, width = image.shape[:2]
    results = []
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(result_picture, cv2.COLOR_BGR2HSV)

    # Define color ranges
    color_areas = {
       "Rot": [(np.array([0, 90, 70]), np.array([5, 255, 255])),
                (np.array([160, 90, 70]), np.array([180, 255, 255]))],
        "Blau": [(np.array([95, 130, 70]), np.array([150, 255, 255]))],
        "Gelb": [(np.array([20, 50, 100]), np.array([44, 255, 255]))],
        "Grun": [(np.array([45, 50, 70]), np.array([95, 255, 255]))],
        "Orange": [(np.array([5, 100, 70]), np.array([19, 255, 255]))],
        "Weiss": [(np.array([0, 0, 160]), np.array([180, 50, 255]))],
    }

    for color, tresholds in color_areas.items():

        # Create mask for the color
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        for lower_treshold, upper_treshold in tresholds:
            mask |= cv2.inRange(hsv_image, lower_treshold, upper_treshold)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))     

        # Find the contours of the objects
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (math.sqrt(w * h) > math.sqrt(width * height)/6) & (math.sqrt(w * h) < math.sqrt(width * height)/3) & (w / h > 0.8) & (w / h < 1.2) & (cv2.contourArea(contour)/(w * h) > 0.7):
                filtered_contours.append(contour)
        
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Extract area
            roi_mask = mask[y:y+h, x:x+w]
            roi_hsv = hsv_image[y:y+h, x:x+w]

            # Draw the contour in white on the black picture
            image_roi_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(image_roi_mask, [contour], -1, (255), thickness=cv2.FILLED)

            # Combine this color with the overall mask
            combined_mask |= image_roi_mask

            # Calculate average Hue value
            hue_values = roi_hsv[:, :, 0][roi_mask > 0]
            average_hue = hue_values.mean() if len(hue_values) > 0 else 0

            # Ergebnis speichern
            results.append({
                "bbox": [x, y, w, h],
                "area_focus_point": [x+w//2, y+h//2],
                "color": color,
                "grid_position": None,
                "average_hue": average_hue
            })
    return results, combined_mask


def color_detection(image, center_point):
    """
    Detects the color of a region in an image centered at a given point.
    Args:
        image (numpy.ndarray): The input image in BGR format.
        center_point (tuple): The (x, y) coordinates of the center point of the region of interest.
    Returns:
        str: The detected color as a string. Possible values are "Rot", "Blau", "Gelb", "Grun", "Orange", "Weiss", or "none" if no color is detected.
    """
    size_cube = 10  # Size of the region around the center point to consider for color detection

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define the region of interest (ROI) around the center point
    roi = [center_point[0] - size_cube / 2, center_point[1] - size_cube / 2, size_cube, size_cube]

    # Extract the HSV values of the ROI
    roi_hsv = hsv_image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    
    # Calculate the average hue, saturation, and value of the ROI
    average_hue = np.mean(roi_hsv[:, :, 0])
    average_saturation = np.mean(roi_hsv[:, :, 1])
    average_value = np.mean(roi_hsv[:, :, 2])

    # Define the color ranges in HSV space
    color_areas = {
        "Rot": [(np.array([0, 90, 70]), np.array([5, 255, 255])),
                (np.array([160, 90, 70]), np.array([180, 255, 255]))],
        "Blau": [(np.array([95, 130, 70]), np.array([150, 255, 255]))],
        "Gelb": [(np.array([20, 20, 100]), np.array([45, 255, 255]))],
        "Grun": [(np.array([45, 50, 70]), np.array([95, 255, 255]))],
        "Orange": [(np.array([5, 100, 70]), np.array([20, 255, 255]))],
        "Weiss": [(np.array([0, 0, 160]), np.array([180, 50, 255]))],
    }

    color = None  
    
    # Iterate through the defined color ranges to find a match
    for color_detect, tresholds in color_areas.items():
        for lower_treshold, upper_treshold in tresholds:
            # Check if the average HSV values fall within the current color range
            if (lower_treshold[0] <= average_hue <= upper_treshold[0] and
                lower_treshold[1] <= average_saturation <= upper_treshold[1] and
                lower_treshold[2] <= average_value <= upper_treshold[2]):
                color = color_detect  # Set the detected color
                break
        if color is not None:
            break

    return color  # Return the detected color

def update_color_for_none_entries(dict, image):
    """
    Aktualisiert den 'color'-Wert für alle Einträge in dict, bei denen 'color' None ist.
    
    Args:
        dict (dict): Das Dictionary mit den Einträgen.
        image (numpy.ndarray): Das Bild, das für die colorrkennung verwendet wird.
    
    Returns:
        dict: Aktualisiertes Dictionary.
    """
    if dict is None:
        return None
    
    for _, entry in dict.items():
        if entry['color'] is None:
            # Apply color detection and update 'color'
            center_point = entry['area_focus_point']
            detected_color = color_detection(image, center_point)
            entry['color'] = detected_color

    return dict