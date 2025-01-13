import cv2
import numpy as np

# Trackbar callback function
def nothing(x):
    pass

def sort_rois_into_grid(rois, image_width, image_height):
    # Get grid cell height and width
    cell_height = image_height // 3
    cell_width = image_width // 3

    # Sort ROIs by their top-left corner position (first by y, then by x)
    print("before sort",rois)
    
    # We want to make sure we have exactly 9 ROIs for the 3x3 grid
    # Now, we map these sorted ROIs to the expected grid positions
    sorted_rois = []
    for i in range(len(rois)):
        # Expected positions in a 3x3 grid
        row = i // 3
        col = i % 3
        ex = col * cell_width
        ey = row * cell_height

        # Find the closest ROI that fits into the expected grid cell
        closest_roi = None
        for roi in rois:
            cx, cy, cw, ch = roi
            if (ex <= cx < (ex + cell_width) and
                ey <= cy < (ey + cell_height)):
                closest_roi = roi
                break
        
        if closest_roi is None:
            # If no closest ROI found, insert a placeholder (0, 0, 0, 0)
            sorted_rois.append((0, 0, 0, 0))
        else:
            sorted_rois.append(closest_roi)
    print("sorted rois", sorted_rois)
    return sorted_rois



def get_rois(image):
    # Step 1: Cropping the image to the center square (fixed size)
    height, width = image.shape[:2]
    side_length = min(width, height) // 3
    start_x = (width - side_length) // 2
    start_y = (height - side_length) // 2
    end_x = start_x + side_length
    end_y = start_y + side_length
    cropped_image = image[start_y:end_y, start_x:end_x]

    # Step 2: Set up trackbars for black color filtering
    cv2.namedWindow("Black Color Filtering")
    cv2.createTrackbar("Lower Black Threshold", "Black Color Filtering", 0, 255, nothing)
    cv2.createTrackbar("Upper Black Threshold", "Black Color Filtering", 20, 255, nothing)

    # Loop to adjust black filtering thresholds
    while True:
        # Get current trackbar values
        lower_black_val = cv2.getTrackbarPos("Lower Black Threshold", "Black Color Filtering")
        upper_black_val = cv2.getTrackbarPos("Upper Black Threshold", "Black Color Filtering")

        # Convert to HSV color space to detect black
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        
        # Define lower and upper bounds for black color in HSV
        lower_black = np.array([0, 0, lower_black_val])
        upper_black = np.array([180, 255, upper_black_val])

        # Create a mask for black color
        black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
        
        # Show the masked image to visualize the result
        cv2.imshow("Masked Image", black_mask)

        # Break the loop when 'Esc' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 'Esc' key
            break

    # Step 3: Detect contours on the binary image to find the largest contour
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image or a blank image for visualization
    visualization_image = cv2.cvtColor(black_mask, cv2.COLOR_GRAY2BGR)  # Convert black_mask to a 3-channel image for visualization

    # Ensure contours are found
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        #x, y, w, h = x-5, y-5, w+5, h+5
        # Draw all contours in green (this will show everything)
        for contour in contours:
            cv2.drawContours(visualization_image, [contour], -1, (0, 255, 0), 2)  # Green contours
        
        # Draw the bounding box on the image for the largest contour
        cv2.rectangle(visualization_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red bounding box

    # Display the image with the contour and bounding box
    cv2.imshow("BoundingBox", visualization_image)  # Show the color image with contours and bounding box
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Step 4: Crop the image to the bounding box of the largest contour
    cropped_image = black_mask[y:y+h, x:x+w]

    # Step 6: Detect smaller contours inside the cropped area
    contours, _ = cv2.findContours(cropped_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Set up trackbars to filter small contours based on area
    cv2.namedWindow("Small Contour Filter")
    cv2.createTrackbar("Min Area", "Small Contour Filter", 1500, 3500, nothing)
    cv2.createTrackbar("Max Area", "Small Contour Filter", 2500, 5000, nothing)

    # Loop for adjusting contour area thresholds
    while True:
        # Get current values for area thresholds
        min_area = cv2.getTrackbarPos("Min Area", "Small Contour Filter")
        max_area = cv2.getTrackbarPos("Max Area", "Small Contour Filter")

        # Initialize an empty list to store ROIs and a copy for drawing
        rois = []
        roi_debug_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

        # Loop through contours and filter based on area
        for contour in contours:
            area = cv2.contourArea(contour)
            # Optionally, draw all contours on the image (can be used instead of bounding boxes)
            cv2.drawContours(roi_debug_image, [contour], -1, (0, 255, 0), 2)  # Green contours
            if min_area < area < max_area:
                # Get bounding box for each smaller contour
                cx, cy, cw, ch = cv2.boundingRect(contour)
                rois.append((x + cx, y + cy, cw, ch))  # Adjust ROI position relative to the original image

                # Draw bounding box for debugging
                cv2.rectangle(roi_debug_image, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)  # Blue for inner squares

        # Display the detected ROIs within the cropped area
        cv2.imshow("Filtered Smaller Contours", roi_debug_image)

        # Break the loop when 'Esc' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # Close all windows
    cv2.destroyAllWindows()

    # Step 7: Get the sorted ROIs based on the 3x3 grid and position them
    sorted_rois = sort_rois_into_grid(rois, cropped_image.shape[1], cropped_image.shape[0])
    final_rois = []
    for roi in sorted_rois:
        # For each ROI, we adjust its position to the full image's coordinates
        cx, cy, cw, ch = roi
        adjusted_roi = (cx + start_x, cy + start_y, cw, ch)  # Adjust position
        final_rois.append(adjusted_roi)

    for roi in final_rois:
        cx, cy, cw, ch = roi
        cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)  # Green bounding box

    # Display the original image with the sorted rois
    # cv2.imshow("FinalSegmentation", image)  # Show the color image with contours and bounding box
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("---------------------------Segmentation done---------------------------")

    return final_rois

# Example usage (uncomment and replace with an actual image path to test)
# image = cv2.imread("path_to_your_image.jpg")
# rois = get_rois(image)
# print("ROIs:", rois)
