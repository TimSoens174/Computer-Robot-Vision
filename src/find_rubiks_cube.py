import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """
    Preprocess the image: convert to grayscale and filter for edges.
    """
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding for binary image
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    return image, binary


def find_grid_structure(binary_image):
    """
    Find the grid-like structure using contours and filtering by aspect ratio and area.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour is rectangular (4 points) and area is significant
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Filtering based on aspect ratio and minimum size
            if 0.8 <= aspect_ratio <= 1.2:
                candidates.append((x, y, w, h))

    return candidates


def draw_bounding_boxes(image, candidates):
    """
    Draw bounding boxes around detected grid structures.
    """
    for (x, y, w, h) in candidates:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


def detect_rubiks_cube_with_orb(image_path, template_path):
    """
    Full pipeline for detecting the Rubik's Cube grid structure using ORB for feature matching.
    """
    print("Step 1: Preprocessing the input image...")
    original_image, binary_image = preprocess_image(image_path)
    cv2.imshow("Original Image", original_image)
    cv2.imshow("Binary Image", binary_image)

    # Load and preprocess the template image
    print("Step 2: Preprocessing the template image...")
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError("Template image not found at the specified path.")
    cv2.imshow("Template Image", template)

    # Step 3: Initialize ORB detector and find keypoints and descriptors
    print("Step 3: Detecting features using ORB...")
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(binary_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template, None)

    # Draw keypoints for debugging
    debug_image = cv2.drawKeypoints(original_image, keypoints1, None, color=(0, 255, 0), flags=0)
    cv2.imshow("ORB Keypoints", debug_image)

    # Step 4: Match descriptors using BFMatcher
    print("Step 4: Matching features...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches for visualization
    match_visualization = cv2.drawMatches(original_image, keypoints1, template, keypoints2, matches[:20], None, flags=2)
    cv2.imshow("Feature Matches", match_visualization)

    # Step 5: Locate the grid structure in the original image
    print("Step 5: Locating the grid structure...")
    if len(matches) > 4:  # Ensure enough matches for homography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if matrix is not None:
            # Use the homography matrix to project the template's bounding box onto the input image
            h, w = template.shape
            points = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(points, matrix)

            # Draw the bounding box
            result_image = original_image.copy()
            cv2.polylines(result_image, [np.int32(transformed_points)], True, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("Detected Grid", result_image)
        else:
            print("Homography computation failed. No grid detected.")
            result_image = original_image
    else:
        print("Not enough matches to locate the grid.")
        result_image = original_image

    return result_image




if __name__ == "__main__":
    image_path = os.path.join("..", "Logitech Webcam", "Picture 1.jpg")
    template_path = os.path.join("rubiks_cube_inverted_template.jpg")
    # Run the detection pipeline
    result = detect_rubiks_cube_with_orb(image_path, template_path)

    # Display the result
    cv2.imshow("Detected Rubik's Cube", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the result
    #cv2.imwrite("result.jpg", result)
