
import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Threshold for Black Regions
    # Black pixels will have low intensity in grayscale
    _, black_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

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

    # # First, apply MORPH_CLOSE to fill gaps and remove small noise
    clean_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # Increase iterations for stronger effect

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Increase kernel size for more intense effect
    # Apply MORPH_OPEN (erosion followed by dilation) to remove smaller noise and smooth out the edges
    #clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    #clean_mask = black_mask
    # # Alternatively, apply a morphological gradient to sharpen edges
    gradient_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_GRADIENT, kernel)

    # Combine the cleaned mask with the gradient mask to highlight edges
    clean_mask = cv2.bitwise_or(clean_mask, gradient_mask)

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

    # Listen initialisieren
    valid_rectangles = []
    list_dict_edge = []

    for contour in contours:
        # Annäherung der Kontur durch ein Polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Fläche berechnen, um kleine oder große Regionen zu filtern
        area = cv2.contourArea(approx)
        if 500 < area < 12000:  # Grenzwerte anpassen basierend auf der Bildgröße
            # Schwerpunkte berechnen
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                area_focus_point = (cx, cy)
            else:
                area_focus_point = (None, None)

            # Kontur speichern
            valid_rectangles.append(approx)

            # Eintrag im Attribut-Wörterbuch erstellen
            valid_element = {
                "bbox": cv2.boundingRect(approx),
                "area_focus_point": area_focus_point,
                "color": None,
                "grid_position": None,
                "avg_hue": None,
            }
            list_dict_edge.append(valid_element)


        # Step 5: Draw the Detected Rectangles on the Original Image
        # output_image = image.copy()
        # for rect in valid_rectangles:
        #     cv2.drawContours(output_image, [rect], -1, (0, 255, 0), 2)

        # Display the Result
        # plt.figure(figsize=(6, 6))
        # plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        # plt.title("Detected Rectangles (Colored Squares)")
        # plt.axis("off")
        # plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    return list_dict_edge, clean_mask