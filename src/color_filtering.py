import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def getMaskPixels(image, outer_correction_frame, inner_correction_frame):
    
    height, width = image.shape[:2]

    # Maske für die große Box
    mask_large = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask_large, (outer_correction_frame[0], outer_correction_frame[1]), (outer_correction_frame[0] + outer_correction_frame[2], outer_correction_frame[1] + outer_correction_frame[3]), 255, -1)

    # Maske für die kleine Box
    mask_small = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask_small, (inner_correction_frame[0], inner_correction_frame[1]), (inner_correction_frame[0] + inner_correction_frame[2], inner_correction_frame[1] + inner_correction_frame[3]), 255, -1)

    # Bereich zwischen den Boxen: große Maske minus kleine Maske
    mask_between = cv2.subtract(mask_large, mask_small)

    # Pixel im Bereich extrahieren
    all_pixels = image[mask_between == 255]
    pixels = all_pixels[~np.all(all_pixels[:,:] == 0, axis=1)]

    return pixels


def getCorrectionValues(input_image, Box, smallBox):
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
    
    sigma_b_s = np.std(pixels[:, 0])  # Blau-Kanal
    sigma_g_s = np.std(pixels[:, 1])  # Grün-Kanal
    sigma_r_s = np.std(pixels[:, 2])  # Rot-Kanal

    peak_b_s = np.argmax(np.histogram(pixels[:,0],bins=256, range=[0, 256])[0])
    peak_g_s = np.argmax(np.histogram(pixels[:,1],bins=256, range=[0, 256])[0])
    peak_r_s = np.argmax(np.histogram(pixels[:,2],bins=256, range=[0, 256])[0])

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
    
    # Korrekturwerte berechnen
    correction_values = getCorrectionValues(image, outer_correction_frame, inner_correction_frame)

    corrected_image = image.copy()
    
    c_0 = (correction_values[2]/ground_thruth[2] + correction_values[1]/ground_thruth[1] + correction_values[0]/ground_thruth[3])/3
    c_r = correction_values[5]/c_0 - ground_thruth[5]
    c_g = correction_values[4]/c_0 - ground_thruth[4]
    c_b = correction_values[3]/c_0 - ground_thruth[3]

    corrected_image[:, :, 0] = np.clip((image[:, :, 0]/c_0 - c_b), 0, 255)  # B
    corrected_image[:, :, 1] = np.clip((image[:, :, 1]/c_0 - c_g), 0, 255)  # G
    corrected_image[:, :, 2] = np.clip((image[:, :, 2]/c_0 - c_r), 0, 255)  # R

    
    print(f"Korrekturfaktoren:")
    print(f"C_B: {c_b:.2f}, C_G: {c_g:.2f}, C_R: {c_r:.2f}, C_0: {c_0:.2f}")
    return corrected_image

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

    bild = image.copy()
    height, width = image.shape[:2]
    # Bild in den HSV-Farbraum umwandeln
    hsv_bild = cv2.cvtColor(bild, cv2.COLOR_BGR2HSV)

    # Farbbereiche definieren
    farb_bereiche = {
       "Rot": [(np.array([0, 90, 70]), np.array([5, 255, 255])),
                (np.array([160, 90, 70]), np.array([180, 255, 255]))],
        "Blau": [(np.array([95, 130, 70]), np.array([150, 255, 255]))],
        "Gelb": [(np.array([20, 20, 100]), np.array([45, 255, 255]))],
        "Grun": [(np.array([45, 50, 70]), np.array([95, 255, 255]))],
        "Orange": [(np.array([5, 100, 70]), np.array([20, 255, 255]))],
        "Weiss": [(np.array([0, 0, 160]), np.array([180, 50, 255]))],
    }

    # Ergebnisbild kopieren
    ergebnisbild = bild.copy()

    # Ergebnisse speichern
    ergebnisse = []

    # Leere kombinierte Maske erstellen
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Über alle Farben iterieren
    i = 0
    for farbe, grenzen in farb_bereiche.items():
        i =i+1
        # Maske für die Farbe erstellen
        maske = np.zeros(hsv_bild.shape[:2], dtype=np.uint8)
        for untere_grenze, obere_grenze in grenzen:
            maske |= cv2.inRange(hsv_bild, untere_grenze, obere_grenze)

        maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

        #if farbe == "Gelb":
        #    cv2.imshow(f"test {i}", maske)
        #   cv2.waitKey(0)
        #    cv2.destroyAllWindows()

        

        # Konturen der Objekte finden
        konturen, hierarchie = cv2.findContours(maske, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_konturen = []
        for kontur in konturen:
            x, y, w, h = cv2.boundingRect(kontur)
            if (math.sqrt(w * h) > math.sqrt(width * height)/6) & (math.sqrt(w * h) < math.sqrt(width * height)/3) & (w / h > 0.8) & (w / h < 1.2) & (cv2.contourArea(kontur)/(w * h) > 0.7):
                filtered_konturen.append(kontur)
        
        for kontur in filtered_konturen:
            x, y, w, h = cv2.boundingRect(kontur)
            # Bereich extrahieren
            roi_maske = maske[y:y+h, x:x+w]
            roi_hsv = hsv_bild[y:y+h, x:x+w]


            # Zeichne die Kontur in weiß auf das schwarze Bild
            image_roi_maske = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(image_roi_maske, [kontur], -1, (255), thickness=cv2.FILLED)
            # Kombiniere diese Farbe mit der Gesamtmaske
            combined_mask |= image_roi_maske

            # Durchschnittlichen Hue-Wert berechnen
            hue_werte = roi_hsv[:, :, 0][roi_maske > 0]
            durchschnittlicher_hue = hue_werte.mean() if len(hue_werte) > 0 else 0

            # Ergebnis speichern
            ergebnisse.append({
                "bbox": [x, y, w, h],
                "area_focus_point": [x+w//2, y+h//2],
                "color": farbe,
                "grid_position": None,
                "average_hue": durchschnittlicher_hue
            })

            # # Bounding-Box zeichnen und Farbe beschriften
            # farben_rgb = {
            #     "Rot": (0, 0, 255),
            #     "Blau": (255, 0, 0),
            #     "Gelb": (0, 255, 255),
            #     "Grun": (0, 255, 0),
            #     "Orange": (0, 165, 255),
            #     "Weiss": (255, 255, 255),
            # }
            # cv2.rectangle(ergebnisbild, (x, y), (x + w, y + h), farben_rgb[farbe], 2)
            # cv2.putText(ergebnisbild, f"{farbe}: {int(durchschnittlicher_hue)}", 
            #             (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, farben_rgb[farbe], 2)

    # Ergebnisse ausgeben
    #print("Gefundene Objekte ohne Kinder:")
    #for i, obj in enumerate(ergebnisse, 1):
    #    print(f"Objekt {i}: Farbe={obj['Farbe']}, X={obj['x']}, Y={obj['y']}, "
    #          f"Breite={obj['width']}, Höhe={obj['height']}, Durchschnittlicher Hue={obj['average_hue']:.2f}")
    return ergebnisse, combined_mask


def color_detection(image, center_point):
    """
    Detects the color of a region in an image centered at a given point.
    Args:
        image (numpy.ndarray): The input image in BGR format.
        center_point (tuple): The (x, y) coordinates of the center point of the region of interest.
    Returns:
        str: The detected color as a string. Possible values are "Rot", "Blau", "Gelb", "Grun", "Orange", "Weiss", or "none" if no color is detected.
    """
    size_cube = 10  # Pixel

    hsv_bild = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    roi = [center_point[0]-size_cube/2, center_point[1]-size_cube/2, size_cube, size_cube]

    roi_hsv = hsv_bild[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    average_hue = np.mean(roi_hsv[:, :, 0])
    average_saturation = np.mean(roi_hsv[:, :, 1])
    average_value = np.mean(roi_hsv[:, :, 2])

    farb_bereiche = {
        "Rot": [(np.array([0, 90, 70]), np.array([5, 255, 255])),
                (np.array([160, 90, 70]), np.array([180, 255, 255]))],
        "Blau": [(np.array([95, 130, 70]), np.array([150, 255, 255]))],
        "Gelb": [(np.array([20, 20, 100]), np.array([45, 255, 255]))],
        "Grun": [(np.array([45, 50, 70]), np.array([95, 255, 255]))],
        "Orange": [(np.array([5, 100, 70]), np.array([20, 255, 255]))],
        "Weiss": [(np.array([0, 0, 160]), np.array([180, 50, 255]))],
    }

    color = None
    for farbe, grenzen in farb_bereiche.items():
        for untere_grenze, obere_grenze in grenzen:
            if (untere_grenze[0] <= average_hue <= obere_grenze[0] and
                untere_grenze[1] <= average_saturation <= obere_grenze[1] and
                untere_grenze[2] <= average_value <= obere_grenze[2]):
                color = farbe
                break
        if color != None:
            break

    return color

def update_color_for_none_entries(dict, image):
    """
    Aktualisiert den 'color'-Wert für alle Einträge in dict, bei denen 'color' None ist.
    
    Args:
        dict (dict): Das Dictionary mit den Einträgen.
        image (numpy.ndarray): Das Bild, das für die Farberkennung verwendet wird.
    
    Returns:
        dict: Aktualisiertes Dictionary.
    """
    if dict is None:
        return None
    
    for pos, entry in dict.items():
        if entry['color'] is None:
            # Wende color_detection an und aktualisiere 'color'
            center_point = entry['area_focus_point']
            detected_color = color_detection(image, center_point)
            entry['color'] = detected_color

    return dict