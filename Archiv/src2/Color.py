import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    image = input_image.copy()
    
    pixels = getMaskPixels(image, Box, smallBox)
    
    sigma_b_s = np.std(pixels[:, 0])  # Blau-Kanal
    sigma_g_s = np.std(pixels[:, 1])  # Grün-Kanal
    sigma_r_s = np.std(pixels[:, 2])  # Rot-Kanal

    peak_b_s = np.argmax(np.histogram(pixels[:,0],bins=256, range=[0, 256])[0])
    peak_g_s = np.argmax(np.histogram(pixels[:,1],bins=256, range=[0, 256])[0])
    peak_r_s = np.argmax(np.histogram(pixels[:,2],bins=256, range=[0, 256])[0])

    return [sigma_b_s, sigma_g_s, sigma_r_s, peak_b_s, peak_g_s, peak_r_s]


def correctImage(image, ground_thruth, image_values):
    corrected_image = image.copy()
    
    c_0 = (image_values[2]/ground_thruth[2] + image_values[1]/ground_thruth[1] + image_values[0]/ground_thruth[3])/3
    c_r = image_values[5]/c_0 - ground_thruth[5]
    c_g = image_values[4]/c_0 - ground_thruth[4]
    c_b = image_values[3]/c_0 - ground_thruth[3]

    corrected_image[:, :, 0] = np.clip((image[:, :, 0]/c_0 - c_b), 0, 255)  # B
    corrected_image[:, :, 1] = np.clip((image[:, :, 1]/c_0 - c_g), 0, 255)  # G
    corrected_image[:, :, 2] = np.clip((image[:, :, 2]/c_0 - c_r), 0, 255)  # R

    
    #print(f"Korrekturfaktoren:")
    #print(f"C_B: {c_b:.2f}, C_G: {c_g:.2f}, C_R: {c_r:.2f}, C_0: {c_0:.2f}")
    return corrected_image

def showHistogram(original_image, corrected_image, outer_correction_frame, inner_correction_frame):

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



def filter(image):
    bild = image.copy()
    # Bild in den HSV-Farbraum umwandeln
    hsv_bild = cv2.cvtColor(bild, cv2.COLOR_BGR2HSV)

    # Farbbereiche definieren
    farb_bereiche = {
        "Rot": [(np.array([0, 90, 70]), np.array([5, 255, 255])),
                (np.array([170, 90, 70]), np.array([180, 255, 255]))],
        "Blau": [(np.array([95, 130, 70]), np.array([140, 255, 255]))],
        "Gelb": [(np.array([20, 20, 100]), np.array([38, 255, 255]))],
        "Grun": [(np.array([40, 50, 70]), np.array([90, 255, 255]))],
        "Orange": [(np.array([5, 130, 70]), np.array([20, 255, 255]))],
        "Weiss": [(np.array([0, 0, 160]), np.array([180, 20, 255]))],
    }

    # Ergebnisbild kopieren
    ergebnisbild = bild.copy()

    # Ergebnisse speichern
    ergebnisse = []

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

        if farbe == "Blau":
            cv2.imshow(f"test {i}", maske)

        # Konturen der Objekte finden
        konturen, hierarchie = cv2.findContours(maske, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_konturen = []
        for kontur in konturen:
            x, y, w, h = cv2.boundingRect(kontur)
            if (w * h > 100) & (w * h < 10000) & (w / h > 0.7) & (w / h < 1.3) & ( cv2.contourArea(kontur)/(w * h) > 0.5):
                filtered_konturen.append(kontur)
        
        for kontur in filtered_konturen:
            x, y, w, h = cv2.boundingRect(kontur)
            # Bereich extrahieren
            roi_maske = maske[y:y+h, x:x+w]
            roi_hsv = hsv_bild[y:y+h, x:x+w]

            # Durchschnittlichen Hue-Wert berechnen
            hue_werte = roi_hsv[:, :, 0][roi_maske > 0]
            durchschnittlicher_hue = hue_werte.mean() if len(hue_werte) > 0 else 0

            # Ergebnis speichern
            ergebnisse.append({
                "Farbe": farbe,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "average_hue": durchschnittlicher_hue
            })

            # Bounding-Box zeichnen und Farbe beschriften
            farben_rgb = {
                "Rot": (0, 0, 255),
                "Blau": (255, 0, 0),
                "Gelb": (0, 255, 255),
                "Grun": (0, 255, 0),
                "Orange": (0, 165, 255),
                "Weiss": (255, 255, 255),
            }
            cv2.rectangle(ergebnisbild, (x, y), (x + w, y + h), farben_rgb[farbe], 2)
            cv2.putText(ergebnisbild, f"{farbe}: {int(durchschnittlicher_hue)}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, farben_rgb[farbe], 2)

    # Ergebnisse ausgeben
    #print("Gefundene Objekte ohne Kinder:")
    #for i, obj in enumerate(ergebnisse, 1):
    #    print(f"Objekt {i}: Farbe={obj['Farbe']}, X={obj['x']}, Y={obj['y']}, "
    #          f"Breite={obj['width']}, Höhe={obj['height']}, Durchschnittlicher Hue={obj['average_hue']:.2f}")
    return ergebnisbild, ergebnisse
