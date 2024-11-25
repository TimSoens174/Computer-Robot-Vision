import cv2
import numpy as np



# Maus-Callback-Funktion
def select_roi(event, x, y, flags, param):
    global roi, ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi = [(ix, iy), (x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = [(ix, iy), (x, y)]

def color_correction(picture):

    # Bild laden
    image = cv2.imread(picture)
    if image is None:
        print("Fehler: Bild konnte nicht geladen werden. Prüfe den Pfad.")
        exit()

    cv2.namedWindow("Bild")
    cv2.setMouseCallback("Bild", select_roi)

    print("Wähle einen Bereich im Bild aus, indem du mit der linken Maustaste klickst und ziehst.")

    # ROI-Auswahl
    while True:
        temp_image = image.copy()

        if roi:
            cv2.rectangle(temp_image, roi[0], roi[1], (0, 255, 0), 2)

        cv2.imshow("Bild", temp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC-Taste zum Abbrechen
            print("Auswahl abgebrochen.")
            cv2.destroyAllWindows()
            exit()
        elif key == ord('q'):  # 'q'-Taste zur Berechnung
            if roi:
                break

    # ROI verarbeiten
    if roi:
        x1, y1 = roi[0]
        x2, y2 = roi[1]

        # Koordinaten sicherstellen
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        selected_region = image[y1:y2, x1:x2]

        if selected_region.size > 0:
            # Sigma-Werte berechnen
            sigma_b_s = np.std(selected_region[:, :, 0])  # Blau-Kanal
            sigma_g_s = np.std(selected_region[:, :, 1])  # Grün-Kanal
            sigma_r_s = np.std(selected_region[:, :, 2])  # Rot-Kanal

            peak_b_s = np.argmax(cv2.calcHist([selected_region], [0], None, [256], [0, 256]))
            peak_g_s = np.argmax(cv2.calcHist([selected_region], [1], None, [256], [0, 256]))
            peak_r_s = np.argmax(cv2.calcHist([selected_region], [2], None, [256], [0, 256]))


            print(f"Standardabweichungen:")
    #        print(f"Blau-Kanal: {sigma_b:.2f}, Grün-Kanal: {sigma_g:.2f}, Rot-Kanal: {sigma_r:.2f}")

            # Korrekturfaktoren berechnen
            c_0 = (sigma_r_s/sig_r + sigma_g_s/sig_g + sigma_b_s/sig_b)/3
            c_r = peak_r_s/c_0 - peak_r
            c_g = peak_g_s/c_0 - peak_g
            c_b = peak_b_s/c_0 - peak_b

            print(f"Korrekturfaktoren:")
            print(f"C_B: {c_b:.2f}, C_G: {c_g:.2f}, C_R: {c_r:.2f}, C_0: {c_0:.2f}")

            # Farbkorrektur anwenden
            corrected_image = image.copy()
            corrected_image[:, :, 0] = np.clip(c_0*(image[:, :, 0] - c_b), 0, 255)  # Blau-Kanal
            corrected_image[:, :, 1] = np.clip(c_0*(image[:, :, 1] - c_g), 0, 255)  # Grün-Kanal
            corrected_image[:, :, 2] = np.clip(c_0*(image[:, :, 2] - c_r), 0, 255)  # Rot-Kanal

            # Ergebnisse anzeigen
            #cv2.imshow("Originalbild", image)
            cv2.imshow("Korrigiertes Bild", corrected_image)

            # Korrigiertes Bild speichern
            #corrected_path = "pictures/corrected_image.jpg"  # Zielpfad
            #cv2.imwrite(corrected_path, corrected_image)
            #print(f"Korrigiertes Bild gespeichert: {corrected_path}")

            cv2.waitKey(0)
        else:
            print("Fehler: Der ausgewählte Bereich ist leer.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    sig_r = 12.08
    sig_g = 12.37
    sig_b = 11.91
    peak_r = 138
    peak_g = 148
    peak_b = 150

    # Globale Variablen für ROI
    roi = None
    drawing = False
    ix, iy = -1, -1


    image_path = "Pictures2/Picture 10.jpg"  # Ersetze mit dem tatsächlichen Bildpfad
    color_correction(image_path)