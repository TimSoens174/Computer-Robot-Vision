import cv2
import numpy as np

# Funktion für das Zeichnen und Erfassen eines Rechtecks
roi = None
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

# Bild laden
image_path = "Pictures/Picture 3.jpg"  # Pfad zum Bild
image = cv2.imread(image_path)
if image is None:
    print("Fehler: Bild konnte nicht geladen werden. Prüfe den Pfad.")
    exit()

cv2.namedWindow("Bild")
cv2.setMouseCallback("Bild", select_roi)

drawing = False
ix, iy = -1, -1

print("Wähle einen Bereich im Bild aus, indem du mit der linken Maustaste klickst und ziehst.")

while True:
    temp_image = image.copy()

    if roi:
        cv2.rectangle(temp_image, roi[0], roi[1], (0, 255, 0), 2)

    cv2.imshow("Bild", temp_image)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC zum Beenden
        print("Auswahl abgebrochen.")
        break
    elif key == ord('q'):  # 'q' zur Berechnung
        if roi:
            break

if roi:
    x1, y1 = roi[0]
    x2, y2 = roi[1]

    # Sicherstellen, dass die Koordinaten korrekt sind
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    selected_region = image[y1:y2, x1:x2]

    if selected_region.size > 0:
        # Sigma (Standardabweichung) für jeden Farbkanal berechnen
        sigma_b = np.std(selected_region[:, :, 0])  # Blau-Kanal
        sigma_g = np.std(selected_region[:, :, 1])  # Grün-Kanal
        sigma_r = np.std(selected_region[:, :, 2])  # Rot-Kanal

        hist_b = cv2.calcHist([selected_region], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([selected_region], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([selected_region], [2], None, [256], [0, 256])

        peak_b = np.argmax(hist_b)  # Peak des Blau-Kanals
        peak_g = np.argmax(hist_g)  # Peak des Grün-Kanals
        peak_r = np.argmax(hist_r)  # Peak des Rot-Kanals

        print(f"Standardabweichung im ausgewählten Bereich:")
        print(f"Blau-Kanal: {sigma_b:.2f}")
        print(f"Grün-Kanal: {sigma_g:.2f}")
        print(f"Rot-Kanal: {sigma_r:.2f}")

        print(f"Blau-Peak: {peak_b:.2f}")
        print(f"Grün-Peak: {peak_g:.2f}")
        print(f"Rot-Peak: {peak_r:.2f}")
    else:
        print("Fehler: Der ausgewählte Bereich ist leer.")

cv2.destroyAllWindows()
