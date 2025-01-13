import cv2
import numpy as np
import os

def detect_outer_gray_frame(image_path, i):
    # Bild laden
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bild glätten
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Kanten erkennen (Canny) mit angepassten Schwellenwerten
    edges = cv2.Canny(blurred, 10, 20, L2gradient=True)

    # Lücken in Kanten schließen
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Konturen finden
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #img = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Ergebnis anzeigen
   # cv2.imshow(str(i+20), img)

    # Größte rechteckige Kontur finden
    largest_contour = None
    max_area = 0

    for contour in contours:
        # Berechne Approximation der Kontur
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # Rechtecke haben 4 Punkte
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:  # Nur die größte Kontur speichern
                max_area = area
                largest_contour = approx

    if largest_contour is not None:
        # Berechne die Bounding Box der Kontur
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Zeichne die ursprüngliche Bounding Box (grün)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Berechnung der neuen, verkleinerten Bounding Box (10% kleiner)
        center_x, center_y = x + w // 2, y + h // 2  # Mittelpunkt der Box
        new_w, new_h = int(w * 0.85), int(h * 0.85)    # 10% kleinere Breite und Höhe
        new_x, new_y = center_x - new_w // 2, center_y - new_h // 2  # Obere linke Ecke

        # Zeichne die kleinere Bounding Box (rot)
        cv2.rectangle(image, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 2)

    # Ergebnis anzeigen
    cv2.imshow(str(i), image)

    # Ergebnis anzeigen
    #cv2.imshow(str(i), image)


ordner_pfad = 'Pictures2'
i=0
# Beispielaufruf mit Bild
for filename in os.listdir(ordner_pfad):
        # Prüfe, ob die Datei die Endung .jpg hat
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(ordner_pfad, filename)
            print(f"Processing: {file_path}")
            
            # Aufruf der Funktion mit dem Pfad zur aktuellen Datei
            detect_outer_gray_frame(file_path, i)
            i=i+1
cv2.waitKey(0)
cv2.destroyAllWindows()

