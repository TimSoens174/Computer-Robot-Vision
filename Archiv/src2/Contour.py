import cv2
import numpy as np

def cut_image(image, x ,y ,w ,h):
    img = image.copy()
    return img[y:y+h, x:x+w]

def detect_outer_gray_frame(image):
    bild = image.copy()
    x,y,w,h = 0,0,0,0
    gray = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)

    # Bild glätten
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Kanten erkennen (Canny) mit angepassten Schwellenwerten
    edges = cv2.Canny(blurred, 10, 20, L2gradient=True)

    # Lücken in Kanten schließen
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Konturen finden
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        bild = cv2.rectangle(bild, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return bild, [x,y,w,h]

def small_greyFrame(frameImage, gb, Faktor):
    center_x, center_y = gb[0] + gb[2] // 2, gb[1] + gb[3] // 2  # Mittelpunkt der Box
    small_w, small_h = int(gb[2] * Faktor), int(gb[3] * Faktor)    # 15% kleinere Breite und Höhe
    small_x, small_y = center_x - small_w // 2, center_y - small_h // 2  # Obere linke Ecke

    frameImage = cv2.rectangle(frameImage, (small_x, small_y), (small_x + small_w, small_y + small_h), (0, 0, 255), 2)

    return frameImage, [small_x, small_y, small_w, small_h]

