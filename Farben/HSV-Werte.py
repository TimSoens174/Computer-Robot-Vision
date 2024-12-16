import cv2
import numpy as np
import os

# Callback-Funktion für Mausereignisse
def get_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Wenn die linke Maustaste gedrückt wird
        # Hole den HSV-Wert des angeklickten Pixels
        hsv_value = hsv_image[y, x]
        print(f"HSV-Werte: H={hsv_value[0]}, S={hsv_value[1]}, V={hsv_value[2]}")

# Lade ein Bild
ordner_pfad = 'Pictures'
bild_datei = 'Picture 7.jpg'  # Name der Bilddatei im Ordner
bild_pfad = os.path.join(ordner_pfad, bild_datei)
bild = cv2.imread(bild_pfad)

# Konvertiere das Bild von BGR nach HSV
hsv_image = cv2.cvtColor(bild, cv2.COLOR_BGR2HSV)

# Zeige das Bild an
cv2.imshow('Image', bild)

# Setze die Mausereignisfunktion
cv2.setMouseCallback('Image', get_hsv_value)

# Warte auf eine Tasteneingabe, um das Programm zu beenden
cv2.waitKey(0)
cv2.destroyAllWindows()
