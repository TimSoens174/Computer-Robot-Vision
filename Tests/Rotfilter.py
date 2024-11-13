import cv2
import numpy as np
import os

# Pfad zum Ordner, in dem sich das Bild befindet
ordner_pfad = 'Logitech Webcam'
bild_datei = 'Picture 1.jpg'  # Name der Bilddatei im Ordner

# Bild aus dem Ordner laden
bild_pfad = os.path.join(ordner_pfad, bild_datei)
bild = cv2.imread(bild_pfad)

# Überprüfen, ob das Bild geladen wurde
if bild is None:
    print("Das Bild konnte nicht geladen werden. Überprüfen Sie den Pfad und den Dateinamen.")
else:
    # Bild in den HSV-Farbraum umwandeln
    hsv_bild = cv2.cvtColor(bild, cv2.COLOR_BGR2HSV)

    # Grenzen für die rote Farbe definieren (zwei Bereiche für helles und dunkles Rot)
    untere_rot_1 = np.array([0, 120, 70])
    obere_rot_1 = np.array([10, 255, 255])
    untere_rot_2 = np.array([170, 120, 70])
    obere_rot_2 = np.array([180, 255, 255])

    # Maske für rote Bereiche erstellen
    maske_rot_1 = cv2.inRange(hsv_bild, untere_rot_1, obere_rot_1)
    maske_rot_2 = cv2.inRange(hsv_bild, untere_rot_2, obere_rot_2)
    maske_rot = maske_rot_1 | maske_rot_2

    # Nur die roten Bereiche im Bild extrahieren
    rot_objekte = cv2.bitwise_and(bild, bild, mask=maske_rot)

    # Originalbild und gefilterte rote Objekte anzeigen
    cv2.imshow("Originalbild", bild)
    cv2.imshow("Rote Objekte", rot_objekte)

    # Warten bis eine Taste gedrückt wird und Fenster schließen
    cv2.waitKey(0)
    cv2.destroyAllWindows()
