import cv2
import numpy as np
import os


def Filter(bild):
    # Bild in den HSV-Farbraum umwandeln
    hsv_bild = cv2.cvtColor(bild, cv2.COLOR_BGR2HSV)

    # Farbbereiche definieren
    farb_bereiche = {
        "Rot": [(np.array([0, 90, 70]), np.array([5, 255, 255])),
                (np.array([170, 90, 70]), np.array([180, 255, 255]))],
        "Blau": [(np.array([100, 130, 70]), np.array([140, 255, 255]))],
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

        #cv2.imshow(f"test {i}", maske)

        # Konturen der Objekte finden
        konturen, hierarchie = cv2.findContours(maske, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_konturen = []
        for kontur in konturen:
            x, y, w, h = cv2.boundingRect(kontur)
            if (w * h > 3000) & (w * h < 10000) & (w / h > 0.9) & (w / h < 1.1):
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
    print("Gefundene Objekte ohne Kinder:")
    for i, obj in enumerate(ergebnisse, 1):
        print(f"Objekt {i}: Farbe={obj['Farbe']}, X={obj['x']}, Y={obj['y']}, "
              f"Breite={obj['width']}, Höhe={obj['height']}, Durchschnittlicher Hue={obj['average_hue']:.2f}")
    return ergebnisbild, ergebnisse
