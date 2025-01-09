import cv2

def main():
    # Kamera öffnen (0 steht für die Standardkamera)
    cap = cv2.VideoCapture(0)

    # Überprüfen, ob die Kamera geöffnet werden konnte
    if not cap.isOpened():
        print("Fehler: Kamera konnte nicht geöffnet werden!")
        return

    print("Drücke 'q', um die Übertragung zu beenden.")

    while True:
        # Ein Frame von der Kamera lesen
        ret, frame = cap.read()
        
        if not ret:
            print("Fehler beim Lesen des Kamerabildes!")
            break

        # Das Bild anzeigen
        cv2.imshow('Live-Kamerabild', frame)

        # Mit 'q' das Programm beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Ressourcen freigeben
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
