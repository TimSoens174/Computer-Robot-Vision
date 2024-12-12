# main.py
import cv2
import os
import kontur
import Farbkorrektur
import Farbfilter

sigma_b = 22.11
sigma_g = 22.03
sigma_r = 22.10
max_b = 139
max_g = 138
max_r = 137

Ground_Truth = [sigma_b, sigma_g, sigma_r, max_b, max_g, max_r]

def main(image_path):
    # Bild laden
    image = cv2.imread(image_path)
    
    # Überprüfen, ob das Bild erfolgreich geladen wurde
    if image is None:
        print("Fehler: Das Bild konnte nicht geladen werden. Bitte überprüfe den Bildpfad.")
        return
    
    # Schritt 1: Segmentierung und ROI-Bounding Boxes erhalten
    image = kontur.cut_image(image, int(image.shape[1]/4), int(image.shape[0]/6), int((1-(2/4))*image.shape[1]), int((1-(2/6))*image.shape[0]))
    frame_pic = image.copy()

    frame_pic, gb = kontur.detect_outer_gray_frame(frame_pic)

    sgb = kontur.small_greyFrame(gb, 0.85)

    werte = Farbkorrektur.get_Werte(image, gb, sgb)
    print(werte)

    correctet_image = Farbkorrektur.corrected(image, Ground_Truth, werte)

    #cv2.imshow('corr', correctet_image)

    ergbild, erg = Farbfilter.Filter(correctet_image)
    
    # Bild anzeigen
    cv2.imshow("Rubik's Cube mit Legende", ergbild)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = os.path.join("Pictures", "Picture 10.jpg")
    main(image_path)
