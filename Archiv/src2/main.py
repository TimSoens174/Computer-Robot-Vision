# main.py
import cv2
import os
import Contour
import Color
import time

livecam = False
camera = 1
image_path = os.path.join("Pictures", "Picture 10.jpg")
color_correction = True

sigma_b = 18.84
sigma_g = 18.92
sigma_r = 18.23
max_b = 173
max_g = 155
max_r = 145

color_correction_ground_truth = [sigma_b, sigma_g, sigma_r, max_b, max_g, max_r]

def main(image):
    if image is None:
        print("Fehler: Das Bild konnte nicht geladen werden. Bitte überprüfe den Bildpfad.")
        return
    
    # Bild zuschneiden
    cut_image = Contour.cut_image(image, int(image.shape[1]/4), int(image.shape[0]/6), int((1-(2/4))*image.shape[1]), int((1-(2/6))*image.shape[0]))

    # Äußeren grauen Rahmen erkennen
    frame_image, outer_correction_frame = Contour.detect_outer_gray_frame(cut_image)

    if(outer_correction_frame[2] != 0 and color_correction == True):
        # Kleinere graue Rahmen berechnen
        frame_image, inner_correction_frame = Contour.small_greyFrame(frame_image, outer_correction_frame, 0.85)

        # Korrekturwerte berechnen
        correction_values = Color.getCorrectionValues(cut_image, outer_correction_frame, inner_correction_frame)
        #print(correction_values)

        # Farbkorrektur
        corrected_image = Color.correctImage(cut_image, color_correction_ground_truth, correction_values)

        Color.showHistogram(cut_image, corrected_image, outer_correction_frame, inner_correction_frame)

        # Farbfilter
        result_image, erg = Color.filter(corrected_image)

    else:   
        # Farbfiler
        result_image, erg = Color.filter(cut_image)
        

    #cv2.imshow('corr', corrected_image)

    # Bild anzeigen
    cv2.imshow("Rubik's Cube mit Legende", result_image)
    if erg:
        print(erg)

    return


if __name__ == "__main__":
    
    if livecam:
        cap = cv2.VideoCapture(camera)

        # Überprüfen, ob die Kamera geöffnet werden konnte
        if not cap.isOpened():
            print("Fehler: Kamera konnte nicht geöffnet werden!")

        while True:
            # Ein Frame von der Kamera lesen
            ret, frame = cap.read()
            
            if not ret:
                print("Fehler beim Lesen des Kamerabildes!")
                break

            main(frame)
            time.sleep(0.3)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Ressourcen freigeben
        cap.release()
        cv2.destroyAllWindows()
    else: 
        image = cv2.imread(image_path)
        main(image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 