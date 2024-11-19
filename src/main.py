# main.py
import cv2
import os
import segmentation
import color_correction
import color_detection

def draw_legend(image, rois, colors):
    # Originalbildgröße ermitteln
    img_height, img_width, _ = image.shape
    
    # Segment-Nummern und Farben in die ROIs schreiben
    for i, ((x, y, w, h), color) in enumerate(zip(rois, colors)):
        # Mittelpunkte des ROIs berechnen
        center_x, center_y = x + w // 2, y + h // 2

        # Segmentnummer auf das Bild schreiben
        segment_number = str(i + 1)
        cv2.putText(image, segment_number, (center_x - 10, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
    # Legende unten rechts erstellen
    legend_start_x = img_width - 200  # Legende unten rechts, angepasst an die Bildgröße
    legend_start_y = img_height - 300
    legend_rect_height = 30

    for i, color in enumerate(colors):
        # Farbbox der Legende
        color_box_position = (legend_start_x, legend_start_y + i * legend_rect_height)
        cv2.rectangle(image, color_box_position, 
                      (legend_start_x + 30, legend_start_y + (i + 1) * legend_rect_height),
                      get_bgr_color(color), -1)

        # Farbname und Segmentnummer in der Legende schreiben
        text_position = (legend_start_x + 40, legend_start_y + i * legend_rect_height + 20)
        cv2.putText(image, f"{i + 1}: {color}", text_position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def get_bgr_color(color_name):
    # Farbtabellen für OpenCV (BGR Format)
    color_dict = {
        "Red": (0, 0, 255),
        "Orange": (0, 165, 255),
        "Yellow": (0, 255, 255),
        "Green": (0, 255, 0),
        "Blue": (255, 0, 0),
        "Unknown": (200, 200, 200)  # Grau für unbekannte Farben
    }
    return color_dict.get(color_name, (255, 255, 255))  # Weiß als Standardfarbe

def main(image_path):
    # Bild laden
    image = cv2.imread(image_path)
    
    # Überprüfen, ob das Bild erfolgreich geladen wurde
    if image is None:
        print("Fehler: Das Bild konnte nicht geladen werden. Bitte überprüfe den Bildpfad.")
        return
    
    # Schritt 1: Segmentierung und ROI-Bounding Boxes erhalten
    rois = segmentation.get_rois(image)
    
    # Schritt 2: Farbkorrektur der einzelnen ROIs
    corrected_rois = [color_correction.correct_colors(image[y:y+h, x:x+w]) for (x, y, w, h) in rois]
    
    # Schritt 3: Farberkennung in den ROIs
    colors = [color_detection.detect_color(roi) for roi in corrected_rois]
    
    # Schritt 4: Legende und Segmentnummern im Bild zeichnen
    draw_legend(image, rois, colors)
    
    # Bild anzeigen
    cv2.imshow("Rubik's Cube mit Legende", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = os.path.join("..", "Logitech Webcam", "Picture 1.jpg")
    main(image_path)
