# color_correction.py
import cv2
import numpy as np

def correct_colors(roi, c0=1.2, c1=10):
    # Konvertiere in float, damit die Berechnungen keine Überläufe erzeugen
    roi = roi.astype(np.float32)
    
    # Farbkorrektur durchführen
    roi_corrected = c0 * (roi + c1)
    
    # Werte auf gültigen Bereich begrenzen und zurück zu uint8 konvertieren
    roi_corrected = np.clip(roi_corrected, 0, 255).astype(np.uint8)
    return roi_corrected
