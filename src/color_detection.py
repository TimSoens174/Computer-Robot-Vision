# color_detection.py
import cv2
import numpy as np

def detect_color(roi):
    # In HSV-Farbraum konvertieren
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Durchschnittlicher Farbton des ROIs berechnen
    mean_hue = np.mean(hsv[:, :, 0])
    
    # Farben auf Basis des Farbtons klassifizieren
    if 0 <= mean_hue < 10 or 160 <= mean_hue <= 180:
        color = "Red"
    elif 10 <= mean_hue < 30:
        color = "Orange"
    elif 30 <= mean_hue < 90:
        color = "Yellow"
    elif 90 <= mean_hue < 150:
        color = "Green"
    elif 150 <= mean_hue < 180:
        color = "Blue"
    else:
        color = "Unknown"
    
    return color
