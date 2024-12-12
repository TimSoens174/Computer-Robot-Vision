import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_Werte(image, Box, smallBox):
    #Ausschnitt = image[Box[0]:Box[0]+Box[1], Box[3]:Box[3]+Box[3]] - image[smallBox[0]:smallBox[0]+smallBox[2], smallBox[3]:smallBox[3]+smallBox[3]]

    height, width = image.shape[:2]

    # Maske für die große Box
    mask_large = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask_large, (Box[0], Box[1]), (Box[0] + Box[2], Box[1] + Box[3]), 255, -1)

    # Maske für die kleine Box
    mask_small = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask_small, (smallBox[0], smallBox[1]), (smallBox[0] + smallBox[2], smallBox[1] + smallBox[3]), 255, -1)

    # Bereich zwischen den Boxen: große Maske minus kleine Maske
    mask_between = cv2.subtract(mask_large, mask_small)

    # Pixel im Bereich extrahieren
    all_pixels = image[mask_between == 255]
    pixels = all_pixels[~np.all(all_pixels[:,:] == 0, axis=1)]
    result = cv2.bitwise_and(image, image, mask=mask_between)
    
    #cv2.imshow('test', result)

    sigma_b_s = np.std(pixels[:, 0])  # Blau-Kanal
    sigma_g_s = np.std(pixels[:, 1])  # Grün-Kanal
    sigma_r_s = np.std(pixels[:, 2])  # Rot-Kanal

    peak_b_s = np.argmax(np.histogram(pixels[:,0],bins=256, range=[0, 256])[0])
    peak_g_s = np.argmax(np.histogram(pixels[:,1],bins=256, range=[0, 256])[0])
    peak_r_s = np.argmax(np.histogram(pixels[:,2],bins=256, range=[0, 256])[0])
    
    fig, (ax1, ax2, ax3)= plt.subplots(1,3)
    ax1.hist(pixels[:,0], bins=256, range=[0, 256])
    ax2.hist(pixels[:,1], bins=256, range=[0, 256])
    ax3.hist(pixels[:,2], bins=256, range=[0, 256])
    #plt.show()


    return [sigma_b_s, sigma_g_s, sigma_r_s, peak_b_s, peak_g_s, peak_r_s]


def corrected(image, Ground_thruth, Bildwerte):
    corrected_image = image.copy()
    
    c_0 = (Bildwerte[2]/Ground_thruth[2] + Bildwerte[1]/Ground_thruth[1] + Bildwerte[0]/Ground_thruth[3])/3
    c_r = Bildwerte[5]/c_0 - Ground_thruth[5]
    c_g = Bildwerte[4]/c_0 - Ground_thruth[4]
    c_b = Bildwerte[3]/c_0 - Ground_thruth[3]

    #print(f"Korrekturfaktoren:")
    #print(f"C_B: {c_b:.2f}, C_G: {c_g:.2f}, C_R: {c_r:.2f}, C_0: {c_0:.2f}")

    corrected_image[:, :, 0] = np.clip((image[:, :, 0]/c_0 - c_b), 0, 255)  # Blau-Kanal
    corrected_image[:, :, 1] = np.clip((image[:, :, 1]/c_0 - c_g), 0, 255)  # Grün-Kanal
    corrected_image[:, :, 2] = np.clip((image[:, :, 2]/c_0 - c_r), 0, 255)  # Rot-Kanal
    
    return corrected_image

