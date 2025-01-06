# main.py
import cv2
import os
import image_preprocessing as ip
import edge_detection as edge
import color_filtering as color
import grid_sorting as sorting
import utils
import time

livecam = False
camera = 1
color_correction = True

sigma_b = 18.84
sigma_g = 18.92
sigma_r = 18.23
max_b = 173
max_g = 155
max_r = 145
INNER_GREY_FRAME_FAKTOR = 0.9
OUTER_GREY_FRAME_FAKTOR = 0.95

color_correction_ground_truth = [sigma_b, sigma_g, sigma_r, max_b, max_g, max_r]

def main(image):

    if image is None:
            print("Fehler: Das Bild konnte nicht geladen werden. Bitte überprüfe den Bildpfad.")
            return

    outer_grey_frame = ip.detect_outer_gray_frame(image)
    cropped_image = ip.cut_image(image, outer_grey_frame)
    # Relocate Boundingbox in cropped frame
    cropped_outer_grey_frame = [0, 0, outer_grey_frame[2], outer_grey_frame[3]]
    reduced_outer_grey_frame = ip.reduce_boundingbox(cropped_outer_grey_frame, OUTER_GREY_FRAME_FAKTOR)
    inner_grey_frame = ip.reduce_boundingbox(reduced_outer_grey_frame, INNER_GREY_FRAME_FAKTOR)

    # Zeichne die Bounding Boxen (grün)
    debug_cropped_image = cropped_image.copy()
    cv2.rectangle(debug_cropped_image, (reduced_outer_grey_frame[0],reduced_outer_grey_frame[1]), (reduced_outer_grey_frame[0]+reduced_outer_grey_frame[2],reduced_outer_grey_frame[1]+reduced_outer_grey_frame[3]), (0, 255, 0), 2)
    cv2.rectangle(debug_cropped_image, (inner_grey_frame[0],inner_grey_frame[1]), (inner_grey_frame[0]+inner_grey_frame[2],inner_grey_frame[1]+inner_grey_frame[3]), (0, 255,0), 2)
    cv2.imshow('Cropped Image',debug_cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    corrected_image = color.correctImage(cropped_image, color_correction_ground_truth, reduced_outer_grey_frame, inner_grey_frame)
    if corrected_image is None:
        print("Error: Image correction failed.")
        return
    else:
        cv2.imshow('Corrected Image', corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#  valid_element = {
#                 "bbox": cv2.boundingRect(approx),
#                 "area_focus_point": area_focus_point,
#                 "color": None,
#                 "grid_position": None,
#                 "avg_hue": None,
#             }

    dict_color = color.color_filter(corrected_image)
    print("Color Dictionary:")
    for element in dict_color:
         print(element)

    dict_edge = edge.edge_detection(corrected_image)
    print("Edge Dictionary:")
    for element in dict_edge:
         print(element)

    # Gruppierung und Sortierung
    dict_color_sorted, x_groups1, y_groups1 = sorting.assign_grid_positions(dict_color, threshold=50)
    dict_edge_sorted, x_groups2, y_groups2 = sorting.assign_grid_positions(dict_edge, threshold=50)   

    # Ausgabe der Gruppen
    print("Dict1 Gruppen:")
    print("X-Gruppen:", x_groups1)
    print("Y-Gruppen:", y_groups1)

    print("\nDict2 Gruppen:")
    print("X-Gruppen:", x_groups2)
    print("Y-Gruppen:", y_groups2)

    # Mergen und Interpolation
    merged_dict = sorting.merge_and_interpolate(dict_color_sorted, x_groups1, y_groups1, dict_edge_sorted, x_groups2, y_groups2, threshold=50)

    # Ausgabe der finalen Daten
    print("\nKompensiertes Dictionary:")
    for pos, entry in sorted(merged_dict.items(), key=lambda x: x[1]['grid_position']):
        print(f"Position {entry['grid_position']}: {entry}")

    utils.draw_grid_positions(image, merged_dict, outer_grey_frame)    

if __name__ == "__main__":
    
    image_path = os.path.join("Pictures2", "Picture 13.jpg")
    image = cv2.imread(image_path)
    
    main(image)
