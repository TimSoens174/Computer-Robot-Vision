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
camera = 0
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
    reduced_outer_grey_frame = ip.reduce_boundingbox(outer_grey_frame, OUTER_GREY_FRAME_FAKTOR)
    inner_grey_frame = ip.reduce_boundingbox(reduced_outer_grey_frame, INNER_GREY_FRAME_FAKTOR)


    cropped_image = ip.cut_image(image, outer_grey_frame)
    # Relocate Boundingbox in cropped frame
    cropped_outer_grey_frame = [0, 0, outer_grey_frame[2], outer_grey_frame[3]]
    croped_reduced_outer_grey_frame = ip.reduce_boundingbox(cropped_outer_grey_frame, OUTER_GREY_FRAME_FAKTOR)
    croped_inner_grey_frame = ip.reduce_boundingbox(croped_reduced_outer_grey_frame, INNER_GREY_FRAME_FAKTOR)

    # Zeichne die Bounding Boxen (grün)
    # debug_cropped_image = cropped_image.copy()
    # cv2.rectangle(debug_cropped_image, (croped_reduced_outer_grey_frame[0],croped_reduced_outer_grey_frame[1]), (croped_reduced_outer_grey_frame[0]+croped_reduced_outer_grey_frame[2],croped_reduced_outer_grey_frame[1]+croped_reduced_outer_grey_frame[3]), (0, 255, 0), 2)
    # cv2.rectangle(debug_cropped_image, (croped_inner_grey_frame[0],croped_inner_grey_frame[1]), (croped_inner_grey_frame[0]+croped_inner_grey_frame[2],croped_inner_grey_frame[1]+croped_inner_grey_frame[3]), (0, 255,0), 2)
    # cv2.imshow('Cropped Image',debug_cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corrected_image = color.correctImage(cropped_image, color_correction_ground_truth, croped_reduced_outer_grey_frame, croped_inner_grey_frame)
    if corrected_image is None:
        print("Error: Image correction failed.")
        return
    else:
        # cv2.imshow('Corrected Image', corrected_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pass


#  valid_element = {
#                 "bbox": cv2.boundingRect(approx),
#                 "area_focus_point": area_focus_point,
#                 "color": None,
#                 "grid_position": None,
#                 "avg_hue": None,
#             }

    dict_color, color_mask = color.color_filter(corrected_image)
    print("Color Dictionary:")
    for element in dict_color:
         print(element)

    dict_edge, edge_mask = edge.edge_detection(corrected_image)
    print("Edge Dictionary:")
    for element in dict_edge:
         print(element)

    # Gruppierung und Sortierung
    dict_color_sorted, x_groups1, y_groups1 = sorting.assign_grid_positions(dict_color)
    dict_edge_sorted, x_groups2, y_groups2 = sorting.assign_grid_positions(dict_edge)   

    # Ausgabe der Gruppen
    print("Dict Color Gruppen:")
    print("X-Gruppen:", x_groups1)
    print("Y-Gruppen:", y_groups1)

    print("\nDict Edge Gruppen:")
    print("X-Gruppen:", x_groups2)
    print("Y-Gruppen:", y_groups2)

    # Merge the dictionaries and groups
    merged_dict, merged_x_group, merged_y_group = sorting.merge_dictionaries(dict_color_sorted, dict_edge_sorted, x_groups1, y_groups1, x_groups2, y_groups2)

    # Interpolate missing entries
    interpolated_dict = sorting.interpolate_missing_entries(merged_dict, merged_x_group, merged_y_group)

    # Farberkennung für Noneeinträge

    final_dict = color.update_color_for_none_entries(interpolated_dict, corrected_image)

    # Ausgabe der finalen Daten
    print("\nKompensiertes Dictionary:")
    for pos, entry in sorted(final_dict.items(), key=lambda x: x[1]['grid_position']):
        print(f"Position {entry['grid_position']}: {entry}")

    # Berechne die Edge- und Color-Masken
    # edge_mask = edge.create_edge_mask(image, outer_grey_frame)
    # color_mask = color.create_color_mask(image, outer_grey_frame)
    # edge_mask = corrected_image.copy()
    # color_mask = corrected_image.copy()

    # Zeige die Bilder zusammen
    utils.draw_images(image, corrected_image, edge_mask, color_mask, outer_grey_frame, reduced_outer_grey_frame, inner_grey_frame, final_dict)
 

if __name__ == "__main__":
    
    image_path = os.path.join("Pictures2", "Picture 13.jpg")
    image = cv2.imread(image_path)
    
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
