import cv2
import numpy as np

def create_inverted_template(output_path):
    """
    Creates an inverted square template image with a 3x3 grid structure.
    """
    # Define the size of the image
    image_size = 300  # 300x300 pixels
    line_thickness = 6
    cell_size = image_size // 3

    # Create a blank black image
    template = np.zeros((image_size, image_size), dtype=np.uint8)

    # Draw the outer square
    cv2.rectangle(template, (0 + line_thickness // 2, 0 + line_thickness // 2), (image_size-1-line_thickness // 2, image_size-1-line_thickness // 2), (255, 255, 255), thickness=line_thickness)

    # Draw horizontal grid lines
    for i in range(1, 3):  # Two lines
        y = i * cell_size
        cv2.line(template, (0, y), (image_size, y), (255, 255, 255), thickness=line_thickness)

    # Draw vertical grid lines
    for i in range(1, 3):  # Two lines
        x = i * cell_size
        cv2.line(template, (x, 0), (x, image_size), (255, 255, 255), thickness=line_thickness)

    # Save the template
    cv2.imwrite(output_path, template)
    print(f"Template saved to: {output_path}")

    # Show the template for debugging
    cv2.imshow("Template", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Usage
create_inverted_template("rubiks_cube_inverted_template.jpg")
