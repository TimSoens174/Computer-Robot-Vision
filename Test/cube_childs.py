import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load image
image_path = "Pictures2/Picture 10.jpg"  # Update the path if needed
image = cv2.imread(image_path)

# Global variables for trackbars
params = {"Blur Kernel": 2, "Canny Low": 25, "Canny High": 180}

def sort_rois_into_grid(rois, image_width, image_height):
    # Get grid cell height and width
    cell_height = image_height // 3
    cell_width = image_width // 3

    # Sort ROIs by their top-left corner position (first by y, then by x)
    print("before sort",rois)
    
    # We want to make sure we have exactly 9 ROIs for the 3x3 grid
    # Now, we map these sorted ROIs to the expected grid positions
    sorted_rois = []
    for i in range(len(rois)):
        # Expected positions in a 3x3 grid
        row = i // 3
        col = i % 3
        ex = col * cell_width
        ey = row * cell_height

        # Find the closest ROI that fits into the expected grid cell
        closest_roi = None
        for roi in rois:
            cx, cy, cw, ch = roi
            if (ex <= cx < (ex + cell_width) and
                ey <= cy < (ey + cell_height)):
                closest_roi = roi
                break
        
        if closest_roi is None:
            # If no closest ROI found, insert a placeholder (0, 0, 0, 0)
            sorted_rois.append((0, 0, 0, 0))
        else:
            sorted_rois.append(closest_roi)
    print("sorted rois", sorted_rois)
    return sorted_rois

def process_image(kernel_size, canny_low, canny_high):
    """Process the image with the given parameters."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Morphological operations (Pre-Canny)
    morphed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Apply Canny edge detection
    edges = cv2.Canny(morphed, canny_low, canny_high)

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours for visualization
    all_contours = image.copy()
    cv2.drawContours(all_contours, contours, -1, (0, 255, 0), 1)

    parent_contours = []
    child_contours = []

    # Filter and visualize contours based on hierarchy
    if hierarchy is not None:
        for idx, h in enumerate(hierarchy[0]):
            child_count = 0
            child = h[2]  # Index of the first child
            while child != -1:
                child_count += 1
                child_contours.append(contours[child])  # Collect child contours
                child = hierarchy[0][child][0]  # Move to next sibling

            if 5 <= child_count <= 9:
                peri = cv2.arcLength(contours[idx], True)
                approx = cv2.approxPolyDP(contours[idx], 0.02 * peri, True)
                if len(approx) == 4:  # Quadrilateral
                    parent_contours.append(contours[idx])

    # Draw detected parent and child contours
    detected_image = image.copy()
    cv2.drawContours(detected_image, parent_contours, -1, (255, 0, 0), 3)  # Parents in blue
    cv2.drawContours(detected_image, child_contours, -1, (0, 0, 255), 2)  # Children in red

    filtered_contours_image = image.copy()
    cv2.drawContours(filtered_contours_image, parent_contours, -1, (255, 0, 0), 3)  # Only filtered contours

    # Display detected image with parent and child contours
    cv2.imshow("Detected Contours (Parent and Child)", detected_image)

    return gray, blurred, morphed, edges, filtered_contours_image

# Callback to update the pipeline
def update_pipeline(val):
    """Callback function to update images when sliders are moved."""
    kernel_size = slider_kernel.val * 2 + 1  # Ensure odd kernel size
    canny_low = slider_canny_low.val
    canny_high = slider_canny_high.val

    # Process the image
    gray, blurred, morphed, edges, filtered_contours_image = process_image(
        int(kernel_size), int(canny_low), int(canny_high)
    )

    # Update subplots
    axs[0].imshow(gray, cmap='gray')
    axs[1].imshow(blurred, cmap='gray')
    axs[2].imshow(morphed, cmap='gray')
    axs[3].imshow(edges, cmap='gray')
    axs[4].imshow(cv2.cvtColor(filtered_contours_image, cv2.COLOR_BGR2RGB))

    # Refresh the canvas
    plt.draw()

# Set up the figure and axes
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25, hspace=0.3, wspace=0.3)
axs = axs.ravel()

# Initial placeholders for images
titles = [
    "Grayscale Image",
    "Blurred Image",
    "Morphology (Pre-Canny)",
    "Canny Edges",
    "Filtered Parent Contours",
]
for ax, title in zip(axs, titles):
    ax.set_title(title)
    ax.axis("off")

# Initial processing
gray, blurred, morphed, edges, filtered_contours_image = process_image(
    params["Blur Kernel"] * 2 + 1, params["Canny Low"], params["Canny High"]
)
axs[0].imshow(gray, cmap='gray')
axs[1].imshow(blurred, cmap='gray')
axs[2].imshow(morphed, cmap='gray')
axs[3].imshow(edges, cmap='gray')
axs[4].imshow(cv2.cvtColor(filtered_contours_image, cv2.COLOR_BGR2RGB))

# Trackbars
ax_kernel = plt.axes([0.1, 0.15, 0.65, 0.03])
slider_kernel = Slider(ax_kernel, 'Blur Kernel', 1, 20, valinit=params["Blur Kernel"], valstep=1)

ax_canny_low = plt.axes([0.1, 0.1, 0.65, 0.03])
slider_canny_low = Slider(ax_canny_low, 'Canny Low', 0, 255, valinit=params["Canny Low"], valstep=1)

ax_canny_high = plt.axes([0.1, 0.05, 0.65, 0.03])
slider_canny_high = Slider(ax_canny_high, 'Canny High', 0, 255, valinit=params["Canny High"], valstep=1)

# Connect sliders to the callback
slider_kernel.on_changed(update_pipeline)
slider_canny_low.on_changed(update_pipeline)
slider_canny_high.on_changed(update_pipeline)

# Display the figure
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
