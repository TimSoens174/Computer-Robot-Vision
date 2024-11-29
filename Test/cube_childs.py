import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load image
image_path = "Pictures2/Picture 10.jpg"  # Update the path if needed
image = cv2.imread(image_path)

# Global variables for trackbars
params = {"Blur Kernel": 5, "Canny Low": 50, "Canny High": 150}

# Define processing pipeline
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

    # Morphological operations (Post-Canny)
    edges_refined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(edges_refined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = image.copy()
    cv2.drawContours(all_contours, contours, -1, (0, 255, 0), 1)

    # Filter contours with specific properties
    parent_contours = []
    if hierarchy is not None:
        for idx, h in enumerate(hierarchy[0]):
            child_count = 0
            child = h[2]  # Index of the first child
            while child != -1:
                child_count += 1
                child = hierarchy[0][child][0]

            if 5 <= child_count <= 9:
                peri = cv2.arcLength(contours[idx], True)
                approx = cv2.approxPolyDP(contours[idx], 0.02 * peri, True)
                if len(approx) == 4:  # Quadrilateral
                    parent_contours.append(contours[idx])

    detected_image = image.copy()
    cv2.drawContours(detected_image, parent_contours, -1, (255, 0, 0), 3)

    return gray, blurred, morphed, edges, edges_refined, all_contours, detected_image

# Callback to update the pipeline
def update_pipeline(val):
    """Callback function to update images when sliders are moved."""
    kernel_size = slider_kernel.val * 2 + 1  # Ensure odd kernel size
    canny_low = slider_canny_low.val
    canny_high = slider_canny_high.val

    # Process the image
    gray, blurred, morphed, edges, edges_refined, all_contours, detected_image = process_image(
        int(kernel_size), int(canny_low), int(canny_high)
    )

    # Update subplots
    axs[0].imshow(gray, cmap='gray')
    axs[1].imshow(blurred, cmap='gray')
    axs[2].imshow(morphed, cmap='gray')
    axs[3].imshow(edges, cmap='gray')
    axs[4].imshow(edges_refined, cmap='gray')
    axs[5].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

    # Refresh the canvas
    plt.draw()

# Set up the figure and axes
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25, hspace=0.3, wspace=0.3)
axs = axs.ravel()

# Initial placeholders for images
titles = [
    "Grayscale Image",
    "Blurred Image",
    "Morphology (Pre-Canny)",
    "Canny Edges",
    "Edges Refined (Post-Canny)",
    "Filtered Parent Contours",
]
for ax, title in zip(axs, titles):
    ax.set_title(title)
    ax.axis("off")

# Initial processing
gray, blurred, morphed, edges, edges_refined, all_contours, detected_image = process_image(
    params["Blur Kernel"] * 2 + 1, params["Canny Low"], params["Canny High"]
)
axs[0].imshow(gray, cmap='gray')
axs[1].imshow(blurred, cmap='gray')
axs[2].imshow(morphed, cmap='gray')
axs[3].imshow(edges, cmap='gray')
axs[4].imshow(edges_refined, cmap='gray')
axs[5].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))

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
