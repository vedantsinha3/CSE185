import cv2
import numpy as np
from matplotlib import pyplot as plt

# Ensure the image 'bay.png' is in the same folder as the script
try:
    # --- Step 1: Read the image and compute its histogram and CDF ---
    
    # Read the input image in grayscale
    img = cv2.imread('bay.png', cv2.IMREAD_GRAYSCALE)
    
    # Check if the image was loaded successfully
    if img is None:
        raise FileNotFoundError("The image 'bay.png' could not be found. Please check the file path.")

    # Compute the histogram of the original image
    # The `flatten()` method converts the 2D image array to a 1D array
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize the CDF for visualization purposes
    cdf_normalized = cdf * float(hist.max()) / cdf.max()


    # --- Step 2: Apply histogram equalization using the CDF ---
    
    # Create a lookup table (LUT) using the CDF
    # First, mask out the zero values in the CDF to avoid division by zero errors
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Apply the histogram equalization formula
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    
    # Fill the masked values back with 0
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    
    # Apply the lookup table to create the equalized image
    img_equalized = cdf_final[img]


    # --- Step 3: Compute and visualize the histogram of the output image ---
    
    # Compute the histogram of the equalized image
    hist_equalized, bins_equalized = np.histogram(img_equalized.flatten(), 256, [0, 256])


    # --- Visualization ---

    # Set up the plot
    plt.figure(figsize=(15, 10))
    plt.suptitle('Histogram Equalization', fontsize=16)

    # Plot Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Low-Contrast Image')
    plt.axis('off')

    # Plot Original Histogram and CDF
    plt.subplot(2, 2, 2)
    plt.plot(cdf_normalized, color='b', label='CDF')
    plt.hist(img.flatten(), 256, [0, 256], color='r', label='Histogram')
    plt.xlim([0, 256])
    plt.legend(loc='upper left')
    plt.title('Original Histogram & CDF')

    # Plot Equalized Image
    plt.subplot(2, 2, 3)
    plt.imshow(img_equalized, cmap='gray')
    plt.title('High-Contrast Equalized Image')
    plt.axis('off')
    
    # Plot Equalized Histogram
    plt.subplot(2, 2, 4)
    plt.hist(img_equalized.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.title('Equalized Histogram')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")