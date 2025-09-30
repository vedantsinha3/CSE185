import cv2
import numpy as np
from matplotlib import pyplot as plt

try:
    img = cv2.imread('bay.png', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError("The image 'bay.png' could not be found. Please check the file path.")

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')
    img_equalized = cdf_final[img]    
    hist_equalized, bins_equalized = np.histogram(img_equalized.flatten(), 256, [0, 256])


    plt.figure(figsize=(15, 10))
    plt.suptitle('Histogram Equalization', fontsize=16)

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Low-Contrast Image')
    plt.axis('off')


    plt.subplot(2, 2, 2)
    plt.plot(cdf_normalized, color='b', label='CDF')
    plt.hist(img.flatten(), 256, [0, 256], color='r', label='Histogram')
    plt.xlim([0, 256])
    plt.legend(loc='upper left')
    plt.title('Original Histogram & CDF')

    plt.subplot(2, 2, 3)
    plt.imshow(img_equalized, cmap='gray')
    plt.title('High-Contrast Equalized Image')
    plt.axis('off')
    
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