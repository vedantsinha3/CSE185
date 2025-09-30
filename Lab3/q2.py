import cv2
import numpy as np
from matplotlib import pyplot as plt


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 20.0) -> np.ndarray:
    """Add Gaussian noise to a grayscale image.

    Args:
        image: Grayscale input image (uint8).
        mean: Mean of Gaussian noise.
        sigma: Standard deviation of Gaussian noise.

    Returns:
        Noisy image (uint8), values clipped to [0, 255].
    """
    image_float = image.astype(np.float32)
    noise = np.random.normal(loc=mean, scale=sigma, size=image.shape).astype(np.float32)
    noisy = image_float + noise
    noisy_clipped = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_clipped


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.02, s_vs_p: float = 0.5) -> np.ndarray:
    """Add Salt-and-Pepper noise to a grayscale image.

    Args:
        image: Grayscale input image (uint8).
        amount: Proportion of image pixels to replace with noise (0 to 1).
        s_vs_p: Proportion of salt (white) vs pepper (black). 0.5 = equal.

    Returns:
        Noisy image (uint8), with salt and pepper applied.
    """
    noisy = image.copy()
    num_pixels = image.size

    # Salt (set to 255)
    num_salt = int(np.ceil(amount * num_pixels * s_vs_p))
    coords_salt_rows = np.random.randint(0, image.shape[0], num_salt)
    coords_salt_cols = np.random.randint(0, image.shape[1], num_salt)
    noisy[coords_salt_rows, coords_salt_cols] = 255

    # Pepper (set to 0)
    num_pepper = int(np.ceil(amount * num_pixels * (1.0 - s_vs_p)))
    coords_pepper_rows = np.random.randint(0, image.shape[0], num_pepper)
    coords_pepper_cols = np.random.randint(0, image.shape[1], num_pepper)
    noisy[coords_pepper_rows, coords_pepper_cols] = 0

    return noisy


def apply_mean_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply mean filtering using a normalized box filter of given kernel size."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / float(kernel_size * kernel_size)
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filtering using the specified odd kernel size."""
    return cv2.medianBlur(src=image, ksize=kernel_size)


def compute_psnr(reference: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between reference and target images (grayscale)."""
    return cv2.PSNR(reference, target)


def main() -> None:
    # Step 1: Read the input image and convert to grayscale
    image_path = 'lena.png'
    lena = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if lena is None:
        raise FileNotFoundError("The image 'lena.png' could not be found. Please place it next to this script.")

    # Step 2: Add Gaussian noise and Salt/Pepper noise (custom implementations)
    gaussian_sigma = 20.0  # adjust as desired
    sp_amount = 0.05       # percent of pixels to corrupt
    sp_ratio = 0.5         # 50% salt, 50% pepper

    lena_gauss = add_gaussian_noise(lena, mean=0.0, sigma=gaussian_sigma)
    lena_sp = add_salt_and_pepper_noise(lena, amount=sp_amount, s_vs_p=sp_ratio)

    # Step 3: Mean and Median filtering in 5x5 windows
    kernel_size = 5
    lena_gauss_mean = apply_mean_filter(lena_gauss, kernel_size)
    lena_gauss_median = apply_median_filter(lena_gauss, kernel_size)

    lena_sp_mean = apply_mean_filter(lena_sp, kernel_size)
    lena_sp_median = apply_median_filter(lena_sp, kernel_size)

    # Step 4: Evaluate and compare (PSNR) and visualize
    print('PSNR vs Original (higher is better):')
    print(f'  Gaussian noisy:           {compute_psnr(lena, lena_gauss):.2f} dB')
    print(f'  Gaussian -> mean (5x5):   {compute_psnr(lena, lena_gauss_mean):.2f} dB')
    print(f'  Gaussian -> median (5x5): {compute_psnr(lena, lena_gauss_median):.2f} dB')
    print(f'  S&P noisy:                {compute_psnr(lena, lena_sp):.2f} dB')
    print(f'  S&P -> mean (5x5):        {compute_psnr(lena, lena_sp_mean):.2f} dB')
    print(f'  S&P -> median (5x5):      {compute_psnr(lena, lena_sp_median):.2f} dB')

    # Save outputs next to this script
    cv2.imwrite('lena_gray.png', lena)
    cv2.imwrite('lena_gaussian.png', lena_gauss)
    cv2.imwrite('lena_gaussian_mean5.png', lena_gauss_mean)
    cv2.imwrite('lena_gaussian_median5.png', lena_gauss_median)
    cv2.imwrite('lena_sp.png', lena_sp)
    cv2.imwrite('lena_sp_mean5.png', lena_sp_mean)
    cv2.imwrite('lena_sp_median5.png', lena_sp_median)

    # Visualization: two rows (Gaussian pipeline, S&P pipeline)
    plt.figure(figsize=(16, 8))
    plt.suptitle('Image Denoising (5x5 Mean vs Median)', fontsize=16)

    # Row 1: Gaussian
    plt.subplot(2, 4, 1)
    plt.imshow(lena, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(lena_gauss, cmap='gray')
    plt.title('Gaussian noisy')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(lena_gauss_mean, cmap='gray')
    plt.title('Mean 5x5')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(lena_gauss_median, cmap='gray')
    plt.title('Median 5x5')
    plt.axis('off')

    # Row 2: Salt & Pepper
    plt.subplot(2, 4, 5)
    plt.imshow(lena, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(lena_sp, cmap='gray')
    plt.title('S&P noisy')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(lena_sp_mean, cmap='gray')
    plt.title('Mean 5x5')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(lena_sp_median, cmap='gray')
    plt.title('Median 5x5')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()


