import cv2
import numpy as np
from matplotlib import pyplot as plt


def compute_sobel_gradients(image_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_x = cv2.Sobel(src=image_gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad_y = cv2.Sobel(src=image_gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    return grad_x, grad_y


def gradient_magnitude(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    magnitude = cv2.magnitude(grad_x, grad_y)
    return magnitude


def normalize_to_uint8(image_float: np.ndarray) -> np.ndarray:
    min_val, max_val = float(np.min(image_float)), float(np.max(image_float))
    if max_val - min_val < 1e-6:
        return np.zeros_like(image_float, dtype=np.uint8)
    norm = (image_float - min_val) / (max_val - min_val)
    return (norm * 255.0).astype(np.uint8)


def threshold_edges(magnitude: np.ndarray, thresholds: list[int]) -> dict[int, np.ndarray]:
    mag_uint8 = normalize_to_uint8(magnitude)
    edges = {}
    for t in thresholds:
        _, edge = cv2.threshold(mag_uint8, t, 255, cv2.THRESH_BINARY)
        edges[t] = edge
    return edges


def main() -> None:
    image_path = 'lena.png'
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError("The image 'lena.png' could not be found. Please place it next to this script.")

    image_gray_blur = cv2.GaussianBlur(image_gray, ksize=(3, 3), sigmaX=0)

    grad_x, grad_y = compute_sobel_gradients(image_gray_blur)

    mag = gradient_magnitude(grad_x, grad_y)

    cv2.imwrite('lena_gray.png', image_gray)
    cv2.imwrite('lena_grad_x.png', normalize_to_uint8(grad_x))
    cv2.imwrite('lena_grad_y.png', normalize_to_uint8(grad_y))
    cv2.imwrite('lena_grad_mag.png', normalize_to_uint8(mag))

    thresholds = [30, 60, 90, 120, 150]
    edge_maps = threshold_edges(mag, thresholds)
    for t, edge in edge_maps.items():
        cv2.imwrite(f'lena_edges_T{t}.png', edge)

    plt.figure(figsize=(14, 8))
    plt.suptitle('Image Gradients and Edge Maps', fontsize=16)

    plt.subplot(2, 3, 1)
    plt.imshow(image_gray, cmap='gray')
    plt.title('Input (Gray)')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(normalize_to_uint8(grad_x), cmap='gray')
    plt.title('Sobel Gx')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(normalize_to_uint8(grad_y), cmap='gray')
    plt.title('Sobel Gy')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(normalize_to_uint8(mag), cmap='gray')
    plt.title('Gradient Magnitude')
    plt.axis('off')

    t_values = thresholds[:2]
    plt.subplot(2, 3, 5)
    plt.imshow(edge_maps[t_values[0]], cmap='gray')
    plt.title(f'Edges T={t_values[0]}')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(edge_maps[t_values[1]], cmap='gray')
    plt.title(f'Edges T={t_values[1]}')
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()


