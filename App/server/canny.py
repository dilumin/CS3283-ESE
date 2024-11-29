import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter
from skimage import io, color


def normalize(img):
    """Normalize the image to range [0, 1]."""
    return (img - img.min()) / (img.max() - img.min())

def sobel_filter(img, direction):
    """Apply Sobel filter in the given direction ('x' or 'y')."""
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    kernel = Gx if direction == 'x' else Gy
    
    if img.ndim == 3:  # If the image has 3 channels (e.g., RGB), process each separately
        filtered = np.zeros_like(img, dtype=np.float32)
        for i in range(img.shape[2]):
            filtered[..., i] = convolve(img[..., i], kernel)
        return filtered
    else:  # For grayscale images
        return convolve(img, kernel)

def non_maximum_suppression(magnitude, gradient):
    """Perform non-maximum suppression."""
    M, N = magnitude.shape
    nms = np.zeros((M, N), dtype=np.float32)
    angle = gradient * 180.0 / np.pi  # Convert to degrees
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    nms[i, j] = magnitude[i, j]
                else:
                    nms[i, j] = 0

            except IndexError as e:
                pass
    return nms

def crop_center(img, target_width, target_height):
    """Crop the image from the center to the target dimensions."""
    y, x = img.shape[:2]
    start_x = (x - target_width) // 2
    start_y = (y - target_height) // 2
    return img[start_y:start_y + target_height, start_x:start_x + target_width]

def edge_detection(img, target_width=500, target_height=500):
    """Complete edge detection pipeline with center cropping."""
    # Load and preprocess the image
    
    if img.ndim == 3:  # If RGB, convert to grayscale
        img = color.rgb2gray(img)
    img = normalize(img)


    

    # Apply Gaussian smoothing
    img_smoothed = gaussian_filter(img, sigma=1)

    # Compute Sobel gradients
    Gx = sobel_filter(img_smoothed, 'x')
    Gy = sobel_filter(img_smoothed, 'y')

    # Compute gradient magnitude and direction
    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = normalize(magnitude)  # Normalize the gradient magnitude
    gradient = np.arctan2(Gy, Gx)

    # Apply Non-Maximum Suppression
    nms = non_maximum_suppression(magnitude, gradient)

    # Apply thresholding
    threshold = 0.2  # Adjust as needed
    edges = np.zeros_like(nms)
    edges[nms >= threshold] = 1
    return edges