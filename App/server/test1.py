import cv2
import numpy as np
import random
from skimage.morphology import skeletonize

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

    # Crop the image from the center
    # img_cropped = crop_center(img, target_width, target_height)
    

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


def crack_detection(img):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a blur to the grayscale image
    blurred_img = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Thresholding to get a binary image (inverse binary to highlight cracks)
    _, binary_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Skeletonize the inverted binary image to get a thin representation of the cracks
    skeleton = skeletonize(binary_img // 255).astype(np.uint8) * 255  # Convert boolean to uint8 for OpenCV

    # Detect edges using Canny edge detection
    med_val = np.median(gray_image)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    # edges = cv2.Canny(image=gray_image, threshold1=lower, threshold2=upper + 50)
    edges = edge_detection(img)

    # Threshold edges for binary image
    _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    # Get the coordinates of the skeleton points
    skeleton_points = np.column_stack(np.where(skeleton > 0))

    if len(skeleton_points) == 0:
        return img, []  # If no skeleton points are detected, return original image and an empty list

    # Set the number of perpendicular lines to visualize
    num_perpendicular_lines = 20
    perpendicular_distances = []  # To store the distances of intersections

    # Draw detected edges in red color on the original image
    img_with_edges = img.copy()
    img_with_edges[edges > 0] = [0, 0, 255]  # Red color for edges

    # Iterate to add perpendicular lines
    for _ in range(num_perpendicular_lines):
        # Randomly select a point on the skeleton
        random_point = skeleton_points[random.randint(0, len(skeleton_points) - 1)]

        # Get the neighbors for directional vector (choosing nearby points)
        index = np.where((skeleton_points[:, 0] == random_point[0]) & (skeleton_points[:, 1] == random_point[1]))[0][0]
        prev_point = skeleton_points[index - 1] if index > 0 else random_point
        next_point = skeleton_points[index + 1] if index < len(skeleton_points) - 1 else random_point

        # Calculate the direction vector and perpendicular vector
        direction_vector = (next_point[0] - prev_point[0], next_point[1] - prev_point[1])
        perpendicular_vector = (-direction_vector[1], direction_vector[0])

        # Normalize the perpendicular vector
        length = np.sqrt(perpendicular_vector[0] ** 2 + perpendicular_vector[1] ** 2)
        if length == 0:  # Avoid division by zero
            continue
        perpendicular_vector = (perpendicular_vector[0] / length, perpendicular_vector[1] / length)

        # Define the length of the perpendicular line
        perpendicular_length = 30

        # Calculate the start and end points of the perpendicular line
        start_perpendicular = (int(random_point[0] - perpendicular_length * perpendicular_vector[0]),
                               int(random_point[1] - perpendicular_length * perpendicular_vector[1]))

        end_perpendicular = (int(random_point[0] + perpendicular_length * perpendicular_vector[0]),
                             int(random_point[1] + perpendicular_length * perpendicular_vector[1]))

        # Draw the perpendicular line on the image with edges
        cv2.line(img_with_edges, (start_perpendicular[1], start_perpendicular[0]),
                 (end_perpendicular[1], end_perpendicular[0]), (0, 255, 0), 2)

        # Detect intersections of the perpendicular line with edges
        intersections = []
        rr, cc = np.linspace(start_perpendicular[0], end_perpendicular[0], num=100).astype(int), \
                 np.linspace(start_perpendicular[1], end_perpendicular[1], num=100).astype(int)

        for r, c in zip(rr, cc):
            if 0 <= r < edges.shape[0] and 0 <= c < edges.shape[1]:
                if edges[r, c] > 0:  # Check if the pixel is part of the edge
                    intersections.append((r, c))

        # If there are intersections, plot them and calculate distance
        if len(intersections) >= 2:
            first_intersection = intersections[0]
            last_intersection = intersections[-1]

            # Plot intersection points
            cv2.circle(img_with_edges, (first_intersection[1], first_intersection[0]), 3, (255, 0, 0), -1)
            cv2.circle(img_with_edges, (last_intersection[1], last_intersection[0]), 3, (255, 0, 0), -1)

            # Calculate the distance between intersections
            distance = np.sqrt((last_intersection[0] - first_intersection[0]) ** 2 +
                               (last_intersection[1] - first_intersection[1]) ** 2)
            if distance > 0:
                perpendicular_distances.append(round(distance,3))


    # Return the modified image and the calculated distances (for further use)
    return img_with_edges, perpendicular_distances
image_path = './sample/1.jpg'  # Replace with your image path
img1 = cv2.imread(image_path)

img_with_edges, perpendicular_distances = crack_detection(img1)

cv2.imshow("Detected Edges", img_with_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


