import cv2
import numpy as np
import random
from skimage.morphology import skeletonize
from canny import edge_detection

# def crack_detection(img):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply a blur to the grayscale image
#     blurred_img = cv2.GaussianBlur(gray_image, (3, 3), 0)

#     # Thresholding to get a binary image (inverse binary to highlight cracks)
#     _, binary_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY_INV)

#     # Skeletonize the inverted binary image to get a thin representation of the cracks
#     skeleton = skeletonize(binary_img // 255).astype(np.uint8) * 255  # Convert boolean to uint8 for OpenCV

#     # Detect edges using Canny edge detection
#     med_val = np.median(gray_image)
#     lower = int(max(0, 0.7 * med_val))
#     upper = int(min(255, 1.3 * med_val))
#     edges = cv2.Canny(image=gray_image, threshold1=lower, threshold2=upper + 50)

#     # Threshold edges for binary image
#     _, edges = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

#     # Get the coordinates of the skeleton points
#     skeleton_points = np.column_stack(np.where(skeleton > 0))

#     if len(skeleton_points) == 0:
#         return img, []  # If no skeleton points are detected, return original image and an empty list

#     # Set the number of perpendicular lines to visualize
#     num_perpendicular_lines = 20
#     perpendicular_distances = []  # To store the distances of intersections

#     # Draw detected edges in red color on the original image
#     img_with_edges = img.copy()
#     img_with_edges[edges > 0] = [0, 0, 255]  # Red color for edges

#     # Iterate to add perpendicular lines
#     for _ in range(num_perpendicular_lines):
#         # Randomly select a point on the skeleton
#         random_point = skeleton_points[random.randint(0, len(skeleton_points) - 1)]

#         # Get the neighbors for directional vector (choosing nearby points)
#         index = np.where((skeleton_points[:, 0] == random_point[0]) & (skeleton_points[:, 1] == random_point[1]))[0][0]
#         prev_point = skeleton_points[index - 1] if index > 0 else random_point
#         next_point = skeleton_points[index + 1] if index < len(skeleton_points) - 1 else random_point

#         # Calculate the direction vector and perpendicular vector
#         direction_vector = (next_point[0] - prev_point[0], next_point[1] - prev_point[1])
#         perpendicular_vector = (-direction_vector[1], direction_vector[0])

#         # Normalize the perpendicular vector
#         length = np.sqrt(perpendicular_vector[0] ** 2 + perpendicular_vector[1] ** 2)
#         if length == 0:  # Avoid division by zero
#             continue
#         perpendicular_vector = (perpendicular_vector[0] / length, perpendicular_vector[1] / length)

#         # Define the length of the perpendicular line
#         perpendicular_length = 30

#         # Calculate the start and end points of the perpendicular line
#         start_perpendicular = (int(random_point[0] - perpendicular_length * perpendicular_vector[0]),
#                                int(random_point[1] - perpendicular_length * perpendicular_vector[1]))

#         end_perpendicular = (int(random_point[0] + perpendicular_length * perpendicular_vector[0]),
#                              int(random_point[1] + perpendicular_length * perpendicular_vector[1]))

#         # Draw the perpendicular line on the image with edges
#         cv2.line(img_with_edges, (start_perpendicular[1], start_perpendicular[0]),
#                  (end_perpendicular[1], end_perpendicular[0]), (0, 255, 0), 2)

#         # Detect intersections of the perpendicular line with edges
#         intersections = []
#         rr, cc = np.linspace(start_perpendicular[0], end_perpendicular[0], num=100).astype(int), \
#                  np.linspace(start_perpendicular[1], end_perpendicular[1], num=100).astype(int)

#         for r, c in zip(rr, cc):
#             if 0 <= r < edges.shape[0] and 0 <= c < edges.shape[1]:
#                 if edges[r, c] > 0:  # Check if the pixel is part of the edge
#                     intersections.append((r, c))

#         # If there are intersections, plot them and calculate distance
#         if len(intersections) >= 2:
#             first_intersection = intersections[0]
#             last_intersection = intersections[-1]

#             # Plot intersection points
#             cv2.circle(img_with_edges, (first_intersection[1], first_intersection[0]), 3, (255, 0, 0), -1)
#             cv2.circle(img_with_edges, (last_intersection[1], last_intersection[0]), 3, (255, 0, 0), -1)

#             # Calculate the distance between intersections
#             distance = np.sqrt((last_intersection[0] - first_intersection[0]) ** 2 +
#                                (last_intersection[1] - first_intersection[1]) ** 2)
#             if distance > 0:
#                 perpendicular_distances.append(round(distance,3))


#     # Return the modified image and the calculated distances (for further use)
#     return img_with_edges, perpendicular_distances


def crack_detection(img):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a blur to the grayscale image
    blurred_img = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Thresholding to get a binary image (inverse binary to highlight cracks)
    _, binary_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Skeletonize the inverted binary image to get a thin representation of the cracks
    skeleton = skeletonize(binary_img // 255).astype(np.uint8) * 255  # Convert boolean to uint8 for OpenCV

    # Detect edges using a custom edge detection function (replace `edge_detection` with your implementation)
    edges = edge_detection(img)

    # Create a list of edge points for distance calculation
    edge_points = np.column_stack(np.where(edges > 0))

    if len(edge_points) == 0:
        return img, []  # If no edges are detected, return the original image and an empty list

    # Get the coordinates of the skeleton points
    skeleton_points = np.column_stack(np.where(skeleton > 0))

    if len(skeleton_points) == 0:
        return img, []  # If no skeleton points are detected, return original image and an empty list

    # Filter skeleton points to keep only those within a 30-pixel range of the edges
    filtered_skeleton_points = []
    threshold_distance = 10

    for sp in skeleton_points:
        distances = np.sqrt(np.sum((edge_points - sp) ** 2, axis=1))
        if np.any(distances <= threshold_distance):
            filtered_skeleton_points.append(sp)

    filtered_skeleton_points = np.array(filtered_skeleton_points)

    # If no skeleton points remain after filtering, return the original image
    if len(filtered_skeleton_points) == 0:
        return img, []

    # Set the number of perpendicular lines to visualize
    num_perpendicular_lines = 20
    perpendicular_distances = []  # To store the distances of intersections

    # Draw detected edges in red color on the original image
    img_with_edges = img.copy()
    img_with_edges[edges > 0] = [0, 0, 255]  # Red color for edges

    # Iterate to add perpendicular lines
    for _ in range(num_perpendicular_lines):
        # Randomly select a point on the filtered skeleton
        random_point = filtered_skeleton_points[random.randint(0, len(filtered_skeleton_points) - 1)]

        # Get the neighbors for directional vector (choosing nearby points)
        index = np.where((filtered_skeleton_points[:, 0] == random_point[0]) & (filtered_skeleton_points[:, 1] == random_point[1]))[0][0]
        prev_point = filtered_skeleton_points[index - 1] if index > 0 else random_point
        next_point = filtered_skeleton_points[index + 1] if index < len(filtered_skeleton_points) - 1 else random_point

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
                perpendicular_distances.append(round(distance, 3))

    # Return the modified image and the calculated distances (for further use)
    return img_with_edges, perpendicular_distances