from PyQt5.QtWidgets import QSlider
import numpy as np
import cv2 as cv
import cv2

from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections.abc import Iterable


def kmeans(image, k, max_iters):
    """
    Perform K-means clustering on an image.

    Parameters:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray: The segmented image.
    """

    # Convert the image to a numpy array and flatten it into a 2D array
    pixels = np.array(image).reshape(-1, 3)

    # Initialize centroids randomly
    centroids = pixels[np.random.choice(pixels.shape[0], size=k, replace=False)]

    for _ in range(max_iters):
        # Compute distances from pixels to centroids
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)

        # Assign each pixel to the closest centroid
        labels = np.argmin(distances, axis=1)

        # Compute new centroids as the mean of the pixels assigned to each centroid
        new_centroids = np.array(
            [
                (
                    pixels[labels == i].mean(axis=0) + 1e-10
                    if np.any(labels == i)
                    else centroids[i]
                )
                for i in range(k)
            ]
        )  # If centroids didn't change, stop the algorithm
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Replace each pixel with its centroid
    segmented_image = centroids[labels].reshape(np.array(image).shape).astype(np.uint8)

    return segmented_image


import cv2
import numpy as np
import PIL.Image
from scipy.spatial import KDTree


def mean_shift_method(
    img,
    window_size=100,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1),
):
    
    """
    This function applies the mean shift algorithm to an image to cluster similar pixels together.

    Args:
        img (numpy.ndarray): The input image as a 3D numpy array.
        window_size (int, optional): The size of the window used for the mean shift algorithm. Defaults to 100.
        criteria (tuple, optional): The termination criteria for the mean shift algorithm. Defaults to (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1).

    Returns:
        numpy.ndarray: The output image with each pixel assigned a unique color corresponding to its cluster.

    Raises:
        ValueError: If the size of the labels array does not match the size of the original image.
    """
    img_to_2dArray = img.reshape(-1, 3)

    num_points, num_features = img_to_2dArray.shape
    point_considered = np.zeros(num_points, dtype=bool)
    labels = -1 * np.ones(num_points, dtype=int)
    label_count = 0

    # Check if the sizes match
    if labels.size != img_to_2dArray.shape[0]:
        raise ValueError(
            "The size of the labels array does not match the size of the original image."
        )

    tree = KDTree(img_to_2dArray)

    for i in range(num_points):
        if point_considered[i]:
            continue

        Center_point = img_to_2dArray[i]
        while True:
            in_window = tree.query_ball_point(Center_point, r=window_size)
            new_center = np.mean(img_to_2dArray[in_window], axis=0)

            if np.linalg.norm(new_center - Center_point) < criteria[1]:
                labels[in_window] = label_count
                point_considered[in_window] = True
                label_count += 1
                break

            Center_point = new_center

    # Generate a unique color for each label
    unique_colors = np.random.randint(0, 255, (label_count, 3))

    # Create a new image where each pixel is assigned the color of its cluster
    new_img = np.zeros_like(img_to_2dArray)
    for i in range(label_count):
        new_img[labels == i] = unique_colors[i]

    output_image = new_img.reshape(img.shape).astype(np.uint8)
    return output_image


import cv2
import numpy as np


def region_growing(img, seed):

    """
    This function performs region growing algorithm on a given image starting from a seed pixel.
    The region growing algorithm is a pixel-based algorithm used for object segmentation.

    Args:
        img (numpy.ndarray): The input image as a 2D numpy array. If it's a color image, it's converted to grayscale.
        seed (tuple): The coordinates of the seed pixel (x, y).

    Returns:
        numpy.ndarray: A binary image where white pixels belong to the region growing from the seed pixel.
    """

    
    # Convert the image to grayscale if it's a color image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # parameters
    height, width = img.shape
    intensity_threshold = 20

    # initialization
    visited = np.zeros_like(img, dtype=bool)
    region = np.zeros_like(img, dtype=bool)
    list_pixels = [seed]

    while len(list_pixels) > 0:
        s = list_pixels.pop()
        x, y = s

        # Check if the pixel is within the image boundaries
        if x < 0 or y < 0 or x >= height or y >= width:
            continue

        # Check if the pixel has been visited
        if visited[x, y]:
            continue

        # Mark the pixel as visited
        visited[x, y] = True

        # Check if the pixel intensity is within the threshold
        if abs(int(img[x, y]) - int(img[seed])) <= intensity_threshold:
            # Add the pixel to the region
            region[x, y] = True

            # Add the neighboring pixels to the list
            list_pixels.append((x - 1, y))
            list_pixels.append((x + 1, y))
            list_pixels.append((x, y - 1))
            list_pixels.append((x, y + 1))

    return region


def highlight_region(img, region):
    # Create a copy of the original image
    img_copy = img.copy()

    # Change the intensity of the selected region in the copy
    img_copy[region] = 150  # Change intensity to 150 (gray)

    return img_copy
