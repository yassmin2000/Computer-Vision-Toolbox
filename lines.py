import numpy as np
import cv2
from edge_detection import canny_operator
def hough_line_detection(image, threshold, low_threshold, high_threshold):
    """
    Detect lines in an image using the Hough Transform.
    
    Parameters:
        image (numpy.ndarray): Input image.
        threshold (int): Threshold value for line detection.
        low_threshold (int): Lower threshold for Canny edge detection.
        high_threshold (int): Upper threshold for Canny edge detection.
    
    Returns:
        tuple: Tuple containing detected lines and the Hough space.
    """
    # Step 1: Define the Hough space
    height, width, _ = image.shape
    diagonal_length = int(np.sqrt(height ** 2 + width ** 2))
    hough_space = np.zeros((2 * diagonal_length, 180), dtype=np.uint8)

    # Step 2: Detect edges using Canny edge detection
    edges = canny_operator(image, low_threshold, high_threshold)
    # cv2.imwrite("output/hough_line_detected.jpg", edges)

    # Step 3: Accumulate votes in the Hough space
    for y in range(height):
        for x in range(width):
            # Check if any element in the edges image indicates an edge pixel
            if np.any(edges[y, x] > 0):
                for theta in range(0, 180):
                    # Compute rho value for the current edge pixel and theta
                    rho = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))
                    # Increment the corresponding bin in the Hough space
                    hough_space[rho + diagonal_length, theta] += 1

    # Step 4: Find peaks in the Hough space
    lines = []
    for rho_idx in range(hough_space.shape[0]):
        for theta_idx in range(hough_space.shape[1]):
            # Check if the Hough space value exceeds the threshold
            if hough_space[rho_idx, theta_idx] > threshold:
                # Compute rho and theta values for the detected line
                rho = rho_idx - diagonal_length
                theta = np.deg2rad(theta_idx)
                # Store the detected line parameters
                lines.append((rho, theta))

    return lines, hough_space

