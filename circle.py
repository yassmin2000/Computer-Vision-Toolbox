import cv2 as cv
import numpy as np
from collections import defaultdict

def calculate_accumulator(img_height, img_width, edge_image, circle_candidates):
    """
    Calculate the Hough accumulator.
    
    Parameters:
        img_height (int): Height of the image.
        img_width (int): Width of the image.
        edge_image (numpy.ndarray): Edge-detected image.
        circle_candidates (list): List of circle candidates.
    
    Returns:
        defaultdict: Hough accumulator containing votes for circle candidates.
    """
    accumulator = defaultdict(int)

    # Step 1: Iterate through each pixel in the edge-detected image
    for y in range(img_height):
        for x in range(img_width):
            if np.any(edge_image[y, x] > 0):  # Check if the pixel is an edge pixel
                # Step 2: Vote for each circle candidate passing through the edge pixel
                for r, r_cos_t, r_sin_t in circle_candidates:
                    x_center = x - r_cos_t
                    y_center = y - r_sin_t
                    accumulator[(x_center, y_center, r)] += 1  # Vote for the current candidate

    return accumulator


def candidate_circles(thetas, radius, num_thetas):
    """
    Generate candidate circles based on radius and theta values.
    
    Parameters:
        thetas (numpy.ndarray): Array of theta values.
        radii (numpy.ndarray): Array of radius values.
        num_thetas (int): Number of theta values.
    
    Returns:
        list: List of circle candidates.
    """
    # parametric equation of circle
    # x = x_center + r * cos(t) and y = y_center + r * sin(t),  
    # (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    circle_candidates = []

    # Step 3: Generate circle candidates for each radius and theta
    for r in radius:
        for t in range(num_thetas):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

    return circle_candidates


def post_process_circles(output_circles, pixel_threshold):
    """
    Post-process detected circles to remove duplicates and close circles.
    
    Parameters:
        output_circles (list): List of detected circles.
        pixel_threshold (int): Threshold for pixel comparison.
    
    Returns:
        list: List of post-processed circles.
    """
    processed_circles = []

    # Step 5: Perform post-processing to filter out redundant circles
    for x, y, r, v in output_circles:
        # Exclude circles that are too close to each other
        # Remove nearby duplicate circles based on pixel_threshold
        if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for xc, yc, rc, v in processed_circles):
            processed_circles.append((x, y, r, v))
    return processed_circles


def hough_circles(circle_color, edge_image, r_min: int = 20, r_max: int = 100, delta_r: int = 1, num_thetas: int = 100, bin_threshold: float = 0.4, pixel_threshold: int = 20, post_process: bool = True):
    """
    Detect circles using Hough transform.
    
    Parameters:
        circle_color: Color of the detected circles.
        edge_image (numpy.ndarray): Edge-detected input image.
        r_min (int): Minimum radius of the circles.
        r_max (int): Maximum radius of the circles.
        delta_r (int): Step size for radius.
        num_thetas (int): Number of theta values.
        bin_threshold (float): Threshold for bin voting.
        pixel_threshold (int): Threshold for pixel comparison.
        post_process (bool): Flag to enable/disable post-processing.
    
    Returns:
        numpy.ndarray: Output image with detected circles drawn.
    """
    if edge_image is None:
        print("Error in input image!")
        return

    img_height, img_width = edge_image.shape[:2]
    # R and Theta ranges
    ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
    ## Radius is the range of radius from r_min to r_max with increment of delta_r      
    dtheta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=dtheta)
    radius = np.arange(r_min, r_max, step=delta_r)
    circle_candidates = candidate_circles(thetas, radius, num_thetas)
    accumulator = calculate_accumulator(img_height, img_width, edge_image, circle_candidates)
    output_img = edge_image.copy()
    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 

    out_circles = []

    # Step 4: Find circle candidates with sufficient votes
    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold:
            out_circles.append((x, y, r, current_vote_percentage))

    if post_process:
        # Step 6: Perform post-processing on detected circles
        out_circles = post_process_circles(out_circles, pixel_threshold)

    # Step 7: Draw detected circles on the output image
    for x, y, r, v in out_circles:
        output_img = cv.circle(output_img, (x, y), r, (255, 0, 0), 2)

    return output_img
