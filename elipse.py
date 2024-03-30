import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from edge_detection import canny_operator
from collections import defaultdict

# def calculate_accumulator(img_height, img_width, edge_image, ellipse_candidates):
#     """
#     Calculate the Hough accumulator for detecting ellipses.
    
#     Parameters:
#         img_height (int): Height of the image.
#         img_width (int): Width of the image.
#         edge_image (numpy.ndarray): Edge-detected image.
#         ellipse_candidates (list): List of candidate ellipses.
    
#     Returns:
#         defaultdict: Hough accumulator containing votes for ellipse candidates.
#     """
#     accumulator = defaultdict(int)
    
#     # Step 1: Iterate through each pixel in the edge-detected image
#     for y in range(img_height):
#         for x in range(img_width):
#             if np.any(edge_image[y, x] > 0):
#                 # Step 2: Vote for each ellipse candidate passing through the edge pixel
#                 for a, b, x_center, y_center in ellipse_candidates:
#                     for theta in range(0, 180):
#                         x_ellipse = int(x_center + a * np.cos(np.deg2rad(theta)))
#                         y_ellipse = int(y_center + b * np.sin(np.deg2rad(theta)))
#                         if 0 <= x_ellipse < img_width and 0 <= y_ellipse < img_height:
#                             accumulator[(x_ellipse, y_ellipse, a, b)] += 1
#     return accumulator


# def candidate_ellipses(a_values, b_values, thetas):
#     """
#     Generate candidate ellipses based on specified parameters.
    
#     Parameters:
#         a_values (numpy.ndarray): Array of 'a' values (semi-major axis).
#         b_values (numpy.ndarray): Array of 'b' values (semi-minor axis).
#         thetas (numpy.ndarray): Array of theta values.
    
#     Returns:
#         list: List of candidate ellipses.
#     """
#     ellipse_candidates = []
#     for a in a_values:
#         for b in b_values:
#             for theta in thetas:
#                 x_center = 0  # Adjust based on application if needed
#                 y_center = 0  # Adjust based on application if needed
#                 ellipse_candidates.append((a, b, x_center, y_center))
#     return ellipse_candidates


# def post_process(out_ellipses, pixel_threshold):
#     """
#     Perform post-processing to filter out redundant ellipses.
    
#     Parameters:
#         out_ellipses (list): List of detected ellipses.
#         pixel_threshold (int): Threshold for pixel comparison.
    
#     Returns:
#         list: List of filtered ellipses after post-processing.
#     """
#     out_ellipses.sort(key=lambda x: x[-1], reverse=True)
#     final_ellipses = []
#     processed_indices = set()
    
#     # Step 5: Perform post-processing to filter out redundant ellipses
#     for i, ellipse1 in enumerate(out_ellipses):
#         if i in processed_indices:
#             continue
        
#         final_ellipses.append(ellipse1)
#         x1, y1, _, _, _, _ = ellipse1
#         center1 = np.array([x1, y1])
        
#         for j, ellipse2 in enumerate(out_ellipses[i+1:], start=i+1):
#             if j in processed_indices:
#                 continue
            
#             x2, y2, _, _, _, _ = ellipse2
#             center2 = np.array([x2, y2])
#             distance = np.linalg.norm(center1 - center2)
            
#             if distance < pixel_threshold:
#                 processed_indices.add(j)
    
#     return final_ellipses


# def hough_ellipses(img_edges, a_min=20, a_max=100, b_min=20, b_max=100,
#                    delta_a=1, delta_b=1, num_thetas=100, bin_threshold=0.4,
#                    min_edge_threshold=100, max_edge_threshold=200,
#                    pixel_threshold=20, post_process=True):
#     """
#     Detect ellipses using Hough transform.
    
#     Parameters:
#         img_edges (numpy.ndarray): Edge-detected input image.
#         a_min (int): Minimum value for 'a' (semi-major axis) of ellipses.
#         a_max (int): Maximum value for 'a' (semi-major axis) of ellipses.
#         b_min (int): Minimum value for 'b' (semi-minor axis) of ellipses.
#         b_max (int): Maximum value for 'b' (semi-minor axis) of ellipses.
#         delta_a (int): Step size for 'a' values.
#         delta_b (int): Step size for 'b' values.
#         num_thetas (int): Number of theta values.
#         bin_threshold (float): Threshold for bin voting.
#         min_edge_threshold (int): Minimum edge threshold for Canny edge detection.
#         max_edge_threshold (int): Maximum edge threshold for Canny edge detection.
#         pixel_threshold (int): Threshold for pixel comparison in post-processing.
#         post_process (bool): Flag to enable/disable post-processing.
    
#     Returns:
#         numpy.ndarray: Output image with detected ellipses drawn.
#     """
#     img_height, img_width = img_edges.shape[:2]
#     dtheta = int(360 / num_thetas)
#     thetas = np.arange(0, 360, step=dtheta)
#     a_values = np.arange(a_min, a_max, step=delta_a)
#     b_values = np.arange(b_min, b_max, step=delta_b)
    
#     # Step 3: Generate candidate ellipses
#     ellipse_candidates = candidate_ellipses(a_values, b_values, thetas)
#     accumulator = calculate_accumulator(img_height, img_width, img_edges, ellipse_candidates)
    
#     output_img = img_edges.copy()
#     out_ellipses = []
    
#     # Step 4: Find ellipses with sufficient votes
#     for y in range(img_height):
#         for x in range(img_width):
#             for idx in range(len(ellipse_candidates)):
#                 if accumulator[y, x, idx] >= bin_threshold * num_thetas:
#                     a, b, theta = ellipse_candidates[idx]
#                     out_ellipses.append((x, y, a, b, theta, accumulator[y, x, idx] / num_thetas))
    
#     if post_process:
#         # Step 6: Perform post-processing on detected ellipses
#         out_ellipses = post_process(out_ellipses, pixel_threshold)
    
#     # Step 7: Draw detected ellipses on the output image
#     for x, y, a, b, theta, v in out_ellipses:
#         cv2.ellipse(output_img, (x, y), (a, b), np.rad2deg(theta), 0, 360, (255, 0, 0), 2)
    
#     return output_img






def find_contours(edges_image):
    """
    Find contours in the edge-detected image using depth-first search (DFS).
    
    Parameters:
        edges_image (numpy.ndarray): Edge-detected image.
    
    Returns:
        list: List of detected contours, where each contour is represented as a list of points.
    """
    detected_contours = []
    height, width, _ = edges_image.shape
    visited = np.zeros((height, width), dtype=bool)
    
    # Step 1: Iterate through each pixel in the edge-detected image
    for y in range(height):
        for x in range(width):
            # Step 2: Check if the pixel intensity indicates an edge pixel and it's not visited yet
            if np.any(edges_image[y, x] > 0) and not visited[y, x]:
                contour = []
                stack = [(x, y)]
                
                # Step 3: Perform depth-first search to trace the contour
                while stack:
                    current_x, current_y = stack.pop()
                    if 0 <= current_x < width and 0 <= current_y < height and np.any(edges_image[current_y, current_x] > 0) and not visited[current_y, current_x]:
                        visited[current_y, current_x] = True
                        contour.append((current_x, current_y))
                        stack.extend([(current_x + 1, current_y), (current_x - 1, current_y), (current_x, current_y + 1), (current_x, current_y - 1)])
                
                # Step 4: Add the detected contour to the list
                if contour:
                    detected_contours.append(contour)
    
    return detected_contours


def draw_contours(image, detected_contours, color, thickness):
    """
    Draw detected contours on the input image.
    
    Parameters:
        image (numpy.ndarray): Input image.
        detected_contours (list): List of detected contours.
        color (tuple): Color of the contours in BGR format.
        thickness (int): Thickness of the contour lines.
    
    Returns:
        numpy.ndarray: Image with contours drawn.
    """
    # Step 5: Draw the detected contours on the input image
    for contour in detected_contours:
        for i in range(len(contour) - 1):
            cv2.line(image, contour[i], contour[i + 1], color, thickness)
    
    return image


def edge_ellipse_detector(input_image, thickness, contour_color):
    """
    Detect edges and draw contours on the input image.
    
    Parameters:
        input_image (numpy.ndarray): Input image.
        thickness (int): Thickness of the contour lines.
        contour_color (str): Color of the contours. Options: 'Red', 'Blue', 'Green'.
    
    Returns:
        numpy.ndarray: Image with contours drawn.
    """
    # Step 6: Resize the input image to a fixed size 
    resized_image = cv2.resize(input_image, (512, 512))
    
    # Step 7: Apply Canny edge detection
    edge_detected_image = canny_operator(resized_image, 100, 200)
    
    # Step 8: Find contours in the edge-detected image
    detected_contours = find_contours(edge_detected_image)
    
    # Step 9: Draw contours on the image based on the contour color
    if contour_color == 'Red':
        draw_contours(resized_image, detected_contours, (255, 0, 0), thickness)
    
    return resized_image
