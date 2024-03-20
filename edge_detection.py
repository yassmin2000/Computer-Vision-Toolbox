import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

def sobel_operator(image, direction):
    kernel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    if direction == "Horizontal":
        kernel = kernel_horizontal
    elif direction == "Vertical":
        kernel = kernel_vertical
    elif direction == "Both":
        kernel_horizontal = convolve2D(image, kernel_horizontal)
        kernel_vertical = convolve2D(image, kernel_vertical)
        edges = np.sqrt(kernel_horizontal**2 + kernel_vertical**2)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype('uint8')
        return edges
    else:
        raise ValueError("Invalid direction. Use 'Horizontal', 'Vertical', or 'Both'.")

    result = convolve2D(image, kernel)
    # result = np.abs(result)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype('uint8')

    return result

def roberts_operator(image, direction):
    kernel_diagonal1 = np.array([[1, 0], [0, -1]])
    kernel_diagonal2 = np.array([[0, 1], [-1, 0]])

    if direction == "Horizontal":
        kernel = kernel_diagonal1
    elif direction == "Vertical":
        kernel = kernel_diagonal2
    elif direction == "Both":
        kernel_horizontal = convolve2D(image, kernel_diagonal1)
        kernel_vertical = convolve2D(image, kernel_diagonal2)
        edges = np.sqrt(kernel_horizontal**2 + kernel_vertical**2)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype('uint8')
        return edges
    else:
        raise ValueError("Invalid direction. Use 'Horizontal', 'Vertical', or 'Both'.")

    result = convolve2D(image, kernel)
    # result = np.abs(result)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype('uint8')

    return result

def prewitt_operator(image, direction):
    kernel_horizontal = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    kernel_vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    if direction == "Horizontal":
        kernel = kernel_horizontal
    elif direction == "Vertical":
        kernel = kernel_vertical
    elif direction == "Both":
        kernel_horizontal = convolve2D(image, kernel_horizontal)
        kernel_vertical = convolve2D(image, kernel_vertical)
        edges = np.sqrt(kernel_horizontal**2 + kernel_vertical**2)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = edges.astype('uint8')
        return edges
    else:
        raise ValueError("Invalid direction. Use 'Horizontal', 'Vertical', or 'Both'.")

    result = convolve2D(image, kernel)
    # result = np.abs(result)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype('uint8')

    return result

def canny_operator(image, low_threshold, high_threshold):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

# def apply_gaussian_blur(image, kernel_size, sigma):
#     return cv2.GaussianBlur(image, kernel_size, sigma)

# def calculate_gradients(image):
#     gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
#     gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
#     return gradient_x, gradient_y

# def calculate_gradient_magnitude_and_direction(gradient_x, gradient_y):
#     gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
#     gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
#     return gradient_magnitude, gradient_direction

# def non_maximum_suppression(gradient_magnitude, gradient_direction):
    # suppressed_image = np.zeros_like(gradient_magnitude)
    # for i in range(1, gradient_magnitude.shape[0] - 1):
    #     for j in range(1, gradient_magnitude.shape[1] - 1):
    #         angle = gradient_direction[i, j]
    #         neighbors = []

    #         if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
    #             neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
    #         elif 22.5 <= angle < 67.5:
    #             neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
    #         elif 67.5 <= angle < 112.5:
    #             neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
    #         elif 112.5 <= angle < 157.5:
    #             neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]

    #         if neighbors and gradient_magnitude[i, j] >= max(neighbors):
    #             suppressed_image[i, j] = gradient_magnitude[i, j]

    # return suppressed_image

# def edge_tracking_by_hysteresis(suppressed_image, high_threshold, low_threshold):
    # edges = np.zeros_like(suppressed_image)
    # strong_edges_i, strong_edges_j = np.where(suppressed_image >= high_threshold)
    # weak_edges_i, weak_edges_j = np.where((suppressed_image >= low_threshold) & (suppressed_image < high_threshold))
    # edges[strong_edges_i, strong_edges_j] = 255
    # edges[weak_edges_i, weak_edges_j] = 50  # Weak edges

    # for i in range(1, edges.shape[0] - 1):
    #     for j in range(1, edges.shape[1] - 1):
    #         if edges[i, j] == 50 and np.max(edges[i-1:i+2, j-1:j+2]) == 255:
    #             edges[i, j] = 255  # Promote weak edges to strong edges

    # return edges.astype('uint8')
def convolve2D(image, kernel, padding=0, strides=1):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        for y in range(image.shape[1]):
            # Exit Convolution
            if y > image.shape[1] - yKernShape:
                break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
                    if x > image.shape[0] - xKernShape:
                        break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * image[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return output
