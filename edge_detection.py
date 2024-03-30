import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
# from main import sig , tap 

def sobel_operator(image, direction):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

# def canny_operator(image, low_threshold, high_threshold):
#     edges = cv2.Canny(image, low_threshold, high_threshold)
#     return edges

def canny_operator(image, low_threshold, high_threshold, kernel_size=5):
        # Step 1: Gaussian Blur
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smoothed_image = gaussian_blur(gray_image, kernel_size)

        # Step 2: Gradient Calculation
        # gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
        # gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
        # gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        # gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
        gradient_magnitude, gradient_direction = sobel(smoothed_image)

        # Step 3: Non-maximum Suppression
        suppressed_image = np.zeros_like(gradient_magnitude) 
        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                direction = gradient_direction[i, j]
                if (0 <= direction < 22.5) or (157.5 <= direction <= 180) or (-22.5 <= direction < 0) or (-180 <= direction < -157.5):
                    neighbors = [gradient_magnitude[i, j + 1], gradient_magnitude[i, j - 1]]
                elif (22.5 <= direction < 67.5) or (-157.5 <= direction < -112.5):
                    neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
                elif (67.5 <= direction < 112.5) or (-112.5 <= direction < -67.5):
                    neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
                else:
                    neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
                if gradient_magnitude[i, j] >= max(neighbors):
                    suppressed_image[i, j] = gradient_magnitude[i, j]


        # Step 4: Double Thresholding
        strong_edges = (suppressed_image > high_threshold)
        weak_edges = (suppressed_image >= low_threshold) & (suppressed_image <= high_threshold)
       
            
        # Step 5: Edge Tracking by Hysteresis
        final_edges = np.zeros_like(image)
        strong_i, strong_j = np.where(strong_edges)
        final_edges[strong_i, strong_j] = 255

        for i, j in zip(*weak_edges.nonzero()):
            if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                final_edges[i, j] = 255
        
        return final_edges


def gaussian_blur(image, kernel_size):
    
    kernel = create_gaussian_kernel(kernel_size)    



    blurred_image = np.zeros_like(image, dtype=np.float64)
    offset = kernel_size // 2

    for i in range(offset, image.shape[0] - offset):
        for j in range(offset, image.shape[1] - offset):
            blurred_image[i, j] = np.sum(kernel * image[i - offset:i + offset + 1, j - offset:j + offset + 1])

    return blurred_image.astype(np.uint8)

def create_gaussian_kernel(size, sigma=1):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    return kernel / np.sum(kernel)

def sobel(image):
    kernel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    kernel_horizontal_result = convolve2D(image, kernel_horizontal)
    kernel_vertical_result = convolve2D(image, kernel_vertical)
    gradient_magnitude = np.sqrt(kernel_horizontal_result**2 + kernel_vertical_result**2)
    gradient_direction = np.arctan2(kernel_vertical_result, kernel_horizontal_result)* (180 / np.pi)
    return gradient_magnitude, gradient_direction

def convolve2D(image, kernel, padding=0, strides=1):
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
