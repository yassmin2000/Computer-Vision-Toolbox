import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
# from main import sig , tap 

def sobel_operator(image, direction):
    
    """
    Apply Sobel operator to an image to detect edges in the specified direction.

    Parameters:
    - image: The input image to apply the Sobel operator.
    - direction: The direction in which to apply the Sobel operator. Can be 'Horizontal', 'Vertical', or 'Both'.

    The function performs the following steps:
    1. Converts the input image to grayscale.
    2. Defines Sobel kernels for horizontal and vertical edge detection.
    3. Based on the specified direction, selects the corresponding kernel ('Horizontal', 'Vertical') or applies both to detect edges in both directions.
    4. If the direction is 'Both', computes the edge magnitude using the horizontal and vertical convolutions, normalizes the result, and returns the edges.
    5. If the direction is 'Horizontal' or 'Vertical', applies the selected kernel to the image using convolution, normalizes the result, and returns the edges.
    6. Raises a ValueError for an invalid direction input.

    Note: This function assumes the presence of the `convolve2D` function for performing 2D convolutions and uses OpenCV (cv2) for image processing operations.
    """
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
    """
    Apply Roberts operator to an image to detect edges in the specified direction.

    Parameters:
    - image: The input image to apply the Roberts operator.
    - direction: The direction in which to apply the Roberts operator. Can be 'Horizontal', 'Vertical', or 'Both'.

    The function performs the following steps:
    1. Converts the input image to grayscale.
    2. Defines Roberts kernels for diagonal edge detection.
    3. Based on the specified direction, selects the corresponding kernel ('Horizontal', 'Vertical') or applies both to detect edges in both directions.
    4. If the direction is 'Both', computes the edge magnitude using the diagonal convolutions, normalizes the result, and returns the edges.
    5. If the direction is 'Horizontal' or 'Vertical', applies the selected kernel to the image using convolution, normalizes the result, and returns the edges.
    6. Raises a ValueError for an invalid direction input.

    Note: This function assumes the presence of the `convolve2D` function for performing 2D convolutions and uses OpenCV (cv2) for image processing operations.
    """
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
    """
    Apply Prewitt operator to an image to detect edges in the specified direction.

    Parameters:
    - image: The input image to apply the Prewitt operator.
    - direction: The direction in which to apply the Prewitt operator. Can be 'Horizontal', 'Vertical', or 'Both'.

    The function performs the following steps:
    1. Converts the input image to grayscale.
    2. Defines Prewitt kernels for horizontal and vertical edge detection.
    3. Based on the specified direction, selects the corresponding kernel ('Horizontal', 'Vertical') or applies both to detect edges in both directions.
    4. If the direction is 'Both', computes the edge magnitude using the horizontal and vertical convolutions, normalizes the result, and returns the edges.
    5. If the direction is 'Horizontal' or 'Vertical', applies the selected kernel to the image using convolution, normalizes the result, and returns the edges.
    6. Raises a ValueError for an invalid direction input.

    Note: This function assumes the presence of the `convolve2D` function for performing 2D convolutions and uses OpenCV (cv2) for image processing operations.
    """
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
        """
    Apply the Canny edge detection algorithm to an image.

    Parameters:
    - image: The input image to apply the Canny operator.
    - low_threshold: The lower threshold for hysteresis procedure.
    - high_threshold: The higher threshold for hysteresis procedure.
    - kernel_size: The size of the Gaussian kernel used for smoothing. Default is 5.

    The function performs the following steps:
    1. Apply Gaussian Blur to the input image using a Gaussian kernel of the specified size.
    2. Calculate the gradient magnitude and direction of the smoothed image using the Sobel operator.
    3. Perform Non-maximum Suppression by comparing the gradient magnitude with its neighbors in different directions.
    4. Double Thresholding by classifying pixels as strong or weak edges based on the high and low thresholds.
    5. Edge Tracking by Hysteresis by iteratively tracking edges from weak edges to strong edges based on thresholds.

    Returns:
    - final_edges: The resulting edge image after applying Canny operator.

    Note: This function assumes the presence of the `gaussian_blur` and `sobel` functions for performing Gaussian blur and gradient calculation respectively.
        """
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
    """
    Apply Gaussian blur to the input image using a Gaussian kernel of the specified size.

    Parameters:
    - image: The input image to apply Gaussian blur.
    - kernel_size: The size of the Gaussian kernel for blurring.

    The function performs the following steps:
    1. Creates a Gaussian kernel of the specified size.
    2. Computes the blurred image by convolving the input image with the Gaussian kernel.

    Returns:
    - blurred_image: The resulting image after applying Gaussian blur.

    Note: This function assumes the presence of the `create_gaussian_kernel` function for generating the Gaussian kernel.
    """
    kernel = create_gaussian_kernel(kernel_size)    



    blurred_image = np.zeros_like(image, dtype=np.float64)
    offset = kernel_size // 2

    for i in range(offset, image.shape[0] - offset):
        for j in range(offset, image.shape[1] - offset):
            blurred_image[i, j] = np.sum(kernel * image[i - offset:i + offset + 1, j - offset:j + offset + 1])

    return blurred_image.astype(np.uint8)


def create_gaussian_kernel(size, sigma=1):
    """
    Create a Gaussian kernel of a given size and standard deviation (sigma).
    
    Parameters:
    - size: the size of the kernel (height and width)
    - sigma: the standard deviation of the Gaussian distribution
    
    Returns:
    - kernel: a 2D numpy array representing the Gaussian kernel
    """
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)), (size, size))
    return kernel / np.sum(kernel)

def sobel(image):

    """
    Apply the Sobel edge detection algorithm to an image.
    
    Parameters:
    - image: the input image to be processed
    
    Returns:
    - gradient_magnitude: a numpy array representing the magnitude of the gradients in the image
    - gradient_direction: a numpy array representing the direction of the gradients in the image
    """
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
