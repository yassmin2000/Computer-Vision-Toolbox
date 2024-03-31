import numpy as np
from PyQt5.QtGui import QImage
import cv2


# def equalization(image):
#     hist = histogram(image)
#     # total number of pixels
#     n = len(image.flatten())
#     pdf = hist / n
#     # cdf = np.zeros(len(pdf))
#     # cdf[0] = pdf[0]
#     # for i in range(1, len(pdf)):
#     #     cdf[i] = pdf[i] + cdf[i - 1]
#     cdf = np.cumsum(pdf)
#     color_levels = np.count_nonzero(hist)
#     equalized_image = np.zeros_like(image)
#     for i in range(len(image)):
#         for j in range(len(image[i])):
#             # equalized_image[i][j] = round(cdf[image[i][j]] * color_levels)
#             pixel_value = image[i][j]
#             equalized_image[i][j] = int(cdf[pixel_value] * (color_levels - 1))

#     return equalized_image

def equalization(image):
    """
    Perform histogram equalization on the input image.
    
    Parameters:
    - image: the input image to be equalized
    
    Returns:
    - equalized_image: the resulting equalized image with pixel values cast to uint8
    """
    # Calculate histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate cumulative distribution function
    cdf = hist.cumsum()

    # Normalize CDF to [0, 1]
    cdf_normalized = cdf / cdf.max()

    # Perform histogram equalization
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized * 255).reshape(image.shape)

    return equalized_image.astype(np.uint8)




# def equalization(image):
#     try:
#         if not isinstance(image, np.ndarray):
#             raise TypeError("Input image must be a NumPy array")
#
#         if len(image.shape) != 2:
#             raise ValueError("Input image must be a grayscale image")
#
#         hist = histogram(image)
#         if hist is None:
#             raise ValueError("Histogram calculation failed")
#
#         n = len(image.flatten())
#         if n == 0:
#             raise ValueError("Input image has zero pixels")
#
#         pdf = hist / n
#         cdf = np.cumsum(pdf)
#         color_levels = np.count_nonzero(hist)
#         if color_levels == 0:
#             raise ValueError("Input image has zero color levels")
#
#         equalized_image = np.zeros_like(image)
#         for i in range(len(image)):
#             for j in range(len(image[i])):
#                 pixel_value = image[i][j]
#                 equalized_image[i][j] = int(cdf[pixel_value] * (color_levels - 1))
#
#         return equalized_image
#
#     except TypeError as te:
#         print("Type error in equalization function:", te)
#     except ValueError as ve:
#         print("Value error in equalization function:", ve)
#     except Exception as e:
#         print("Error in equalization function:", e)


def normalization(image):
    """
    Normalize the input image to the range [0, 255].
    
    Parameters:
    - image: the input image to be normalized
    
    Returns:
    - normalized_image: the resulting normalized image with pixel values cast to uint8
    """
    minimum_value = float(np.min(image))
    maximum_value = float(np.max(image))
    scaled_image = (image - minimum_value) / (maximum_value - minimum_value) * 255.0
    normalized_image = np.clip(scaled_image, 0, 255).astype(np.uint8)
    return normalized_image


# def normalize(img):
#     lmin = float(img.min())
#     lmax = float(img.max())
#     return np.floor((img-lmin)/(lmax-lmin)*225.0)