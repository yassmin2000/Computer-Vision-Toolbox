import numpy as np
from PyQt5.QtGui import QImage

from histogram import histogram
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
    minimum_value = np.min(image)
    maximum_value = np.max(image)
    normalized_image = (image - minimum_value)*(255 / (maximum_value - minimum_value))
    return normalized_image.astype(np.uint8)