import cv2
import numpy as np
import scipy.stats as st
from scipy import stats as st


def apply_average_filter(img, kernel_size):
    image_width, image_height = img.shape
    img_new = np.zeros_like(img, dtype=np.float64)

    # Ensure kernel size is odd and centered around the current pixel
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Calculate padding size for boundary handling
    pad = kernel_size // 2

    # Develop Averaging filter mask
    mask = np.ones([kernel_size, kernel_size], dtype=float)
    mask /= kernel_size * kernel_size

    # Pad the image
    img_padded = np.pad(img, pad, mode="constant")

    # Convolve the kernel over the image
    for i in range(pad, image_width + pad):
        for j in range(pad, image_height + pad):
            img_new[i - pad, j - pad] = np.sum(
                img_padded[i - pad : i + pad + 1, j - pad : j + pad + 1] * mask
            )

    return img_new.astype(img.dtype)




# def gaussian_kernel(kernel_size, sigma):
#     """Generates a Gaussian kernel."""
#     # Generate 1D Gaussian kernels
#     kx = cv2.getGaussianKernel(kernel_size, sigma)
#     ky = cv2.getGaussianKernel(kernel_size, sigma)

#     # Convolve the 1D Gaussian kernels to create a 2D Gaussian kernel
#     gaussian = np.outer(kx, ky.T)

#     return gaussian

# def apply_gaussian_filter(img, kernel_size=(3, 3), sigma=1):
#     # Create a 2D Gaussian kernel
#     kernel = gaussian_kernel(kernel_size, sigma)

#     # Perform convolution with the Gaussian kernel
#     img_new = cv2.filter2D(img, -1, kernel)

#     return img_new


def apply_gaussian_filter(img, sigma):
    kernel_size = 3  # Constant kernel size
    image_width, image_height= img.shape

    # Develop Gaussian filter
    interval = (2 * sigma + 1.) / kernel_size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    mask = np.outer(kern1d, kern1d)
    mask = mask / mask.sum()

    # Convolve the kernel over the image
    img_new = np.zeros([image_width, image_height])

    pad_size = kernel_size // 2
    img_padded = np.pad(img, pad_size, mode='constant')

    for i in range(image_width):
        for j in range(image_height):
            img_patch = img_padded[i:i + kernel_size, j:j + kernel_size]
            img_new[i, j] = np.sum(img_patch * mask)

    img_new = img_new.astype(np.uint8)

    return img_new

# def apply_gaussian_filter(image, sigma):
#     # Define the kernel size
#     kernel_size = 3

#     # Create the Gaussian kernel
#     kernel = cv2.getGaussianKernel(kernel_size, sigma)

#     # Perform convolution
#     filtered_image = cv2.filter2D(image, -1, kernel)

#     return filtered_image


# def gaussian_kernel(kernel_size=(3, 3), sigma=1):
 
#     # Ensure kernel size is odd
#     if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
#         raise ValueError("Kernel size must be odd")

#     # Calculate center of the kernel
#     center_x = kernel_size[0] // 2
#     center_y = kernel_size[1] // 2

#     # Generate grid of indices
#     x = np.arange(-center_x, center_x + 1)
#     y = np.arange(-center_y, center_y + 1)
#     xx, yy = np.meshgrid(x, y)

#     # Calculate Gaussian kernel values
#     kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma **2))
#     kernel /= np.sum(kernel)
#     return kernel



def apply_median_filter(img):
    kernel_size = 3
    image_width, image_height = img.shape

    # Convolve the kernel over the image
    img_new = np.zeros([image_width, image_height])

    for i in range(1, image_width - 1):
        for j in range(1, image_height - 1):
            temp = [
                img[i - 1, j - 1],
                img[i - 1, j],
                img[i - 1, j + 1],
                img[i, j - 1],
                img[i, j],
                img[i, j + 1],
                img[i + 1, j - 1],
                img[i + 1, j],
                img[i + 1, j + 1],
            ]
            temp = sorted(temp)
            img_new[i, j] = temp[4]

    img_new = img_new.astype(np.uint8)

    return img_new


