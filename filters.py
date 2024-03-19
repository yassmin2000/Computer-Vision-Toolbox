import cv2
import numpy as np
import scipy.stats as st


def apply_average_filter(img, kernel_size):
    image_width, image_height = img.shape

    # Develop Averaging filter(3, 3) mask
    mask = np.ones([kernel_size, kernel_size], dtype=int)
    mask = mask / (kernel_size * kernel_size)

    # Convolve the kernel over the image
    img_new = np.zeros_like(img, dtype=int)

    for i in range(1, image_width - 1):
        for j in range(1, image_height - 1):
            temp = (
                img[i - 1, j - 1] * mask[0, 0]
                + img[i - 1, j] * mask[0, 1]
                + img[i - 1, j + 1] * mask[0, 2]
                + img[i, j - 1] * mask[1, 0]
                + img[i, j] * mask[1, 1]
                + img[i, j + 1] * mask[1, 2]
                + img[i + 1, j - 1] * mask[2, 0]
                + img[i + 1, j] * mask[2, 1]
                + img[i + 1, j + 1] * mask[2, 2]
            )

            img_new[i, j] = temp

    img_new = img_new.astype(np.uint8)

    return img_new


# import numpy as np
# import scipy.stats as st

# def apply_gaussian_filter(img, sigma):
#     # Calculate kernel size based on sigma
#     kernel_size = int(6 * sigma) + 1 if int(6 * sigma) % 2 == 1 else int(6 * sigma) + 2
#     image_width, image_height = img.shape

#     # Develop Gaussian filter
#     interval = (2 * sigma + 1.) / kernel_size
#     x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
#     kern1d = np.diff(st.norm.cdf(x))
#     mask = np.outer(kern1d, kern1d)
#     mask = mask / mask.sum()

#     # Convolve the kernel over the image
#     img_new = np.zeros([image_width, image_height])

#     pad_size = kernel_size // 2
#     img_padded = np.pad(img, pad_size, mode='constant')

#     for i in range(image_width):
#         for j in range(image_height):
#             img_patch = img_padded[i:i + kernel_size, j:j + kernel_size]
#             img_new[i, j] = np.sum(img_patch * mask)

#     img_new = img_new.astype(np.uint8)

#     return img_new
def gaussian_kernel(kernel_size, sigma):
    """Generates a Gaussian kernel."""
    # Generate 1D Gaussian kernels
    kx = cv2.getGaussianKernel(kernel_size, sigma)
    ky = cv2.getGaussianKernel(kernel_size, sigma)
    
    # Convolve the 1D Gaussian kernels to create a 2D Gaussian kernel
    gaussian = np.outer(kx, ky.T)

    return gaussian

def apply_gaussian_filter(img, kernel_size=(3, 3), sigma=1):
    # Create a 2D Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Perform convolution with the Gaussian kernel
    img_new = cv2.filter2D(img, -1, kernel)

    return img_new
# def apply_gaussian_filter(img, sigma):
#     kernel_size = 3  # Constant kernel size
#     image_width, image_height= img.shape

#     # Develop Gaussian filter
#     interval = (2 * sigma + 1.) / kernel_size
#     x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
#     kern1d = np.diff(st.norm.cdf(x))
#     mask = np.outer(kern1d, kern1d)
#     mask = mask / mask.sum()

#     # Convolve the kernel over the image
#     img_new = np.zeros([image_width, image_height])

#     pad_size = kernel_size // 2
#     img_padded = np.pad(img, pad_size, mode='constant')

#     for i in range(image_width):
#         for j in range(image_height):
#             img_patch = img_padded[i:i + kernel_size, j:j + kernel_size]
#             img_new[i, j] = np.sum(img_patch * mask)

#     img_new = img_new.astype(np.uint8)

#     return img_new


# import numpy as np
# from scipy import stats as st

# def apply_gaussian_filter(img, sigma):
#     # Calculate kernel size based on sigma
#     kernel_size = int(6 * sigma) + 1 if int(6 * sigma) % 2 == 1 else int(6 * sigma) + 2
#     image_height, image_width = img.shape

#     # Develop Gaussian filter
#     interval = (2 * sigma + 1.) / kernel_size
#     x = np.linspace(-sigma - interval / 2., sigma + interval / 2., kernel_size + 1)
#     kern1d = np.diff(st.norm.cdf(x))
#     mask = np.outer(kern1d, kern1d)
#     mask = mask / mask.sum()

#     # Convolve the kernel over the image
#     img_new = np.zeros([image_height, image_width])

#     pad_size = kernel_size // 2
#     img_padded = np.pad(img, pad_size, mode='constant')

#     for i in range(image_height):
#         for j in range(image_width):
#             img_patch = img_padded[i:i + kernel_size, j:j + kernel_size]
#             img_new[i, j] = np.sum(img_patch * mask)

#     img_new = img_new.astype(np.uint8)

#     return img_new


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


# def apply_median_filter(img):
#     kernel_size=3
#     image_width, image_height,_ = img.shape
#     pad_size = kernel_size // 2
#     img_padded = np.pad(img, pad_size, mode='constant')

#     # Convolve the kernel over the image
#     img_new = np.zeros([image_width, image_height])

#     for i in range(image_width):
#         for j in range(image_height):
#             temp = img_padded[i:i+kernel_size, j:j+kernel_size].flatten()
#             temp = sorted(temp)
#             img_new[i, j] = temp[len(temp) // 2]

#     img_new = img_new.astype(np.uint8)

#     return img_new
