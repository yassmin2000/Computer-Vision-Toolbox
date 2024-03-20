import numpy as np
import cv2





def apply_uniform_noise(image, high=255*0.2):
    # high is the maximum value of the uniform distribution
   
    if len(image.shape) == 2:  # Check if the image is grayscale
        row, col = image.shape
        noise = np.random.uniform(0, high, (row, col))  # Generate random noise from uniform distribution
    else:  # Image is RGB/BGR
        row, col, ch = image.shape
        noise = np.random.uniform(0, high, (row, col, ch))  # Generate random noise from uniform distribution

    noisy = image + noise

    # Clip values to [0, 255] and convert to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy



def apply_gaussian_noise(image, mean=0, sigma=25):
    # mean is the mean of the Gaussian distribution
    # sigma is the standard deviation of the Gaussian distribution
   
    if len(image.shape) == 2:  # Check if the image is grayscale
        row, col = image.shape
        gauss = np.random.normal(mean, sigma, (row, col))
        noisy = image + gauss  # Scale the noise intensity
    else:  # Image is RGB/BGR
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = image + gauss  # Scale the noise intensity

    # Clip values to [0, 255] and convert to uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy





def apply_salt_and_pepper_noise(image, w=0.01, b=0.01):
    # w is the probability of salt noise
    # b is the probability of pepper noise
    
    
    noisy_image = np.copy(image)

    # Add salt noise
    salt_mask = np.random.rand(*image.shape) < w
    noisy_image[salt_mask] = 255

    # Add pepper noise
    pepper_mask = np.random.rand(*image.shape) < b
    noisy_image[pepper_mask] = 0

    return noisy_image