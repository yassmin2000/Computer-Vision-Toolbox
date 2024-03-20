import numpy as np
import cv2

def apply_uniform_noise(image, noise_value):
    # Generate random noise with the same shape as the image
    noise = np.random.uniform(-noise_value, noise_value, image.shape).astype(image.dtype)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    return noisy_image




def apply_gaussian_noise(image, mean, sigma):
    # mean controls intensties to be changed
    # sigma controls the standard deviation of the noise
    
    noise = np.random.normal(mean, sigma, image.shape).astype(image.dtype)

    # Add the noise to the image
    noisy_image = cv2.add(image, noise)

    return noisy_image


def apply_salt_and_pepper_noise(image, w, b):

    # Generate random noise with the same shape as the image
    noise = np.random.uniform(0, 1, image.shape)

    # Add the noise to the image
    noisy_image = image.copy()
    noisy_image[noise < w] = 0
    noisy_image[noise > 1 - b] = 255

    return noisy_image
