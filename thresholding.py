import cv2
import numpy as np
import matplotlib.pyplot as plt
# from pip install scikit-image
# import imread, imsave

def global_thresholding(image, t):
    """
    Apply global thresholding on the input image to create a binary image.
    
    Parameters:
    - image: the input image to be thresholded
    - t: the threshold value
    
    Returns:
    - final_img: the binary image after thresholding
    """
    # If the image is colored, change it to grayscale, otherwise take the image as it is
    if (image.ndim == 3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif (image.ndim == 2):
        gray = image

    # Applying the threshold on the image whether it is calculated or given by the user according to the previous condition
    final_img = gray.copy()
    final_img[gray > t] = 255
    final_img[gray < t] = 0

    return final_img

def local_thresholding(image, t1, t2, t3, t4):
    """
    Apply local thresholding on the input image to create a binary image.
    
    Parameters:
    - image: the input image to be thresholded
    - t1: the threshold value for the top-left section of the image
    - t2: the threshold value for the top-right section of the image
    - t3: the threshold value for the bottom-left section of the image
    - t4: the threshold value for the bottom-right section of the image
    
    Returns:
    - final_img: the binary image after thresholding
    """
    # If the image is colored, change it to grayscale, otherwise take the image as it is
    if (image.ndim == 3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif (image.ndim == 2):
        gray = image

    height, width = gray.shape # get the height and width of the image
    # In this case we will divide the image into a 2x2 grid image
    half_height = height//2 
    half_width = width//2

    # Getting the four section of the 2x2 image
    section_1 = gray[:half_height, :half_width]
    section_2 = gray[:half_height, half_width:]
    section_3 = gray[half_height:, :half_width]
    section_4 = gray[half_height:, half_width:]


    # Applying the threshold of each section on its corresponding section
    section_1[section_1 > t1] = 255
    section_1[section_1 < t1] = 0

    section_2[section_2 > t2] = 255
    section_2[section_2 < t2] = 0

    section_3[section_3 > t3] = 255
    section_3[section_3 < t3] = 0

    section_4[section_4 > t4] = 255
    section_4[section_4 < t4] = 0

    # Regroup the sections to form the final image
    top_section = np.concatenate((section_1, section_2), axis = 1)
    bottom_section = np.concatenate((section_3, section_4), axis = 1)
    final_img = np.concatenate((top_section, bottom_section), axis=0)

        # final_img = gray.copy()
        # final_img[gray > t] = 255
        # final_img[gray < t] = 0

    return final_img