import cv2
import numpy as np

def local_thresholding(image,threshold_mode,threshold_type):
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

    if threshold_type == 0:
        thresholds = optimal_thresholding(image,threshold_mode)
        t1 = thresholds[0]
        t2 = thresholds[1]
        t3 = thresholds[2]
        t4 = thresholds[3]
        # Applying the threshold of each section on its corresponding section
        section_1[section_1 > t1] = 255
        section_1[section_1 < t1] = 0

        section_2[section_2 > t2] = 255
        section_2[section_2 < t2] = 0

        section_3[section_3 > t3] = 255
        section_3[section_3 < t3] = 0

        section_4[section_4 > t4] = 255
        section_4[section_4 < t4] = 0

    elif threshold_type == 1:
        t1 = otsu_threshold(section_1)
        t2 = otsu_threshold(section_2)
        t3 = otsu_threshold(section_3)
        t4 = otsu_threshold(section_4)
        # Applying the threshold of each section on its corresponding section
        section_1[section_1 > t1] = 255
        section_1[section_1 < t1] = 0

        section_2[section_2 > t2] = 255
        section_2[section_2 < t2] = 0

        section_3[section_3 > t3] = 255
        section_3[section_3 < t3] = 0

        section_4[section_4 > t4] = 255
        section_4[section_4 < t4] = 0

    else:
        section_1 = spectral_threshold(section_1)
        section_2 = spectral_threshold(section_2)
        section_3 = spectral_threshold(section_3)
        section_4 = spectral_threshold(section_4)

    # Regroup the sections to form the final image
    top_section = np.concatenate((section_1, section_2), axis = 1)
    bottom_section = np.concatenate((section_3, section_4), axis = 1)
    final_img = np.concatenate((top_section, bottom_section), axis=0)

    return final_img


def optimal_thresholding(image, threshold_mode):
    """
    Apply optimal thresholding on the input image to create a binary image.

    Parameters:
    - image: the input image to be thresholded
    - threshold_mode : Global or Local threshold

    Returns:
    - Threshold value
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if(threshold_mode == 'global'):
        # Select the background pixels from the four corners of the image
        background_pixels = np.array([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]])


        # Define the remainder as object pixels
        object_pixels = image.ravel()
        object_pixels = object_pixels[~np.isin(object_pixels, background_pixels)]

        # Initialize threshold value
        threshold = 0

        # Start the iterative process
        while True:
            # Compute mean background and object gray-level
            mean_background = np.mean(background_pixels)
            mean_object = np.mean(object_pixels)

            # Calculate new threshold
            new_threshold = (mean_background + mean_object) / 2

            # If threshold doesn't change, break the loop
            if abs(new_threshold - threshold) < 0.01:
                break

            threshold = new_threshold

            background_pixels = image[image < threshold]
            object_pixels = image[image > threshold]

            # Apply the final threshold to the image
        return threshold

    else:
        # Initialize the final binary image
        final_image = np.zeros_like(image)

        # Split the image into four equal parts
        height, width = image.shape
        half_height = height // 2
        half_width = width // 2

        # Define the thresholds for each section
        thresholds = np.zeros(4)

        # Iterate over each section of the image
        for i in range(2):
            for j in range(2):
                # Define the coordinates for the current section
                start_row = i * half_height
                end_row = (i + 1) * half_height
                start_col = j * half_width
                end_col = (j + 1) * half_width

                # Extract the current section of the image
                section = image[start_row:end_row, start_col:end_col]

                background_pixels = np.array([section[0, 0], section[0, -1], section[-1, 0], section[-1, -1]])

                # Define the remainder as object pixels
                object_pixels = image.ravel()
                object_pixels = object_pixels[~np.isin(object_pixels, background_pixels)]

                # Initialize threshold value
                threshold = 0

                # Start the iterative process
                while True:
                    # Compute mean background and object gray-level
                    mean_background = np.mean(background_pixels)
                    mean_object = np.mean(object_pixels)

                    # Calculate new threshold
                    new_threshold = (mean_background + mean_object) / 2

                    thresholds[i * 2 + j] = new_threshold

                    # If threshold doesn't change, break the loop
                    if abs(new_threshold - threshold) < 0.01:
                        break

                    threshold = new_threshold

                    background_pixels = section[section < threshold]
                    object_pixels = section[section > threshold]
        return thresholds

def otsu_threshold(image):
    """
    Apply Otsu thresholding on the input image to create a binary image.

    Parameters:
    - image: the input image to be thresholded

    Returns:
    - Threshold value
   """

    # Calculate the total number of pixels in the image
    pixel_number = image.shape[0] * image.shape[1]
    # Calculate the weight of each pixel
    mean_weight = 1.0 / pixel_number
    # Compute the histogram of pixel intensities
    his, bins = np.histogram(image, np.arange(0, 257))
    final_thresh = -1
    final_variance = -1
    intensity_arr = np.arange(256)

    # Iterate over each possible threshold value
    for t in bins[0:-1]:
        # Compute the number of pixels in the foreground and background
        p1 = np.sum(his[:t])
        p2 = np.sum(his[t:])
        # Skip if either foreground or background has no pixels
        if p1 == 0 or p2 == 0:
            continue
        # Compute the weight of each class
        W1 = p1 * mean_weight
        W2 = p2 * mean_weight

        # Compute the mean intensity of each class
        m1 = np.sum(intensity_arr[:t] * his[:t]) / float(p1)
        m2 = np.sum(intensity_arr[t:] * his[t:]) / float(p2)

        # Calculate the between class variance
        variance = W1 * W2 * (m1 - m2) ** 2

        # Update the threshold value if the variance is higher
        if variance > final_variance:
            final_thresh = t
            final_variance = variance

    return final_thresh

def spectral_threshold(img):
    """
   Apply Spectral thresholding on the input image to create a binary image.

   Parameters:
   - image: the input image to be thresholded

   Returns:
   - final_img: the binary image after thresholding
    """

    # Compute the histogram of pixel intensities
    hist, _ = np.histogram(img, 256, [0, 256])
    # Compute the mean intensity of the image
    mean = np.sum(np.arange(256) * hist) / float(img.size)
    low_threshold = 0
    high_threshold = 0
    max_variance = 0

    # Iterate over all possible threshold values
    for high in range(0, 256):
        for low in range(0, high):
            # Compute the weights of each class
            w0 = np.sum(hist[0:low])
            if w0 == 0:
                continue
            mean0 = np.sum(np.arange(0, low) * hist[0:low]) / float(w0)
            w1 = np.sum(hist[low:high])
            if w1 == 0:
                continue
            mean1 = np.sum(np.arange(low, high) * hist[low:high]) / float(w1)
            w2 = np.sum(hist[high:])
            if w2 == 0:
                continue
            mean2 = np.sum(np.arange(high, 256) * hist[high:]) / float(w2)
            # Compute the between class variance
            variance = w0 * (mean0 - mean) ** 2 + w1 * (mean1 - mean) ** 2 + w2 * (mean2 - mean) ** 2
            # Update the threshold values if the variance is higher
            if variance > max_variance:
                max_variance = variance
                low_threshold = low
                high_threshold = high

    # Create a binary image based on the computed thresholds
    binary = np.zeros(img.shape, dtype=np.uint8)
    binary[img < low_threshold] = 0
    binary[(img >= low_threshold) & (img < high_threshold)] = 128
    binary[img >= high_threshold] = 255

    return binary

