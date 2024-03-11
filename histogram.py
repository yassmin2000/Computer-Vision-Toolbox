import numpy as np


# def histogram(image):
#     hist = np.zeros(256, dtype=int)
#     # a loop through all pixels to make the histogram
#     for pixel in image.flatten():
#         hist[pixel] += 1
#     return hist
def histogram(image):
    try:
        hist = np.zeros(256, dtype=int)
        for pixel in image.flatten():
            hist[pixel] += 1
        return hist
    except Exception as e:
        print("Error in histogram function:", e)