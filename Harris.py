import cv2
import numpy as np
from matplotlib import pyplot as plt
import time as Time

from edge_detection import gaussian_blur ,convolve2D
def sobel(image):

    """
    Apply the Sobel edge detection algorithm to an image.
    
    Parameters:
    - image: the input image to be processed
    
    Returns:
    - kernel_horizontal_result: horizontal gradient result
    - kernel_vertical_result: vertical gradient result
    """
    # Define Sobel kernels
    kernel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
   
    # Convolve the image with horizontal and vertical kernels

    kernel_horizontal_result = convolve2D(image, kernel_horizontal)
    kernel_vertical_result = convolve2D(image, kernel_vertical)
    
    return kernel_horizontal_result, kernel_vertical_result



def harris_operator(img,window_size,k,threshold):
    """
    Apply the Harris Corner Detection algorithm to an image.
    
    Parameters:
    - img: the input image
    - window_size: size of the window for computing derivatives
    - k: sensitivity parameter for Harris response function
    - threshold: threshold for corner detection
    
    Returns:
    - img: image with detected corners marked
    - final_time: time taken for corner detection
    """
    start_time = Time.time()


    # Apply Gaussian blur to the image
    img_gaussian = cv2.GaussianBlur(img,(3,3),0)
    height = img.shape[0]   
    width = img.shape[1]    
    matrix_R = np.zeros((height,width))
    
    #   Step 1 -Compute image gradients
    dx,dy = sobel(img_gaussian)
   

    #   Step 2 - Calculate product and second derivatives (dx2, dy2 , dxy)
    dx2=np.square(dx)
    dy2=np.square(dy)
    dxy=dx*dy

    offset = int( window_size / 2 )
    #   Step 3 -  calculate (Sx2, Sy2 , Sxy)
    print ("Finding Corners...")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
             # Compute elements of the structure tensor
            Sx2 = np.sum(dx2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sy2 = np.sum(dy2[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(dxy[y-offset:y+1+offset, x-offset:x+1+offset])

            #   Step 4 - Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
            H = np.array([[Sx2,Sxy],[Sxy,Sy2]])

            #   Step 5 - Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            det=np.linalg.det(H)
            tr=np.matrix.trace(H)
            R=det-k*(tr**2)
            matrix_R[y-offset, x-offset]=R
    
    #   Step 6 - Apply a threshold
    # Normalize the response matrix
    cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            value=matrix_R[y, x]
            if value>threshold:
                
                cv2.circle(img,(x,y),3,(0,255,0))


    end_time = Time.time()
    final_time = end_time - start_time
    return img , final_time




def lambda_minus(img, window_size, threshold):
    """
    Apply the Lambda Minus Corner Detection algorithm to an image.
    
    Parameters:
    - img: the input image
    - window_size: size of the window for computing derivatives
    - threshold: threshold for corner detection
    
    Returns:
    - img_normalized: image with detected corners marked and normalized
    - final_time: time taken for corner detection
    """
    start_time = Time.time()
     # Apply Gaussian blur to the image
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    height = img.shape[0]
    width = img.shape[1]
    #   Step 1 -Compute image gradients
    dx, dy = sobel(img_gaussian)
     #   Step 2 - Calculate product and second derivatives (dx2, dy2 , dxy)

    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxy = dx * dy
    offset = int(window_size / 2)
     #   Step 3 -  calculate (Sx2, Sy2 , Sxy)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Compute elements of the structure tensor
            Sx2 = np.sum(dx2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sy2 = np.sum(dy2[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(dxy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
             #   Step 4 - Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]

            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])
            # calculate eigenvalues and find the min
            eigenvalues = np.linalg.eigvals(H)
            min_eigenvalue = min(eigenvalues)
            # Apply threshold and mark corners
            if min_eigenvalue > threshold:
                cv2.circle(img, (x, y), 5, (255, 0, 0))
    
    # Normalize the image
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
  
    end_time = Time.time()
    final_time = end_time - start_time
    return img_normalized, final_time