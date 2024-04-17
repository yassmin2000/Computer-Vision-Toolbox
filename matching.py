import numpy as np
import cv2

def SSD(img,templ):                               
        
        output = np.zeros((img.shape[0] - templ.shape[0] + 1, img.shape[1] - templ.shape[1] + 1), dtype=np.int32)

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                roi = img[i:i + templ.shape[0], j:j + templ.shape[1]]
                resultImg = cv2.subtract(roi, templ)
                resultImg = np.power(resultImg, 2)
                result = np.sum(resultImg)
                output[i, j] = result

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(output)
        roi = (minLoc[0], minLoc[1], templ.shape[1], templ.shape[0])
        # Draw rectangle around the ROI
        top_left = (roi[0], roi[1])
        bottom_right = (roi[0] + roi[2], roi[1] + roi[3])
        
        for x in range(top_left[0], bottom_right[0] + 1):
            img[top_left[1], x] = [0, 255, 0]  # Green color for top line
            img[bottom_right[1], x] = [0, 255, 0]  # Green color for bottom line

        for y in range(top_left[1], bottom_right[1] + 1):
            img[y, top_left[0]] = [0, 255, 0]  # Green color for left line
            img[y, bottom_right[0]] = [0, 255, 0]  # Green color for right line

        return img

        


def Normalized_Cross_Correlation(roi, target):   
        # Normalised Cross Correlation Equation
        cor=np.sum(roi*target)
        nor = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
        return cor / nor

def template_matching(image,target):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
        height,width=img.shape
        tar_height,tar_width=target.shape
        (max_Y,max_X)=(0, 0)
        MaxValue = 0

        # Set img, target and result value matrix
        img=np.array(img, dtype="int")
        target=np.array(target, dtype="int")
        NccValue=np.zeros((height-tar_height,width-tar_width))

        # calculate value using filter-kind operation from top-left to bottom-right
        for y in range(0,height-tar_height):
            for x in range(0,width-tar_width):
                # img roi
                roi=img[y:y+tar_height,x:x+tar_width]
                # calculate ncc value
                NccValue[y,x] = Normalized_Cross_Correlation(roi,target)
                # find the most match area
                if NccValue[y,x]>MaxValue:
                    MaxValue=NccValue[y,x]
                    (max_Y,max_X) = (y,x) 

        # Draw rectangle around matched area with reduced size
        offset = 20  # Adjust this value to control the size of the rectangle

        top_left_cord = (max_X + offset, max_Y + offset)
        bottom_right_cord = (top_left_cord[0] + tar_width - 1 - 2*offset, top_left_cord[1] + tar_height - 1 - 2*offset)

        # Draw top and bottom lines
        for x in range(top_left_cord[0], bottom_right_cord[0] + 1):
            image[top_left_cord[1], x] = [0, 255, 0]  # Green color for top line
            image[bottom_right_cord[1], x] = [0, 255, 0]  # Green color for bottom line

        # Draw left and right lines
        for y in range(top_left_cord[1], bottom_right_cord[1] + 1):
            image[y, top_left_cord[0]] = [0, 255, 0]  # Green color for left line
            image[y, bottom_right_cord[0]] = [0, 255, 0]  # Green color for right line
        return image

     
        
                