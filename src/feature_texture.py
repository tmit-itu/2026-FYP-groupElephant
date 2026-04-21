import cv2
import numpy as np 

def mean_gradient(image,mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)

    texture_values = magnitude[mask > 0]

    if len(texture_values) == 0:
        return 0.0
    
    return float(np.mean(texture_values))