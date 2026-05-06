import cv2
import numpy as np 

def mean_gradient(image,mask):
    #Reduce dimensions to run the code faster
    image = cv2.resize(image, (256, 256))
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)

    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    texture_values = magnitude[mask > 0]

    if len(texture_values) == 0:
        return 0.0
    
    return float(np.mean(texture_values))