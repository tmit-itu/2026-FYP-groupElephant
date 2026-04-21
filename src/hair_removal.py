import cv2
import numpy as np 

def remove_hair(img):
    #Gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Blackhat transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    #Create a hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    #Inpainting
    dst = cv2.inpaint(img, hair_mask, inpaintRadius = 1, flags = cv2.INPAINT_TELEA)

    return dst, hair_mask