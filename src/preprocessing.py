import cv2

def enhance_color_hsv_clahe(img, clip_limit = 2.0, tile_grid_size = (8, 8)):
    """
    Improve the image contrast by applying CLAHE to the value(Brightness) channel in the HSV color 
    space. This highlights the textures of the lesion without altering the colors (Hue/Saturation)
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit = clip_limit, tileGridSize = tile_grid_size)

    v_enhanced = clahe.apply(v)

    hsv_enhanced = cv2.merge([h, s, v_enhanced])
    enhanced_image = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return enhanced_image