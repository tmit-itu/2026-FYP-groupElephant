
import cv2 as cv
import numpy as np

def extract_color_features(image, slic_segments, mask):
    
    """Extract color features from an object in the image.
    The features describe color variability, dominant colors 
    and color asymmetry inside the masked object.
    """
    features = []
    
    mean_hsv = mean_hsv_in_mask(image, mask)
    features.extend(mean_hsv)

    rgb_variance = rgb_var(image, slic_segments)
    features.extend(rgb_variance)

    hsv_variance = hsv_var(image, slic_segments)
    features.extend(hsv_variance)

    dom_colors = color_dominance(
        image,
        mask,
        clusters=3,
        include_ratios=False
    )

    features.extend(dom_colors.flatten()) 
    
    color_asym = color_asymmetry_hsv(image, mask)
    features.extend(color_asym)
    
    return np.array(features, dtype=np.float32)