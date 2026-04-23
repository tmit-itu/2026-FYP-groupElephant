import numpy as np
from skimage import measure


def get_compactness(mask):
    """
    Calculates the compactness of a lesion based on its binary mask.
    A perfect circle will have a compactness close to 1. 
    Highly irregular or branching shapes will have higher values.
    
    :param mask: A 2D numpy array representing the mask.
    :return: A float representing the compactness.
    """
    # 1. Ensure the mask is binary (True/False). 
    # This protects against masks loaded as 0-255 instead of 0-1.
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    binary_mask = mask > 0
    
    # 2. Calculate Area (count of all pixels in the lesion)
    area = np.sum(binary_mask)
    
    # Handle the edge case where the mask is completely empty to avoid division by zero
    if area == 0:
        return 0.0 
    
    # 3. Calculate Perimeter using skimage's built-in perimeter function
    perimeter = measure.perimeter(binary_mask)

    if perimeter == 0:
        return 0.0
    
    # 4. Apply the formula
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    return float(compactness)