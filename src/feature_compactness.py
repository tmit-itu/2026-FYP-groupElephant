import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

data_path = '../data/'

def load_image_and_mask(image_id, data_path=data_path):
    '''
    Docstring for load_image
    
    :param image_id: "img_id" from metadata.csv
    :param data_path: Relative path of the data folder

    This functions takes as input an image ID, 
    and returns the corresponding image and mask 
    (found in "/data/imgs/" and "/data/masks/" respectively)
    as an array
    '''
    
    img_path = data_path + "imgs/"
    mask_path = data_path + "masks/"

    # Load the image/mask
    file_im = img_path + image_id
    file_mask = (mask_path + image_id).replace(".png", "_mask.png")
    im = plt.imread(file_im)
    mask = plt.imread(file_mask)
    
    return im, mask

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
    binary_mask = mask > 0
    
    # 2. Calculate Area (count of all pixels in the lesion)
    area = np.sum(binary_mask)
    
    # Handle the edge case where the mask is completely empty to avoid division by zero
    if area == 0:
        return 0.0 
    
    # 3. Calculate Perimeter using skimage's built-in perimeter function
    perimeter = measure.perimeter(binary_mask)
    
    # 4. Apply the formula
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    
    return compactness

# Load your specific image and mask
im, mask = load_image_and_mask("PAT_890_1693_904.png")

# Calculate the compactness
lesion_compactness = get_compactness(mask)

print(f"Compactness score: {lesion_compactness:.2f}")