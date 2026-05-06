import cv2
import numpy as np


def get_diameter(mask):
    """
    Calculate lesion diameter in pixels from a binary mask.

    Diameter is defined here as the maximum Euclidean distance
    between any two contour points of the lesion.

    Parameters
    ----------
    mask : numpy.ndarray
        Lesion mask image.

    Returns
    -------
    float
        Diameter in pixels.
    """

    if mask is None:
        return np.nan

    # If mask has 3 channels, convert to grayscale
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Make sure mask is binary
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Find lesion contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return np.nan

    # Keep the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    points = largest_contour[:, 0, :].astype(np.float32)

    if len(points) < 2:
        return 0.0

    # FAST diameter approximation
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    diameter = radius * 2

    return float(diameter)