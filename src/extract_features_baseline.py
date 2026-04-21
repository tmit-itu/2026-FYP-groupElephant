import pandas as pd
import os
import cv2
from feature_asymmetry import get_assymetry
from feature_border import get_border
from feature_texture import mean_gradient


def extract_all():
    metadata = pd.read_csv("../metadata.csv")

    results = []

    for index, row in metadata.iterrows():
        img_id = row["img_id"].str[:-4] 

        img = cv2.imread(f"../data/imgs/{img_id}.png")
        mask = cv2.imread(f"../data/masks/{img_id}_mask.png")

        feature_a = get_assymetry(mask)
        feature_b = get_border(mask)


        results.append({"img_id": img_id,
            "asymmetry": feature_a,
            "border": feature_b,
            "label": row["diagnostic"]})
    
    
