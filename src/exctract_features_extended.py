import pandas as pd
import cv2
import os
from hair_removal import remove_hair
from feature_asymmetry import get_assymetry
from feature_border import get_border
from feature_texture import mean_gradient

def extract_extended():
    metadata = pd.read_csv("../metadata.csv")
    diagnosis_mapping = {'MEL': 1, 'BCC': 1, 'SCC': 1, 'NEV': 0, 'ACK': 0, 'SEK': 0}
    metadata['cancer'] = metadata['cancer'].map(diagnosis_mapping)

    results = []

    for index, row in metadata.iterrows():
        img_id = row["img_id"].str[:-4]
        img = cv2.imread(f"../data/imgs/{img_id}.png")
        mask = cv2.imread(f"../data/masks/{img_id}_mask.png", 0)

        if img is not None and mask is not None:
            img_clean, _ = remove_hair(img)

            feature_a = get_asymmetry(mask)
            feature_b = get_border(mask)
            feature_t = mean_gradient(img_clean, mask)
            
            results.append({
                "img_id": img_id,
                "asymmetry": feature_a,
                "border": feature_b,
                "texture": feature_t,
                "cancer": row["cancer"]
            })
    
    df_features = pd.DataFrame(results)
    df_features.to_csv("../results/features_extended.csv", index=False)

if __name__ == "__main__":
    extract_extended()
