import pandas as pd
import os
import cv2
from feature_asymmetry import get_assymetry
from feature_border import get_border
from feature_texture import mean_gradient
from feature_diameter import get_diameter


def extract_all():
    metadata = pd.read_csv("../metadata.csv")
    diagnosis_mapping = {"MEL":1, "BCC":1, "SCC":1, "NEV":0, "ACK":0, "SEK":0}
    metadata["cancer"] = metadata["diagnostic"].map(diagnosis_mapping)

    results = []

    for index, row in metadata.iterrows():
        img_id = os.path.splitext(row["img_id"])[0] 

        img = cv2.imread(f"../data/imgs/{img_id}.png")
        mask = cv2.imread(f"../data/masks/{img_id}_mask.png", cv2.IMREAD_GRAYSCALE)

        feature_a = get_assymetry(mask)
        feature_b = get_border(mask)
        feature_t = mean_gradient(img, mask)
        feature_d = get_diameter(mask)


        results.append({"img_id": img_id,
            "asymmetry": feature_a,
            "border": feature_b,
             "texture": feature_t,
            "diameter": feature_d,
            "cancer": row["cancer"]})
    
    df_features = pd.DataFrame(results)
    df_features.to_csv("../results/features.csv", index=False)

if __name__ == "__main__":
    extract_all()
    
