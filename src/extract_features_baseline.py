import pandas as pd
import cv2
import os
from feature_asymmetry import mean_asymmetry
from feature_compactness import get_compactness
from feature_color import extract_color_features
from feature_diameter import get_diameter
from feature_texture import mean_gradient
from preprocessing import enhance_color_hsv_clahe

def extract_all():
    metadata = pd.read_csv("../metadata.csv")
    diagnosis_mapping = {'MEL': 1, 'BCC': 1, 'SCC': 1, 'NEV': 0, 'ACK': 0, 'SEK': 0}
    metadata["cancer"] = metadata["diagnostic"].map(diagnosis_mapping)
    imgs_in_folder = os.listdir("../data/imgs/")

    metadata_filtered = metadata[metadata["img_id"].isin(imgs_in_folder)]

    results = []
    total = len(metadata_filtered)
    print(f"Starting feature extraction for {total} images")

    for index, row in metadata_filtered.iterrows():
        img_id = row["img_id"][:-4] 

        img = cv2.imread(f"../data/imgs/{img_id}.png")
        mask = cv2.imread(f"../data/masks/{img_id}_mask.png", cv2.IMREAD_GRAYSCALE)

        if img is not None and mask is not None:
            img_preprocessed = enhance_color_hsv_clahe(img)

            feature_a = mean_asymmetry(mask)
            feature_b = get_compactness(mask)
            feature_c = extract_color_features(img_preprocessed, mask)
            feature_d = get_diameter(mask)
            feature_t = mean_gradient(img_preprocessed, mask)


            results.append({"img_id": row["img_id"],
                "patient_id": row["patient_id"],
                "cancer": row["cancer"],
                "asymmetry": feature_a,
                "border": feature_b,
                "texture": feature_t,
                "h_mean": feature_c[0],
                "s_mean": feature_c[1],
                "v_mean": feature_c[2],
                "h_std": feature_c[3],
                "s_std": feature_c[4],
                "v_std": feature_c[5],
                "color_entropy": feature_c[6]
                })
    
    df_features = pd.DataFrame(results)
    df_features.to_csv("../results/features.csv", index=False)

if __name__ == "__main__":
    extract_all()