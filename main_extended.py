import pandas as pd
from src.classifier import get_features, train_model, evaluate_model

data = pd.read_csv("results/features_extended.csv")

features = get_features(data)

model = train_model(data, features, max_depth=4)

evaluate_model(
    data=data,
    features=features,
    model=model,
    result_dir="results",
    output_name="predictions_extended.csv"
)