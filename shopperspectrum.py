import pickle
import pandas as pd

# Load the similarity matrix
with open("similarity.pkl", "rb") as f:
    similarity_df = pickle.load(f)

def recommend(product):
    if product not in similarity_df.columns:
        return pd.Series(dtype=float)

    scores = similarity_df[product]
    scores = scores.sort_values(ascending=False)

    return scores.iloc[1:6]

