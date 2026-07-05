import pandas as pd
import pickle
import streamlit as st


@st.cache_data
def load_data():
    return pd.read_csv("clean_data.csv")


@st.cache_resource
def load_models():
    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)

    return kmeans, scaler, similarity

