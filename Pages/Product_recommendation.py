import streamlit as st
from utils import load_models

# Load models
_, _, similarity = load_models()

st.title("🎯 Product Recommendation")

st.write("Select a product to get the Top 5 similar products.")

product = st.selectbox(
    "Choose Product",
    sorted(similarity.index.tolist())
)

if st.button("Recommend Products"):

    recommendations = (
        similarity[product]
        .sort_values(ascending=False)
        .iloc[1:6]
    )

    st.success("Top 5 Recommended Products")

    for i, (item, score) in enumerate(recommendations.items(), start=1):
        st.markdown(f"### {i}. {item}")
        st.progress(float(score))
        st.caption(f"Similarity Score: {score:.2f}")
        
        