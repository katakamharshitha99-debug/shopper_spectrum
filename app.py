import streamlit as st

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.success("Select a page above.")

st.title("🛍️ Shopper Spectrum")

st.markdown("""
### Customer Segmentation & Product Recommendation System

Welcome to **Shopper Spectrum**, an AI-powered retail analytics application.

### Features

- 📊 Exploratory Data Analysis
- 👥 Customer Segmentation
- 🎯 Product Recommendation
- 📈 Interactive Dashboard
- 💡 Business Insights

Use the **sidebar** to navigate through different pages.
""")


