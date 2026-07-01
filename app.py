import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
import gdown

FILES = {
    "clean_data.csv": "1Gy04M8teC0eS3Tv8C8mhTiVdEDUm7nrB",
    "online_retail.csv": "1u4HR4sDjsh7GIwEVr_Z5iGoPaF_L0WMD",
    "kmeans.pkl": "1aNNkqM-U9zLUCjXGBdpeeZUig673bkld",
    "scaler.pkl": "11LR4-Xt4ACsmcWUfJC35QqmtPGU-SEr8",
    "similarity.pkl": "1NsgLnRDIsiWi8JhVzoXw78vYxqx8jo8a"
}

for filename, file_id in FILES.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False, fuzzy=True)

st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

fuzzy=True


st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
df = pd.read_csv("clean_data.csv")

st.markdown("""
<h1 style='text-align:center; color:#2E86C1;'>
🛒 Shopper Spectrum
</h1>

<h4 style='text-align:center; color:gray;'>
Customer Segmentation & Product Recommendation System
</h4>
""", unsafe_allow_html=True)


# Load saved models
with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)
    
# Sidebar

st.sidebar.image(
    "https://img.icons8.com/color/96/shopping-cart.png",
    width=80
)

st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "EDA Dashboard", "Customer Segmentation", "Product Recommendation"]
)

# ---------------- HOME ----------------
if menu == "Home":

    st.header("Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Customers", df["CustomerID"].nunique())

    with col2:
        st.metric("Countries", df["Country"].nunique())

    with col3:
        st.metric("Transactions", len(df))

    with col4:
        st.metric("Revenue", round(df["TotalPrice"].sum(), 2))


    with st.expander("Dataset Information"):
        st.write(df.head())
        st.write(df.describe())
        
        
# ---------------- EDA DASHBOARD ----------------
elif menu == "EDA Dashboard":

    st.header("Exploratory Data Analysis")

    # Top Countries
    country = (
        df.groupby("Country")["InvoiceNo"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )

    fig = px.bar(
        x=country.index,
        y=country.values,
        title="Top 10 Countries"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Top Products
    products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig2 = px.bar(
        x=products.index,
        y=products.values,
        title="Top Selling Products"
    )

    st.plotly_chart(fig2, use_container_width=True)

    sales = df.groupby("Month")["TotalPrice"].sum()

    fig3 = px.line(
    x=sales.index.astype(str),
    y=sales.values,
    title="Monthly Sales"
    )

    st.plotly_chart(fig3, use_container_width=True)
    
    revenue = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)

    fig4 = px.bar(
    revenue,
    title="Revenue by Country"
)

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown(
    "<center>Created by <b>Himarshitha</b></center>",
    unsafe_allow_html=True
)

# ---------------- CUSTOMER SEGMENTATION ----------------
elif menu == "Customer Segmentation":

    st.header("Customer Segmentation")

    # We'll add prediction code here later
    r = st.number_input("Recency")
    f = st.number_input("Frequency")
    m = st.number_input("Monetary")
    
    st.markdown("---")
    st.markdown(
    "<center>Created by <b>Himarshitha</b></center>",
    unsafe_allow_html=True
)


     
# ---------------- PRODUCT RECOMMENDATION ----------------
elif menu == "Product Recommendation":

    st.header("Product Recommendation")
    st.success("Top 5 Recommended Products")

    # We'll add recommendation code here later
    
    product = st.text_input("Enter Product Name")
    if st.button("Recommend"):
        # Load the similarity matrix
        similarity_df = pickle.load(open("similarity.pkl", "rb"))

        if product in similarity_df.index:
            recommendations = similarity_df[product].sort_values(ascending=False).head(5)
            st.write("Top 5 Recommended Products:")
            for i, (prod, score) in enumerate(recommendations.items(), start=1):
                st.write(f"{i}. {prod} (Similarity Score: {score:.2f})")
        else:
            st.write("Product not found in the dataset.")
            
            st.title("About Project")



st.write("""
Shopper Spectrum is an E-Commerce analytics application.

Features:

✔ Customer Segmentation

✔ Product Recommendation

✔ Interactive Dashboard

✔ RFM Analysis

✔ KMeans Clustering
""")

col1,col2 = st.columns(2)

st.metric("Customers", df["CustomerID"].nunique())
with st.container():
    st.write("The dataset contains a total of 541,909 rows and 8 columns. It is a transactional dataset that captures the details of each transaction made by customers in an online retail store. The dataset is useful for various analyses, including customer segmentation, product recommendation, and sales trend analysis.")
    st.markdown("---")
st.markdown(
    "<center>Created by <b>Himarshitha</b></center>",
    unsafe_allow_html=True
)
