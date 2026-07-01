import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
import gdown

st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Only similarity.pkl needs to be downloaded from Drive.
# clean_data.csv, online_retail.csv, kmeans.pkl and scaler.pkl
# are small enough to commit directly to the repo — do that
# instead of pulling them from Drive on every app boot.
# ---------------------------------------------------------
SIMILARITY_FILE_ID = "1NsgLnRDIsiWi8JhVzoXw78vYxqx8jo8a"
SIMILARITY_FILENAME = "similarity.pkl"

if not os.path.exists(SIMILARITY_FILENAME):
    try:
        gdown.download(
            id=SIMILARITY_FILE_ID,
            output=SIMILARITY_FILENAME,
            quiet=False
        )
    except Exception as e:
        st.error(
            f"Could not download {SIMILARITY_FILENAME} from Google Drive. "
            f"Make sure the file's sharing setting is 'Anyone with the link'. "
            f"Error: {e}"
        )
        st.stop()

if not os.path.exists(SIMILARITY_FILENAME):
    st.error(f"{SIMILARITY_FILENAME} was not found after download attempt.")
    st.stop()

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

    product = st.text_input("Enter Product Name")
    if st.button("Recommend"):
        similarity_df = similarity  # already loaded above

        if product in similarity_df.index:
            recommendations = similarity_df[product].sort_values(ascending=False).head(5)
            st.write("Top 5 Recommended Products:")
            for i, (prod, score) in enumerate(recommendations.items(), start=1):
                st.write(f"{i}. {prod} (Similarity Score: {score:.2f})")
        else:
            st.write("Product not found in the dataset.")

st.markdown("---")
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

with st.container():
    st.metric("Customers", df["CustomerID"].nunique())
    st.write(
        "The dataset contains a total of 541,909 rows and 8 columns. "
        "It is a transactional dataset that captures the details of each "
        "transaction made by customers in an online retail store. The dataset "
        "is useful for various analyses, including customer segmentation, "
        "product recommendation, and sales trend analysis."
    )
    st.markdown("---")

st.markdown(
    "<center>Created by <b>Himarshitha</b></center>",
    unsafe_allow_html=True
)