import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os
import gdown
import warnings

# Suppress feature name warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
            quiet=True
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

# ---------------- CUSTOM STYLING FIXES ----------------
st.markdown("""
<style>
/* Let the system automatically pick the best contrasting text color for metrics */
div[data-testid="stMetricValue"] > div {
    font-weight: 700 !important;
}

/* Original Card Components */
.result-card {
    background-color: #16213e;
    border-left: 5px solid #4f8bf9;
    padding: 22px 24px;
    border-radius: 10px;
    margin-top: 10px;
}
.result-card h2 {
    margin: 0 0 8px 0;
    color: #ffffff !important;
}
.result-card p {
    color: #c9d1d9 !important;
    margin: 0;
}
.rec-card {
    background-color: #ffffff;
    color: #111111;
    padding: 16px 22px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
}
.rec-badge {
    background-color: #7c3aed;
    color: white;
    border-radius: 50%;
    width: 30px;
    height: 30px;
    min-width: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin-right: 16px;
}
.rec-name {
    flex: 1;
    font-weight: 600;
    font-size: 15px;
    color: #111111;
}
.rec-score {
    color: #555555;
    font-size: 14px;
    white-space: nowrap;
    margin-left: 12px;
}
</style>
""", unsafe_allow_html=True)

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

# Sidebar Navigation Setup
st.sidebar.image(
    "https://img.icons8.com/color/96/shopping-cart.png",
    width=80
)
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Navigation Menu",
    ["Home", "EDA Dashboard", "Customer Segmentation", "Product Recommendation"],
    label_visibility="collapsed"
)

# ---------------- HOME ----------------
if menu == "Home":
    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Customers", f"{df['CustomerID'].nunique():,}")
    with col2:
        st.metric("Countries", df["Country"].nunique())
    with col3:
        st.metric("Transactions", f"{len(df):,}")
    with col4:
        st.metric("Revenue", f"${df['TotalPrice'].sum():,.2f}")

    st.subheader("Operational Summary")
    st.write(
        "The dataset contains a total of 541,909 rows and 8 columns. "
        "It is a transactional dataset that captures the details of each "
        "transaction made by customers in an online retail store. The dataset "
        "is useful for various analyses, including customer segmentation, "
        "product recommendation, and sales trend analysis."
    )

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

    # Monthly Sales
    sales = df.groupby("Month")["TotalPrice"].sum()
    fig3 = px.line(
        x=sales.index.astype(str),
        y=sales.values,
        title="Monthly Sales"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Revenue by Country
    revenue = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
    fig4 = px.bar(
        revenue,
        title="Revenue by Country"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---------------- CUSTOMER SEGMENTATION ----------------
elif menu == "Customer Segmentation":
    st.markdown("## 🧑‍💼 Customer Segmentation Predictor")
    st.write(
        "Enter a customer's **RFM values** to predict their segment using the "
        "trained **K-Means clustering model**."
    )

    SEGMENT_INFO = {
        0: {"name": "High-Value Customer", "icon": "💎",
            "desc": "Buys often, spends a lot, and purchased recently. Your best customers."},
        1: {"name": "Regular Customer", "icon": "🛍️",
            "desc": "Consistent, moderate spending and purchase frequency."},
        2: {"name": "At-Risk Customer", "icon": "⚠️",
            "desc": "Used to buy regularly but hasn't purchased in a while."},
        3: {"name": "New / Low-Engagement Customer", "icon": "🌱",
            "desc": "Recent or infrequent buyer with low total spend so far."},
    }

    with st.expander("📘 Segment Reference Guide"):
        for cluster_id, info in SEGMENT_INFO.items():
            st.markdown(f"**{info['icon']} {info['name']}** — {info['desc']}")

    st.markdown("")
    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown("### Enter RFM Values")
        r = st.number_input(
            "📅 Recency (days since last purchase)",
            min_value=0, value=30, step=1,
            help="How many days ago the customer's last purchase was."
        )
        f = st.number_input(
            "🔁 Frequency (number of transactions)",
            min_value=0, value=10, step=1,
            help="Total number of purchases the customer has made."
        )
        m = st.number_input(
            "💰 Monetary (total spend)",
            min_value=0.0, value=300.0, step=10.0,
            help="Total amount the customer has spent."
        )
        predict_clicked = st.button("🔮 Predict Customer Segment", use_container_width=True)

    with col_result:
        st.markdown("### Prediction Result")
        if predict_clicked:
            input_data = scaler.transform([[r, f, m]])
            cluster = int(kmeans.predict(input_data)[0])
            info = SEGMENT_INFO.get(
                cluster, {"name": f"Cluster {cluster}", "icon": "🔹", "desc": ""}
            )
            st.markdown(f"""
            <div class="result-card">
                <h2>{info['icon']} {info['name']}</h2>
                <p>{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👉 Fill in the RFM values and click **Predict Customer Segment**")

# ---------------- PRODUCT RECOMMENDATION ----------------
elif menu == "Product Recommendation":
    st.markdown("## 🎯 Product Recommendation System")
    st.write(
        "Uses **Item-Based Collaborative Filtering** (cosine similarity) to "
        "recommend the most similar products based on shared purchase history."
    )

    similarity_df = similarity
    product_list = sorted(similarity_df.index.astype(str).tolist())

    col_select, col_count = st.columns([3, 1])
    with col_select:
        product = st.selectbox("Select a Product", product_list)
    with col_count:
        top_n = st.selectbox("# Recommendations", [3, 5, 10], index=1)

    get_recs = st.button("🔍 Get Recommendations", use_container_width=True)

    if get_recs:
        if product in similarity_df.index:
            recommendations = (
                similarity_df[product]
                .sort_values(ascending=False)
                .drop(labels=[product], errors="ignore")
                .head(top_n)
            )

            st.markdown(f"### Recommendations for: *{product}*")

            for i, (prod, score) in enumerate(recommendations.items(), start=1):
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-badge">{i}</div>
                    <div class="rec-name">{prod}</div>
                    <div class="rec-score">Similarity: {score:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

            chart_df = recommendations.reset_index()
            chart_df.columns = ["Product", "Similarity"]

            fig_rec = px.bar(
                chart_df.sort_values("Similarity"),
                x="Similarity",
                y="Product",
                orientation="h",
                title="Cosine Similarity Score",
            )
            st.plotly_chart(fig_rec, use_container_width=True)
        else:
            st.warning("Product not found in the dataset.")

    with st.expander("📦 Browse All Products"):
        st.dataframe(
            pd.DataFrame({"Product Name": product_list}),
            use_container_width=True,
            height=300
        )

# Clean simple footer
st.markdown("<br><hr><center>Created by Himarshitha</center>", unsafe_allow_html=True)

