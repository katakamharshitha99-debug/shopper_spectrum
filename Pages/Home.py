import streamlit as st
from utils import load_data

df = load_data()

st.title("🛒 Shopper Spectrum")

st.subheader("Customer Segmentation & Product Recommendation System")

st.image(
    "https://images.unsplash.com/photo-1556740749-887f6717d7e4?w=1200",
    use_container_width=True
)

st.markdown("---")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Customers", df["CustomerID"].nunique())
c2.metric("Countries", df["Country"].nunique())
c3.metric("Transactions", len(df))
c4.metric("Revenue", f"${df['TotalPrice'].sum():,.2f}")

st.markdown("---")

st.header("Project Overview")

st.write("""
Shopper Spectrum is an intelligent retail analytics platform that uses Machine Learning to analyze customer purchase behaviour.

### Main Modules

- 📊 Interactive Dashboard
- 👥 Customer Segmentation
- 🎯 Product Recommendation
- 📈 Business Insights
""")

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(df.head(10))

