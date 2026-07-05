import streamlit as st
import plotly.express as px
from utils import load_data

df = load_data()

st.title("📊 Exploratory Data Analysis")

# ---------------- Top Countries ----------------

country = (
    df.groupby("Country")["InvoiceNo"]
    .count()
    .sort_values(ascending=False)
    .head(10)
)

fig = px.bar(
    x=country.index,
    y=country.values,
    labels={"x": "Country", "y": "Transactions"},
    title="Top 10 Countries by Transactions"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Top Products ----------------

products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig2 = px.bar(
    x=products.index,
    y=products.values,
    labels={"x": "Product", "y": "Quantity"},
    title="Top Selling Products"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- Monthly Sales ----------------

sales = df.groupby("Month")["TotalPrice"].sum()

fig3 = px.line(
    x=sales.index.astype(str),
    y=sales.values,
    title="Monthly Sales Trend"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- Revenue by Country ----------------

revenue = (
    df.groupby("Country")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig4 = px.bar(
    x=revenue.index,
    y=revenue.values,
    labels={"x": "Country", "y": "Revenue"},
    title="Top 10 Countries by Revenue"
)

st.plotly_chart(fig4, use_container_width=True)

