import streamlit as st
import pandas as pd
from utils import load_models

# Load models
kmeans, scaler, similarity = load_models()

st.title("👥 Customer Segmentation")

st.write("Enter the customer's RFM values to predict the customer segment.")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input("Recency (Days)", min_value=0, value=30)

with col2:
    frequency = st.number_input("Frequency", min_value=1, value=5)

with col3:
    monetary = st.number_input("Monetary Value", min_value=0.0, value=500.0)

if st.button("Predict Customer Segment"):

    # Create input dataframe
    customer = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )

    # Scale input
    customer_scaled = scaler.transform(customer)

    # Predict cluster
    cluster = kmeans.predict(customer_scaled)[0]

    # Segment names
    segment_names = {
        0: "🏆 High Value Customer",
        1: "🙂 Regular Customer",
        2: "⚠️ At Risk Customer",
        3: "🛍️ Occasional Customer"
    }

    # Get segment name
    segment = segment_names.get(cluster, "Unknown")

    st.success(f"Predicted Segment: {segment}")

    # Business insights
    if cluster == 0:
        st.info("""
### 🏆 High Value Customer

**Characteristics**
- Frequent Purchases
- High Spending
- Recently Active

**Recommended Strategy**
- Loyalty Rewards
- Premium Offers
- Early Access
""")

    elif cluster == 1:
        st.info("""
### 🙂 Regular Customer

**Characteristics**
- Consistent Purchases
- Moderate Spending

**Recommended Strategy**
- Cross Selling
- Bundle Offers
- Membership Plans
""")

    elif cluster == 2:
        st.warning("""
### ⚠️ At Risk Customer

**Characteristics**
- Inactive Recently
- Low Purchase Frequency

**Recommended Strategy**
- Discount Coupons
- Email Campaigns
- Win-back Offers
""")

    else:
        st.info("""
### 🛍️ Occasional Customer

**Characteristics**
- Purchases Occasionally
- Moderate Engagement

**Recommended Strategy**
- Festival Offers
- Personalized Recommendations
""")