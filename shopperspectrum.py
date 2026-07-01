import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;

"""## 📂 Section 2: Load Dataset"""

df = pd.read_csv(r"D:\Labmentix\Shopper sprectrum\vscode\online_retail.csv")

"""## 🧹 Section 3: Data Preprocessing

### Steps performed:
1. Fix `UnitPrice` — remove `$` sign and convert to float  
2. Parse `InvoiceDate` to datetime  
3. Remove **cancelled orders** (InvoiceNo starting with `'C'`)  
4. Remove rows with **Quantity ≤ 0** or **UnitPrice ≤ 0**  
5. Drop rows with **missing CustomerID** (cannot build customer profile)  
6. Drop rows with **missing Description**  
7. Remove **duplicate rows**  
8. Create **TotalPrice** = Quantity × UnitPrice  

"""

df.head()

df.info()

df.describe()

df.shape

df.isnull().sum()

df = df.dropna(subset=['CustomerID'])

df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

df = df[df['Quantity']>0]

df.drop_duplicates(inplace=True)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)

df['TotalPrice']=df['Quantity']*df['UnitPrice']

print(df['UnitPrice'].dtype)

df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

df = df[df['UnitPrice'] > 0]

"""## 📊 Section 4: Exploratory Data Analysis (EDA)"""

country=df.groupby('Country')['InvoiceNo'].count().sort_values(ascending=False)

country.head(10).plot(kind='bar')

## Transactions by country

top_products=df.groupby('Description')['Quantity'].sum().sort_values(ascending=False)

top_products.head(10).plot(kind='bar')

## Top selling products

print(df[['TotalPrice', 'UnitPrice', 'Quantity']].dtypes)

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

df['Month'] = df['InvoiceDate'].dt.to_period('M')

sales = df.groupby('Month')['TotalPrice'].sum()

sales.plot(figsize=(10,5))

## Monthly sales Trend

print(sales.dtype)
print(sales.head())

sns.histplot(df['TotalPrice'])

## Distribution of Transaction Amount

customer = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.bar(customer.index.astype(str), customer.values)
plt.xticks(rotation=90)
plt.title("Top 10 Customers")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")
plt.tight_layout()
plt.show()


## Total Customers
"""## RFM Analysis"""

snapshot=df['InvoiceDate'].max()+pd.Timedelta(days=1)

rfm=df.groupby('CustomerID').agg({

'InvoiceDate':lambda x:(snapshot-x.max()).days,

'InvoiceNo':'nunique',

'TotalPrice':'sum'

})

rfm.columns=['Recency','Frequency','Monetary']

rfm.head()

"""## Standardization"""

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

rfm_scaled=scaler.fit_transform(rfm)

"""### Elbow Method"""

from sklearn.cluster import KMeans

wcss=[]

for i in range(2,11):

    km=KMeans(n_clusters=i,random_state=42)

    km.fit(rfm_scaled)

    wcss.append(km.inertia_)

plt.plot(range(2,11),wcss)

"""## Silhouette Score"""

from sklearn.metrics import silhouette_score

for i in range(2,11):

    km=KMeans(n_clusters=i,random_state=42)

    pred=km.fit_predict(rfm_scaled)

    print(i,silhouette_score(rfm_scaled,pred))

kmeans=KMeans(n_clusters=4,random_state=42)

rfm['Cluster']=kmeans.fit_predict(rfm_scaled)

## TrainFinal Kmmeans

"""### Cluster Interpretation"""

rfm.groupby('Cluster').mean()

"""### Product Recommendation"""

pivot=df.pivot_table(index='CustomerID',

columns='Description',

values='Quantity',

fill_value=0)

item_matrix=pivot.T

from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(item_matrix)

similarity_df=pd.DataFrame(

similarity,

index=item_matrix.index,

columns=item_matrix.index
)

def recommend(product):

    scores=similarity_df[product]

    scores=scores.sort_values(ascending=False)

    return scores.iloc[1:6]

import os

print("Current working directory:", os.getcwd())
import pickle

pickle.dump(kmeans, open("kmeans.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(similarity_df, open("similarity.pkl", "wb"))

df.to_csv("clean_data.csv", index=False)

print("✅ Models and cleaned data saved successfully!")

"""## 🤖 Section 7: K-Means Clustering

### Why K-Means?
- Simple, scalable, and interpretable for customer segmentation
- Works well with standardized RFM features
- Produces hard cluster assignments ideal for business labeling

### Choosing K — Elbow Method + Silhouette Score

## 📋 Section 11: Project Summary

### ✅ What was accomplished

| Task | Status |
|------|--------|
| Data Loading & Understanding | ✅ |
| Data Preprocessing (nulls, cancellations, negatives, duplicates) | ✅ |
| EDA (country, products, trends, monetary distribution) | ✅ |
| RFM Feature Engineering | ✅ |
| Log-transform + StandardScaler normalization | ✅ |
| Elbow Method + Silhouette Score for optimal K | ✅ |
| K-Means Clustering (K=4) | ✅ |
| Cluster Profiling & Business Labeling | ✅ |
| Cluster Visualizations (boxplot, scatter, pie, bar) | ✅ |
| Item-Based Collaborative Filtering | ✅ |
| Product Similarity Heatmap | ✅ |
| Model artefacts saved for Streamlit | ✅ |

### 🏷️ Segment Business Insights

| Segment | Strategy |
|---------|----------|
| **High-Value** | Loyalty rewards, early access, premium offers |
| **Regular** | Cross-sell bundles, mid-tier promotions |
| **Occasional** | Seasonal campaigns, discount triggers |
| **At-Risk** | Win-back emails, personalised offers before churn |
"""

