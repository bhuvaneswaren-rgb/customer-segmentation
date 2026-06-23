import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
st.set_page_config(page_title="Customer Segmenter", layout="wide")
st.title("🛍️ Customer Segmentation Web App")
st.write("This app uses **K-Means Clustering** to group 1,000  customers based on their spending habits.")
@st.cache_data 
def generate_data(n=1000):
    np.random.seed(42)
    income = np.random.normal(55, 20, n).clip(15, 150)
    spending = (0.3 * income + np.random.normal(30, 15, n))
    spending = np.interp(spending, (spending.min(), spending.max()), (1, 100))
    return pd.DataFrame({
        'Annual Income (k$)': income,
        'Spending Score (1-100)': spending
    })
df = generate_data()
if st.checkbox('Show Raw Data'):
    st.write(df.head(10))
X = df.values
st.sidebar.header("Cluster Settings")
k_value = st.sidebar.slider("Select Number of Clusters (K)", 2, 10, 5)
kmeans = KMeans(n_clusters=k_value, init='k-means++', n_init='auto', random_state=42)
y_kmeans = kmeans.fit_predict(X)
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.get_cmap('tab10')
for i in range(k_value):
    ax.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
               s=50, c=[cmap(i)], label=f'Cluster {i+1}', alpha=0.6)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           s=250, c='yellow', marker='X', label='Centroids', edgecolors='black')
ax.set_title('Customer Groups (Synthetic Dataset)')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.legend()
st.pyplot(fig)
st.subheader("💡 What does this mean?")
st.write(f"The algorithm has divided your **1,000 customers** into **{k_value} groups**.")
st.info("""
- **Top Right:** High Income, High Spending (Target for Luxury/Loyalty)
- **Bottom Right:** High Income, Low Spending (Target for Promotions)
- **Top Left:** Low Income, High Spending (Potential Brand Enthusiasts)
- **Bottom Left:** Low Income, Low Spending (Budget Conscious)
""")