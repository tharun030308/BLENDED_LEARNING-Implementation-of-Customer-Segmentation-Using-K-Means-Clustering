# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries, load the dataset, and select relevant features for clustering.<br/>
2. Standardize the data using StandardScaler and determine the optimal number of clusters using the Elbow method.<br/>
3. Apply the K-Means clustering algorithm to group customers into clusters.<br/>
4. Evaluate the clustering using silhouette score and visualize the customer segments using a scatter plo
## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# Step 1: Load the dataset
data = pd.read_csv('CustomerData.csv')

# Step 2: Explore the data
print(data.head())
print(data.columns)

# Step 3: Select relevant features
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

# Step 4: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow method to find optimal clusters
inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

plt.figure(figsize=(10,5))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

## Output:
<img width="750" height="198" alt="image" src="https://github.com/user-attachments/assets/e89b2368-9962-4b8d-ab45-7ecc642406c1" />
<img width="935" height="485" alt="image" src="https://github.com/user-attachments/assets/f51e5616-6306-4640-a1b2-e0c4c7e3fc23" />
<img width="1193" height="641" alt="image" src="https://github.com/user-attachments/assets/43355cbb-3fd3-43d6-ada5-9c893360e574" />



## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
