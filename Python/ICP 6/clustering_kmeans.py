import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')
x = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]
print(x.shape, y.shape)

nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


modified_dataset = dataset.fillna(dataset.mean())

nulls1 = pd.DataFrame(modified_dataset.isnull().sum().sort_values(ascending=False)[:25])
modified_dataset.to_csv(index=False)
nulls1.columns = ['Null Count']
nulls1.index.name = 'Feature'
print(nulls1)

modified_dataset.to_csv(index=False)
x = modified_dataset.iloc[:,1:-1]
y = modified_dataset.iloc[:,-1]

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x)
from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print("silhouette_score:", score)

# elbow method to know the number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# Apply transform to both the training set and the test set.
pca = PCA(3)
x_pca = pca.fit_transform(X_scaled)
print(x_pca)

from sklearn.cluster import KMeans
mclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=mclusters)
km.fit(x_pca)

# predict the cluster for each data point
y_cluster_kmeans = km.predict(x_pca)
score = metrics.silhouette_score(x_pca, y_cluster_kmeans)
print("silhouette_score:", score)