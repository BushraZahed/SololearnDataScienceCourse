#In this module, we analyze the result of a chemical analysis of
#  wines grown in a particular region in Italy. And the goal is to 
# try to group similar observations together and determine the number 
# of possible clusters. This would help us make predictions and reduce 
# dimensionality. As we will see there are 13 features for each wine,
#  and if we could group all the wines into, say 3 groups, then it is 
# reducing the 13-dimensional space to a 3-dimensional space. More 
# specifically we can represent each of our original data points in 
# terms of how far it is from each of these three cluster centers.
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)
print(wine.shape)
print(wine.columns)
#For the ease of display, we show the basic statistics of the first 3 features:
print(wine.iloc[:,:3].describe())
print(wine.iloc[:,:3].info())

#The summary statistics provide some of the information, while visualization offers a more direct view showing the distribution and the relationship between features.
#plotting function to display histograms along the diagonal and the scatter plots for every pair of attributes off the diagonal, 'scatter_matrix', for the ease of display, let’s show just two features
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
scatter_matrix(wine.iloc[:,[0,5]])
plt.savefig('wineplot.png')
plt.show()

#After examining all the pairs of scatter plot, we pick two features to better illustrate the algorithm: alcohol and total_phenols, whose scatterplot also suggests three subclusters

X = wine[['alcohol', 'total_phenols']] #using two features

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
# compute the mean and std to be used later for scaling
scale.fit(X)

X_scaled = scale.transform(X)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

#KMEANS modeling

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)
print(y_pred)
#There are 60 wines in cluster 0, 65 in cluster 1, and 53 in cluster 2.

#To inspect the coordinates of the three centroids

print(kmeans.cluster_centers_)

#Visualization

#plot the scaled data
plt.scatter(X_scaled[:,0],
X_scaled[:,1],
c= y_pred)
# identify the centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
kmeans.cluster_centers_[:, 1],
marker="*",
s = 250,
c = [0,1,2],
edgecolors='k')
plt.xlabel('alcohol'); plt.ylabel('total phenols')
plt.title('k-means (k=3)')
plt.savefig("kmeans_plot.png")
plt.show()
#Result = The stars are the centroids. K-means divides wines into three groups: low alcohol but high total phenols (upper right in green), high alcohol and high total phenols (upper left in yellow), and low total phenols (bottom in purple). For any new wine with the chemical report on alcohol and total phenols, we now can classify it based on its distance to each of the centroids. 
#Suppose that there is new wine with alcohol at 13 and total phenols at 2.5, let’s predict which cluster the model will assign the new wine to.
#First we need to put the new data into a 2d array:

X_new = np.array([[13,2.5]])
#standardize the new data
X_new_scaled = scale.transform(X_new)
print(X_new_scaled)

#predict the  cluster
print(kmeans.predict(X_new_scaled))
#One major shortcoming of k-means is that the random initial guess for the centroids can result in bad clustering, and k-means++ algorithm addresses this obstacle by specifying a procedure to initialize the centroids before proceeding with the standard k-means algorithm. In scikit-learn, the initialization mechanism is set to k-means++, by default.
#So which one should we choose, 2, or 3, or 4 for the wines?
#Intuitively, k-means problem partitions n data points into k tight sets such that the data points are closer to each other than to the data points in the other clusters. And the tightness can be measured as the sum of squares of the distance from data point to its nearest centroid, or inertia. In scikit-learn, it is stored as inertia_, e.g. when k = 2, the distortion is 185:
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)
print(kmeans.inertia_)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
print(kmeans.inertia_)

#plotting the intertia for different values of k
# calculate distortion for a range of number of cluster
inertia = []
for i in np.arange(1, 11):
    km = KMeans(
        n_clusters=i
    )
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# plot
plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig("kmeansinertiaplot.png")
plt.show() #As the plot shows, the inertia decreases as the number of clusters increases. The optimal k should be where the inertia no longer decreases as rapidly.
#For example, k=3 seems to be optimal, as we increase the number of clusters from 3 to 4, the decrease 
# in inertia slows down significantly, compared to that from 2 to 3.
#  This approach is called elbow method. One single inertia alone is not suitable to determine the optimal k because the larger k is, the lower the inertia will be.

