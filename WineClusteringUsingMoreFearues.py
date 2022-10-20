import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)
X = wine
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
# compute the mean and std to be used later for scaling
scale.fit(X)

X_scaled = scale.transform(X)

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
# calculate distortion for a range of number of cluster
inertia = []
for i in np.arange(1, 11):
    km = KMeans(
        n_clusters=i 
        )
    km.fit(X_scaled) 
    inertia.append(km.inertia_)
plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title("all features")
plt.savefig("kmeansmorefeatures.png")
plt.show()
#Similarly we spot that the inertia no longer decreases as rapidly after k = 3. We then finalize the model by setting n_clusters = 3 and obtain the predictions.
k_opt = 3
kmeans = KMeans(k_opt)
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)
print(y_pred)