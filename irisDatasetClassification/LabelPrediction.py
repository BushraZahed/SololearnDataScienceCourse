import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
iris.drop('id',axis=1,inplace=True)
X = iris[['petal_len','petal_wd']]
y = iris['species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1, stratify = y)
from sklearn.model_selection import GridSearchCV
knn2 = KNeighborsClassifier()
# create new a knn model
knn2 = KNeighborsClassifier()
# create a dict of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)

knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X, y)
new_data = np.array([3.76,1.20]) #1d array
new_data = new_data.reshape(1,-1) # we need to convert 1d array  to dataframe
#if we feed it to the model, now we are ready for label prediciton
print(knn_final.predict(new_data))
new_data2 = np.array([[3.76,1.20]]) #2d array
#if we feed it to the model, now we are ready for label prediciton
print(knn_final.predict(new_data2))
#Model.predict can also take a 2D list. For example, 
# knn_final.predict([[3.76, 1.2]]) will output the same
#  result as shown in the lesson.

new_data3 = np.array([[3.76, 1.2], [5.25, 1.2], [1.58, 1.2]])
print(knn_final.predict(new_data3))


