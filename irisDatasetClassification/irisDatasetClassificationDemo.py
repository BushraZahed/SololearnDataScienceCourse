import pandas as pd
iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
#inspect the dimensions 
print(iris.shape)
#inspect the first 5 rows
print(iris.head(n=5))
#The column id is the row index, not really informative, 
# so we can drop it from the dataset using drop() function:
iris.drop('id',axis=1,inplace=True)
print(iris.head(n=10))
#summary statistics
print(iris.describe())
#All four features are numeric, each with different ranges. There are no missing 
# values in any of the columns. Therefore, this is a clean dataset.
#The ranges of attributes are still of similar magnitude, thus we will skip standardization. However, standardizing attributes such that each has a mean of zero and a standard deviation of one, can be an important preprocessing step for
#  many machine learning algorithms. This is also called feature scaling

## The data set contains 3 classes of 50 instances each. 
print(iris.groupby('species').size())
# or similar
print(iris['species'].value_counts()) #The method value_counts() is a great utility for quickly understanding the distribution of the data. 
#Iris is a balanced dataset as the data points for each class are evenly distributed.
#univariate plot
import matplotlib.pyplot as plt
plt.style.use("ggplot")
iris.hist()
plt.show()
# sepal_len and sepal_wd have a normal gaussian distribution (beautiful symmetric bell shape)
#the length of petals is not normal. Its plot shows two modes, one peak happening near 0 and 
# the other around 5. Less patterns were observed for the petal width.
#multivariate plot :To see the interactions between attributes we use scatter plots.
# build a dict mapping species to an integer code
inv_name_dict = {'iris-setosa': 0,
'iris-versicolor': 1,
'iris-virginica': 2}

# build integer color code 0/1/2
colors = [inv_name_dict[item] for item in iris['species']]
# scatter plot
scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c = colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
## add legend
plt.legend(handles=scatter.legend_elements()[0],
labels = inv_name_dict.keys())
plt.savefig("plot.png")
plt.show()
#Using sepal_length and sepal_width features, we can distinguish iris-setosa from others; separating iris-versicolor from iris-virginica is harder because of the overlap as seen by the green and yellow datapoints.

###Similarly, between petal length and width:
scatter = plt.scatter(iris['petal_len'], iris['petal_wd'], c = colors)
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm")
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())
plt.savefig("plot.png")
plt.show()

## Data preparation
#we found out earlier petal length and width are the most effective features to separate the species
#defining the features 
X = iris[['petal_len', 'petal_wd']]
y = iris['species']

#splitting up training and test set to mimic the unknown data model that will be presented in the future
from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
#the split was stratified by label (y). This is done to ensure that the distribution of labels remains similar in both train and test
print("Stratified by y")
print(y_train.value_counts())
print(y_test.value_counts())

#Modeling : now we will build and train the model knn
from sklearn.neighbors import KNeighborsClassifier
#now creating an instance of the model
knn = KNeighborsClassifier(n_neighbors=5)
#Note that the only parameter we need to set in this problem is n_neighbors, or k as in knn. We set k to be 5 by random choice.
#Use the data X_train and y_train to train the model:
knn.fit(X_train,y_train) #It outputs the trained model. We use most the default values for the parameters, e.g., metric = 'minkowski' and p = 2 together defines that the distance is euclidean distance.

#label prediction : We are trying to predict the species of iris using given features in feature matrix X.
y_pred = knn.predict(X_test) #predict() returns an array of predicted class labels for the predictor data.
#reviewing the first five predictions
print(y_pred[:5]) #Each prediction is a species of iris and stored in a 1darray.
#predict() returns an array of predicted class labels for the predictor data.
#probability prediction : 'predict_prob'. Instead of splitting the label, it outputs the probability for the target in array form. 
#the predicted probabilities are for the 11th and 12th flowers:
y_pred_prob = knn.predict_proba(X_test)
print(y_pred_prob[10:12]) # result = the probability of the 11th flower being predicted an iris-setosa is 1, an iris-versicolor and an iris-virginica are both 0. For the next flower, there is a 20% chance that it would be classified as iris-versicolor but 80% chance to be iris-virginica.
print(y_pred[10:12]) #result=of the five nearest neighbours of the 12th flower in the testing set, 1 is an iris-versicolor, the rest 4 are iris-virginica.
#In classification tasks, soft prediction returns the predicted probabilities of data points belonging to each of the classes while hard prediction outputs the labels only.
#In classification the most straightforward metric is accuracy. It calculates the proportion of data points whose predicted labels exactly match the observed labels.

((y_pred == y_test.values).sum())
print(y_test.size)
#The classifier made one mistake. Thus, the accuracy is 44/45:
print((y_pred == y_test.values).sum()/y_test.size )
#same as
print(knn.score(X_test,y_test))

#Classification accuracy alone can be misleading if there is an unequal number of observations in each class or if there are more than two classes in the dataset. Calculating a confusion matrix will provide a better idea of what the classification is getting right and what types of errors it is making.
#a confusion matrix? It is a summary of the counts of correct and incorrect predictions, broken down by each class.
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues);
plt.savefig("plot.png")
plt.show()
#A confusion matrix is a table that is often used to describe the 
# performance of a classification model (or "classifier") on 
# a set of test data for which the true values are known.
#Previously we made train-test split before fitting the model so that we can report the model performance on the test data. This is a simple kind of cross validation technique, also known as the holdout method. However, the split is random, as a result, model performance can be sensitive to how the data is split. To overcome this, we introduce k-fold cross validation.
#K-fold cross validation :n k fold cross validation, the data is divided into k subsets. Then the holdout method is repeated k times, such that each time, one of the k subsets is used as the test set and the other k-1 subsets are combined to train the model. Then the accuracy is averaged over k trials to provide total effectiveness of the model. In this way, all data points are used; and there are more metrics so we don’t rely on one test data for model performance evaluation.
from sklearn.model_selection import cross_val_score
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#now fitting a 2nn model
#train model with 5-fold cv
cv_scores = cross_val_score(knn_cv,X,y,cv=5)
#each of the holdout set contains 20% of the original data, print each cv score(accuracy)
print(cv_scores)
# due to the random assignments, the accuracies on the holdsets fluctuates from 0.9 to 1
#average them
print(cv_scores.mean()) 
#As a general rule, 5-fold or 10-fold cross validation is preferred; 
# but there is no formal rule. 

print(cv_scores.shape)

#When we built our first knn model, we set the hyperparameter k to 5, and then to 3 later in k-fold cross validation; random choices really. What is the best k? Finding the optimal k is called tuning the hyperparameter. A handy tool is grid search.
from sklearn.model_selection import GridSearchCV
import numpy as np
# create new a knn model
knn2 = KNeighborsClassifier()
# create a dict of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)
#To check the top performing n_neighbors value:
print(knn_gscv.best_params_)
#accuracy of the model when k = 4
print(knn_gscv.best_score_)
#now buildding the final model as the best score increases by 1%
knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X,y)
y_pred = knn_final.predict(X)
print(knn_final.score(X,y))

#We can report that our final model, 4nn, has an accuracy of 97.3% 
# in predicting the species of iris!
#The techniques of k-fold cross validation and tuning parameters with grid search is applicable to both classification and regression problems.



