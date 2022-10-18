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
