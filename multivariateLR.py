import pandas as pd
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston_dataset = load_boston()
## build a DataFrame
boston = pd.DataFrame(boston_dataset.data, 
                      columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
X = boston[['RM']]
Y = boston['MEDV']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size = 0.3, 
                                                    random_state=1)
model = LinearRegression()
model.fit(X_train, Y_train)
y_test_predicted = model.predict(X_test)
## data preparation
X2 = boston[['RM', 'LSTAT']]
Y = boston['MEDV']
## train test split
## same random_state to ensure the same splits
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y, 
                                                    test_size = 0.3, 
                                                    random_state=1)
model2 = LinearRegression()
print(model2.fit(X2_train, Y_train))
print(model2.intercept_)
print(model2.coef_)

y_test_predicted2 = model2.predict(X2_test)
print(y_test_predicted2)


#Which model is better? An easy metric for linear regression is the mean squared error (MSE) on the testing data. Better models have lower MSEs.
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, y_test_predicted).round(2))
print(mean_squared_error(Y_test, y_test_predicted2).round(2))
#In general, the more features the model includes the lower the MSE would be. Yet be careful about including too many features. Some features could be random noise, thus hurt the interpretability of the model.

#residual is the difference between the observation and the presiction
