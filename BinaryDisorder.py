#Confusion matrix of binary classification.For binary classifications, 
# a confusion matrix is a two-by-two matrix to visualize 
# the performance of an algorithm. Each row of the matrix 
# represents the instances in a predicted class while each 
# column represents the instances in an actual class.
#Task :Given two lists of 1s and 0s (1 represents the true label, and 0 represents the false false) of the same length, output a 2darrary of counts, each cell is defined as follows

#Top left: Predicted true and actually true (True positive)
#Top right: Predicted true but actually false (False positive)
#Bottom left: Predicted false but actually true (False negative)
#Bottom right: Predicted false and actually false (True negative)

##solution##
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = [int(x) for x in input().split()]
y_pred =  [int(x) for x in input().split()]

y_true = np.array(y_true)
y_pred = np.array(y_pred)

confusion = confusion_matrix(y_pred,y_true,labels=[1,0])
print(np.array(confusion, dtype='f'))