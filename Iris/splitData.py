import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris=load_iris()
X,y =iris.data,iris.target
#Split the input data into test and train data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#Train our classifier

from sklearn.neighbors import KNeighborsClassifier
my_classifier=KNeighborsClassifier()
my_classifier.fit(X_train,y_train)
predicitons= my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predicitons))