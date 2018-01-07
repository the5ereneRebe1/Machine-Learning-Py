import numpy as np
from scipy.spatial import distance 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def euc(a,b):
	 return distance.euclidean(a,b)

class OneNeighborsClassifier():
	def fit(self, X_train, y_train):
		self.X_train= X_train
		self.y_train= y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	def closest(self, row):
		best_distance =euc(row, self.X_train[0])
		best_index=0
		for i in range(1,len(self.X_train)):
			dist= euc(row,self.X_train[i])
			if(dist<best_distance):
				best_distance=dist
				best_index=i
		return self.y_train[best_index]



iris=load_iris()
X,y =iris.data,iris.target
#Split the input data into test and train data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#Train our classifier

my_classifier=OneNeighborsClassifier()
my_classifier.fit(X_train,y_train)
predicitons= my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predicitons))