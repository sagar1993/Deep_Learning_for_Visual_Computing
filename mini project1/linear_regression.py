from numpy import array, dot, transpose
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline



def linear_regression(x_train, y_train):
	
	X = np.array(x_train)
	ones = np.ones(len(X))
	X = np.column_stack((ones,X))
	y = np.array(y_train)
	
	Xt = transpose(X)
	product = dot(Xt, X)
	theInverse = inv(product)
	w = dot(dot(theInverse, Xt), y)
	
	cost = cost_linear(X, y, w)
	print("rmse" + str(cost[0][0]))
	predictions = np.dot(X,w)
	
	return predictions

def cost_linear(x, y, w):
	val = y-np.dot(x,w)
	m = x.shape[0]
	return np.dot(np.transpose(val),val) * 1/m
	
def plot1(title, x ,y, Label, x1, y1, Label1, xLabel, yLabel):
	fig = plt.figure(figsize=(10,7))
	ax = fig.add_subplot(111)
	ax.set_title(title)
	ax.scatter(x, y, label=Label)
	plt.plot(x1, y1 ,color='red',label=Label1)
	ax.set_xlabel(xLabel)
	ax.set_ylabel(yLabel)
	ax.legend(loc='best')
	plt.show()

def plot(title, x ,y, Label, xLabel, yLabel):
	fig = plt.figure(figsize=(10,7))
	ax = fig.add_subplot(111)
	ax.set_title(title)
	plt.plot(x, y, color='red',label=Label)
	#ax.scatter(x, y, label=Label)
	ax.set_xlabel(xLabel)
	ax.set_ylabel(yLabel)
	ax.legend(loc='best')
	plt.show()
	
data = np.load('linRegData.npy')

predict = linear_regression(data[:,0,None], data[:,1,None])

plot1("Line fit to data Linear Regression", data[:,0,None], data[:,1,None], 'y', data[:,0,None], predict, 'y_predicted', 'x', 'y')

