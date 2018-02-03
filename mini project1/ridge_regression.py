from numpy import array, dot, transpose
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
%matplotlib inline


def increase_diamentionality(x, n):
    shape = x.shape[0]
    col = np.stack([np.prod(x**d, axis=1) for d in range(2,n+1)], axis=-1)
    x = np.column_stack((x, col))
    return x

def ridge_regression(x_train, y_train, lam):
	
	X = np.array(x_train)
	ones = np.ones(len(X))
	X = np.column_stack((ones,X))
	y = np.array(y_train)
	
	Xt = transpose(X)
	lambda_identity = np.identity(len(Xt)) * lam*lam
	theInverse = np.linalg.inv(np.dot(Xt, X)+lambda_identity)
	w = dot(dot(theInverse, Xt), y)
	
	return w
	
def predict_ridge(x,w):
	X = np.array(x)
	ones = np.ones(len(X))
	X = np.column_stack((ones,X))
	return np.dot(X,w)
	
def cost_ridge(x, y, w, lam):
    
    X = np.array(x)
    ones = np.ones(len(X))
    X = np.column_stack((ones,x))
    val = y-np.dot(X,w)
    m = X.shape[0]
    reg = lam * lam * np.dot(np.transpose(w), w)
    return (np.dot(np.transpose(val),val) + reg) * 1/m
	
def error_ridge(x, y, w, lam):
    
    X = np.array(x)
    ones = np.ones(len(X))
    X = np.column_stack((ones,x))
    
    val = y-np.dot(X,w)
    m = X.shape[0]
    return (np.dot(np.transpose(val),val)) * 1/m
	
	
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
X = increase_diamentionality(data[:,0,None], 15)
y = data[:,1,None]
	
kf = KFold(n_splits=5,shuffle=True)
train_error = {}
cv_error = {}
for alpha in [.01, .05, .1, .5, 1, 5, 10]:
    print(alpha)
    train_errors = []
    cv_errors = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        w = ridge_regression(X_train, y_train, alpha)
        train_e = error_ridge(X_train, y_train, w, alpha)
        cv_e = error_ridge(X_test, y_test, w, alpha)
        
        train_errors.append(train_e)
        cv_errors.append(cv_e)
    train_error[alpha] = sum(train_errors)[0][0] / float(len(train_errors))
    cv_error[alpha] = sum(cv_errors)[0][0] / float(len(cv_errors))
	
plot("Ridge Regression Training Error vs Lambda", train_error.keys(), train_error.values(), "Training Error", 'lambda', 'training error')

plot("Ridge Regression Cross Validation Error vs Lambda", cv_error.keys(), cv_error.values(), "CV Error", 'lambda', 'cv error')



predict = predict_ridge(X,ridge_regression(X, data[:,1,None], .5))

plot1("Line fit to data Ridge Regression", data[:,0,None], data[:,1,None], 'y', data[:,0,None], predict, 'y_predicted', 'x', 'y')
