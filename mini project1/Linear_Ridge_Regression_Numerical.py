import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def increase_diamentionality(x, n):
    col = np.stack([np.prod(x**d, axis=1) for d in range(1,n)], axis=-1)
    x = np.hstack((x, col))
    one = np.ones((x.shape[0],1))
    X = np.hstack((one, x))
    return X

class LinearRegression(object):
    def __init__(self):
        self.w = None
        self.costs = {}
    
    def predict1(self, x):
        return np.dot(x, self.w)
    
    def predict(self,x):
        one = np.ones((x.shape[0],1))
        X = np.hstack((one, x))
        return np.dot(X, self.w)
    
    def print_weight(self):
        print(self.w)
        
    def score(self, x, y):
        one = np.ones((x.shape[0],1))
        X = np.hstack((one, x))
        prediction = self.predict1(X)
        error = prediction - y
        score = np.sum(error ** 2) * 1.0 / (2*x.shape[0])
        return score 
        
    def fit(self, x, y, learning_rate, number_iteration=1, l2_penalty=0):
        self.w = None
        self.costs = {}
        one = np.ones((x.shape[0],1))
        X = np.hstack((one, x))
        self.w = np.zeros((x.shape[1] + 1,1))
        
        for i in range(number_iteration):
            prediction = self.predict1(X)
            error = prediction - y
            
            gradient = 2.0 * np.dot(X.T, error) / x.shape[0] + 2 * l2_penalty
            
            #self.w -= learning_rate * gradient
            self.w = (1 - learning_rate*l2_penalty) * self.w - learning_rate * gradient
            
            cost = np.sum(error ** 2) * 1.0 / x.shape[0]
            self.costs[i] = cost
            if i % 10 == 0:
                print("Cost at iteration %i is: %f" %(i, cost))
            if i % 100 == 0:
                print("Cost at iteration %i is: %f" %(i, cost))
                
    def plot_line(self, x, y):
        prediction = l.predict(x)
        plt.plot(x, y, '.')
        plt.plot(x, prediction, '-')
        
    def plot_costs(self):
        plt.plot(self.costs.keys(), self.costs.values(), '-')