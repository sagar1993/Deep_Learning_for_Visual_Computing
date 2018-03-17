import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from layers import *

class Config:
    epsilon = 0.0002 
    reg_lambda = 0.01


def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    return x_train, y_train, x_test, y_test
	
def plot(title, x ,y, Label, xLabel, yLabel):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plt.plot(x, y, color='red',label=Label, marker='o')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.legend(loc='best')
    plt.show()	
	
def predict_cnn(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    out, cache_1 = conv_forward(x, W1, b1)
    a1, cache_r_1 = relu_forward(out)
    z, cache_2 = nn_forward(a1,W2,b2)
    a2, cache_r_2 = relu_forward(z)
    z, cache_3 = nn_forward(a2,W3,b3)
    a3 = np.tanh(z)
    exp_scores = np.exp(a3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def predict_maxpool(model, X):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    out, cache_1 = conv_forward(X, W1, b1)
    a1, cache_r_1 = relu_forward(out)
    out, cache_pool = max_pool_forward(a1, 2, 2, 1)
    z, cache_2 = nn_forward(out,W2,b2)
    a2, cache_r_2 = relu_forward(z)
    z, cache_3 = nn_forward(a2,W3,b3)
    a3 = np.tanh(z)
    exp_scores = np.exp(a3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def predict_maxpool_dropout(model, X):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    out, cache_1 = conv_forward(X, W1, b1)
    a1, cache_r_1 = relu_forward(out)
    out, cache_pool = max_pool_forward(a1, 2, 2, 1)
    out, cache_drop = dropout_forward(out, .25, "test", seed=0)
    z, cache_2 = nn_forward(out,W2,b2)
    a2, cache_r_2 = relu_forward(z)
    z, cache_3 = nn_forward(a2,W3,b3)
    a3 = np.tanh(z)
    exp_scores = np.exp(a3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

	
def build_model_cnn(X, y, nn_hdim, num_passes=20, print_loss=False):
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(5,1,3,3) / np.sqrt(27)
    b1 = np.zeros((5,1))
    W2 = np.random.randn(3920, 50) / np.sqrt(50)
    b2 = np.zeros((1, 50))
    W3 = np.random.randn(50, 10) / np.sqrt(10)
    b3 = np.zeros((1, 10))

    model = {}
    loss_l = []

    for i in range(0, num_passes):

        ## CNN        
        out, cache_1 = conv_forward(X, W1, b1)
        a1, cache_r_1 = relu_forward(out)
        z, cache_2 = nn_forward(a1,W2,b2)
        a2, cache_r_2 = relu_forward(z)
        z, cache_3 = nn_forward(a2,W3,b3)
        a3 = np.tanh(z)

        loss, dx = softmax_loss(a3, y)
        dx3, dw3, db3 = nn_backward(dx, cache_3)
        dx3 = relu_backward(dx3, cache_r_2)
        dx2, dw2, db2 = nn_backward(dx3, cache_2)
        dx2 = relu_backward(dx2, cache_r_1);
        dx1, dw1, db1 = conv_backward(dx2, cache_1)

        
        dw3 += Config.reg_lambda * W3
        dw2 += Config.reg_lambda * W2
        dw1 += Config.reg_lambda * W1

        W1 += -Config.epsilon * dw1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dw2
        b2 += -Config.epsilon * db2
        W3 += -Config.epsilon * dw3
        b3 += -Config.epsilon * db3
        
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

        loss_l.append(loss)
        print("Loss after iteration %i: %f" % (i, loss))
        
    return model, loss_l


def build_model_cnn_maxpool(X, y, nn_hdim, num_passes=1, print_loss=False):
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(5,1,3,3) / np.sqrt(27)
    b1 = np.zeros((5,1))
    W2 = np.random.randn(3645, 50) / np.sqrt(50)
    b2 = np.zeros((1, 50))
    W3 = np.random.randn(50, 10) / np.sqrt(10)
    b3 = np.zeros((1, 10))

    model = {}
    loss_l = []

    for i in range(0, num_passes):

        ## CNN        
        out, cache_1 = conv_forward(X, W1, b1)
        a1, cache_r_1 = relu_forward(out)
        out, cache_pool = max_pool_forward(a1, 2, 2, 1)
        z, cache_2 = nn_forward(out,W2,b2)
        a2, cache_r_2 = relu_forward(z)
        z, cache_3 = nn_forward(a2,W3,b3)
        a3 = np.tanh(z)

        loss, dx = softmax_loss(a3, y)
        dx3, dw3, db3 = nn_backward(dx, cache_3)
        dx3 = relu_backward(dx3, cache_r_2)
        dx2, dw2, db2 = nn_backward(dx3, cache_2)
        dx2 = max_pool_backward(dx2, cache_pool)
        dx2 = relu_backward(dx2, cache_r_1);
        dx1, dw1, db1 = conv_backward(dx2, cache_1)
        
        dw3 += Config.reg_lambda * W3
        dw2 += Config.reg_lambda * W2
        dw1 += Config.reg_lambda * W1

        W1 += -Config.epsilon * dw1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dw2
        b2 += -Config.epsilon * db2
        W3 += -Config.epsilon * dw3
        b3 += -Config.epsilon * db3
        
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

        loss_l.append(loss)
        print("Loss after iteration %i: %f" % (i, loss))
        
    return model, loss_l


def build_model_maxpool_dropout(X, y, nn_hdim, num_passes=20, print_loss=False):
    
    num_examples = len(X)
    np.random.seed(0)
    W1 = np.random.randn(5,1,3,3) / np.sqrt(27)
    b1 = np.zeros((5,1))
    W2 = np.random.randn(3645, 50) / np.sqrt(50)
    b2 = np.zeros((1, 50))
    W3 = np.random.randn(50, 10) / np.sqrt(10)
    b3 = np.zeros((1, 10))

    model = {}
    loss_l = []

    for i in range(0, num_passes):

        ## CNN        
        out, cache_1 = conv_forward(X, W1, b1)
        a1, cache_r_1 = relu_forward(out)
        out, cache_pool = max_pool_forward(a1, 2, 2, 1)
        out, cache_drop = dropout_forward(out, .25, "train", seed=0)
        z, cache_2 = nn_forward(out,W2,b2)
        a2, cache_r_2 = relu_forward(z)
        z, cache_3 = nn_forward(a2,W3,b3)
        a3 = np.tanh(z)

        loss, dx = softmax_loss(a3, y)
        dx3, dw3, db3 = nn_backward(dx, cache_3)
        dx3 = relu_backward(dx3, cache_r_2)
        dx2, dw2, db2 = nn_backward(dx3, cache_2)
        dx2 = dropout_backward(dx2, cache_drop)
        dx2 = max_pool_backward(dx2, cache_pool)
        dx2 = relu_backward(dx2, cache_r_1);
        dx1, dw1, db1 = conv_backward(dx2, cache_1)

        dw3 += Config.reg_lambda * W3
        dw2 += Config.reg_lambda * W2
        dw1 += Config.reg_lambda * W1

        W1 += -Config.epsilon * dw1
        b1 += -Config.epsilon * db1
        W2 += -Config.epsilon * dw2
        b2 += -Config.epsilon * db2
        W3 += -Config.epsilon * dw3
        b3 += -Config.epsilon * db3

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

        loss_l.append(loss)
        print("Loss after iteration %i: %f" % (i, loss))

    return model, loss_l
