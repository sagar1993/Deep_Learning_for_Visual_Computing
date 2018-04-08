import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import numpy as np


class DenseLayer(object):
    def __init__(self, num_input, num_hidden, num_output, learning_rate, num_iterations):
        
        self.num_iterations = num_iterations
        self.batch_size = 64
        
        self.X = tf.placeholder(tf.float32, shape=(None, num_input))
        self.Y = tf.placeholder(tf.float32, shape=(None))
        
        self.weights = {
            "w1" : tf.Variable(tf.random_normal([num_input, num_hidden])),
            "w2" : tf.Variable(tf.random_normal([num_hidden, num_output]))
        }
        
        self.biases = {
            "b1" : tf.Variable(tf.random_normal([num_hidden])), 
            "b2" : tf.Variable(tf.random_normal([num_output])) 
        }
        
        self.layer1 = tf.nn.sigmoid(tf.matmul(self.X, self.weights["w1"]) + self.biases["b1"])
        self.logits = tf.matmul(self.layer1, self.weights["w2"]) + self.biases["b2"]
        
        prediction = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(prediction, 1)
        correct_pred = tf.equal(self.prediction, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        
        self.sess.run(self.init)
        
        
        
    def fit(self):
        for i in range(self.num_iterations):
            X, Y = mnist.train.next_batch(self.batch_size)
            l, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.X:X, self.Y:Y})
            if i % 1000 == 0:
                print(l)
        
    def predict(self, X, Y):
        prediction = None
        accuracy = None
        prediction, accuracy = self.sess.run([self.prediction, self.accuracy], feed_dict={self.X:X, self.Y:Y})
        return prediction, accuracy
		
		
d = DenseLayer(784, 50, 10, 0.001, 20000)
train_x, train_y = mnist.train.images, mnist.train.labels
d.fit()
test_x, test_y = mnist.test.images, mnist.test.labels
pred, accuracy = d.predict(test_x, test_y)
accuracy

# 0.94999999