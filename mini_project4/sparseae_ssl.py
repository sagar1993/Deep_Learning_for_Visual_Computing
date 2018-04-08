import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import numpy as np


## ------------------------------------------ sparse auto encoder with softmax layer ------------------------------------------ ##

class SparseAutoEncoder(object):
    def __init__(self, num_input, num_hidden, num_output, learning_rate, num_steps, batch_size, sparsity_parameter, weight_decay, sparsity_coef):
        
        
        """
        num_input 
        num_hidden 
        num_output 
        learning_rate 
        num_steps 
        batch_size 
        sparsity_parameter 
        weight_decay 
        sparsity_coef
        """
        
        self.sparsity_coef = sparsity_coef
        self.sparsity_parameter = sparsity_parameter
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        
        ## define placeholder
        self.X = tf.placeholder(tf.float32, shape=(None, num_input))
        
        ## define weights and biases
        self.weights = {
            "encoder_w1" : tf.Variable(tf.random_normal([num_input, num_hidden])),
            "decoder_w1" : tf.Variable(tf.random_normal([num_hidden, num_output]))
        }
        
        self.biases = {
            "encoder_b1" : tf.Variable(tf.random_normal([num_hidden])),
            "decoder_b1" : tf.Variable(tf.random_normal([num_output]))
        }
        
        self.e = self.encode(self.X)
        self.d = self.decode(self.e)
        self.l = self.loss(self.X, self.d, self.e)
        self.optimize = self.optimizer(self.l)
        
        self.init = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        self.sess.run(self.init)

    def encode(self, x):
        return tf.nn.sigmoid(tf.matmul(x, self.weights["encoder_w1"]) + self.biases["encoder_b1"])
    
    def decode(self, x):
        return tf.nn.sigmoid(tf.matmul(x, self.weights["decoder_w1"]) + self.biases["decoder_b1"])
    
    def kl_divergence(self, p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1-p) * tf.log(1-p) - (1-p) * tf.log(1-p_hat)
        
    def loss(self, y_true, y_predicted, p):
        p_hat = tf.reduce_mean(p, axis=0)
        return tf.reduce_mean(tf.pow((y_true - y_predicted),2)) + 0.5 * self.weight_decay * tf.nn.l2_loss(self.weights["encoder_w1"]) + 0.5 * self.weight_decay * tf.nn.l2_loss(self.weights["decoder_w1"]) + self.sparsity_coef * tf.reduce_sum(self.kl_divergence(self.sparsity_parameter, p_hat))
    
    def optimizer(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.l)
        
    def fit(self):
        
        batch_x, batch_y = mnist.train.images[:1000], mnist.train.labels[:1000]
        for i in range (self.num_steps):
            l, _ = self.sess.run([self.l, self.optimize], feed_dict = {self.X: batch_x})
            print("sparse loss at iteration %d : %f"% (i,l))
            
    def getEncode(self, x):
        e = None
        e = self.sess.run([self.e], feed_dict = {self.X: x})
        return e

class SoftMaxLastLayer(object):
    def __init__(self, num_input, num_output, learning_rate, num_steps):
        
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.s = SparseAutoEncoder(784, 200, 784, learning_rate, num_steps, 64, 0.01, 0.001, 3)
        
        self.X = tf.placeholder(tf.float32, shape=(None, num_input))
        self.Y = tf.placeholder(tf.float32, shape=(None, num_output))
        
        self.weights = {
            "w1" : tf.Variable(tf.random_normal([num_input, num_output]))
        }
        
        self.biases = {
            "b1" : tf.Variable(tf.random_normal([num_output])),
        }
        
        logits = tf.add(tf.matmul(self.X, self.weights["w1"]), self.biases["b1"])
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        
        
        pred = tf.nn.softmax(logits)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        self.init = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        self.sess.run(self.init)
        
    def fit(self):
        self.s.fit()
        batch_x, batch_y = mnist.train.images[:1000], mnist.train.labels[:1000]
        for i in range (self.num_steps):
            data = self.s.getEncode(batch_x)
            l, _ = self.sess.run([self.loss, self.optimize], feed_dict = {self.X: data[0], self.Y: batch_y})
            print("loss at iteration %d : %f"% (i,l))
                
            
    def get_accuracy(self):
        batch_x, batch_y = mnist.test.images, mnist.test.labels
        data = self.s.getEncode(batch_x)
        accuracy = self.sess.run([self.accuracy], feed_dict = {self.X: data[0], self.Y: batch_y})
        return accuracy[0]
		
		

## ------------------------------------------ fully connected 3 layer nn ------------------------------------------ ##		
		
		
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
        X, Y = mnist.train.images[:1000], mnist.train.labels[:1000]
        for i in range(self.num_iterations):
            l, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.X:X, self.Y:Y})
            if i % 1000 == 0:
                print(l)
        
    def predict(self, X, Y):
        prediction = None
        accuracy = None
        prediction, accuracy = self.sess.run([self.prediction, self.accuracy], feed_dict={self.X:X, self.Y:Y})
        return prediction, accuracy
		
		
		
		
print("## ------------------------------------------ fully connected 3 layer nn ------------------------------------------ ##")
		
d = DenseLayer(784, 200, 10, 0.1, 401)
d.fit()
test_x, test_y = mnist.test.images, mnist.test.labels
pred, accuracy = d.predict(test_x, test_y)
print("accuracy: " + str(accuracy*100))

print("## ------------------------------------------ fully connected 3 layer nn end ------------------------------------------ ##")


print("## ------------------------------------------ sparse auto encoder with softmax layer ------------------------------------------ ##")

s = SoftMaxLastLayer(200,10,0.1, 401)
s.fit()
accuracy = s.get_accuracy()
print("accuracy: " + str(accuracy*100))

print("## ------------------------------------------ sparse auto encoder with softmax layer end ------------------------------------------ ##")