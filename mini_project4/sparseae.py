import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import numpy as np

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
        
        for i in range(self.num_steps):
            batch_x, _ = mnist.train.next_batch(self.batch_size)
            l, _ = self.sess.run([self.l, self.optimize], feed_dict = {self.X: batch_x})

            if i % 10000 == 0:
                print("loss at iteration %d : %f"% (i,l))
        weight = self.sess.run(self.weights["encoder_w1"])
        self.images = weight.T.reshape(200, 28, 28)

        weight1 = self.sess.run(self.weights["decoder_w1"])
        self.images1 = weight.reshape(28, 28, 200).T

        return weight
            
    def visualize(self):
        print("Layer 1")
        n = 10
        canvas = np.empty((28 * n, 28 * n))
        k = 0
        for i in range(10):
            for j in range(10):
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = self.images[k]
                k += 1
        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()

        print("Layer 2")
        canvas = np.empty((28 * n, 28 * n))
        k = 0
        for i in range(10):
            for j in range(10):
                canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = self.images1[k]
                k += 1
        plt.figure(figsize=(n, n))
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()
            
    def generate(self, data_noise):
        return self.sess.run([self.decode], feed_dict = {self.X: data_noise})

for sp in [0.01, 0.1, 0.5, 0.8]:
    s = SparseAutoEncoder(784, 200, 784, 0.1, 401, 64, sp, 0.001, 3)
    s.fit()
    print("## sparsity parameter :" + str(sp))
    s.visualize()