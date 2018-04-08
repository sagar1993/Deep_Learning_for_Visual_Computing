class AutoEncoder(object):
    def __init__(self, num_input, num_hidden, num_output, learning_rate, num_steps, batch_size):
        
        
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size
        
        ## define placeholder
        self.X = tf.placeholder(tf.float32, shape=(None, num_input))
        
        ## define weights and biases
        self.weights = {
            "encoder_w1" : tf.Variable(tf.random_normal([num_input, num_hidden])),
            "decoder_w1" : tf.Variable(tf.random_normal([num_hidden, num_output])),
        }
        
        self.biases = {
            "encoder_b1" : tf.Variable(tf.random_normal([num_hidden])),
            "decoder_b1" : tf.Variable(tf.random_normal([num_output])),
        }
        
        self.e = self.encode(self.X)
        self.d = self.decode(self.e)
        self.l = self.loss(self.X, self.d)
        self.optimize = self.optimizer(self.l)
        
        self.init = tf.global_variables_initializer()

    def encode(self, x):
        return tf.nn.sigmoid(tf.matmul(x, self.weights["encoder_w1"]) + self.biases["encoder_b1"])
    
    def decode(self, x):
        return tf.nn.sigmoid(tf.matmul(x, self.weights["decoder_w1"]) + self.biases["decoder_b1"])
    
    def loss(self, y_true, y_predicted):
        return tf.reduce_mean(tf.pow((y_true - y_predicted),2))
    
    def optimizer(self, loss):
        return tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.l)
        
    def fit(self):
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(self.num_steps):
                batch_x, _ = mnist.train.next_batch(self.batch_size)
                l, _ = sess.run([self.l, self.optimize], feed_dict = {self.X: batch_x})
                
                if i % 1000 == 0:
                    print("loss at iteration %d : %f"% (i,l))
                
    
    def predict(self, data_noise):
        with tf.Session() as sess:
            sess.run([self.decode], feed_dict = {self.X: batch_x})