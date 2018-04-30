import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import SupervisedConfig

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class SupervisedAdverserialAutoencoder(object):
    
    def __init__(self):
        
        self.x_ip = tf.placeholder(dtype=tf.float32, shape=[SupervisedConfig.batch_size, SupervisedConfig.input_dim], name='Input')
        self.y_ip = tf.placeholder(dtype=tf.float32, shape=[SupervisedConfig.batch_size, SupervisedConfig.n_labels], name='Labels')
        self.x_op = tf.placeholder(dtype=tf.float32, shape=[SupervisedConfig.batch_size, SupervisedConfig.input_dim], name='Target')
        self.r_distribution = tf.placeholder(dtype=tf.float32, shape=[SupervisedConfig.batch_size, SupervisedConfig.z_dim], name='r_distribution')
        self.m_decoder_ip = tf.placeholder(dtype=tf.float32, shape=[1, SupervisedConfig.z_dim + SupervisedConfig.n_labels], name='Decoder_input')
        
        
        with tf.variable_scope(tf.get_variable_scope()):
            self.encoder_output = self.encoder(self.x_ip)
            self.decoder_input = tf.concat([self.y_ip, self.encoder_output], 1)
            self.decoder_output = self.decoder(self.decoder_input)

        with tf.variable_scope(tf.get_variable_scope()):
            self.d_real = self.discriminator(self.r_distribution)
            self.d_fake = self.discriminator(self.encoder_output, reuse=True)

        with tf.variable_scope(tf.get_variable_scope()):
            self.decoder_op = self.decoder(self.m_decoder_ip, reuse=True)

        self.autoencoder_loss = tf.reduce_mean(tf.square(self.x_op - self.decoder_output))

        self.dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_real), logits=self.d_real))
        self.dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_fake), logits=self.d_fake))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake), logits=self.d_fake))

        self.all_variables = tf.trainable_variables()
        self.dc_var = [var for var in self.all_variables if 'dc_' in var.name]
        self.en_var = [var for var in self.all_variables if 'e_' in var.name]

        self.autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=SupervisedConfig.learning_rate, beta1=SupervisedConfig.beta1).minimize(self.autoencoder_loss)
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=SupervisedConfig.learning_rate, beta1=SupervisedConfig.beta1).minimize(self.dc_loss, var_list=self.dc_var)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=SupervisedConfig.learning_rate, beta1=SupervisedConfig.beta1).minimize(self.generator_loss, var_list=self.en_var)

        self.init = tf.global_variables_initializer()
        
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()


    def Dense(self, x, h1, h2, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            weights = tf.get_variable("weights", shape=[h1, h2], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
            bias = tf.get_variable("bias", shape=[h2], initializer=tf.constant_initializer(0.0))
            out = tf.add(tf.matmul(x, weights), bias, name='matmul')
            return out

    def encoder(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Encoder'):
            e_Dense_1 = tf.nn.relu(self.Dense(x, SupervisedConfig.input_dim, SupervisedConfig.n_l1, 'e_Dense_1'))
            e_Dense_2 = tf.nn.relu(self.Dense(e_Dense_1, SupervisedConfig.n_l1, SupervisedConfig.n_l2, 'e_Dense_2'))
            latent_variable = self.Dense(e_Dense_2, SupervisedConfig.n_l2, SupervisedConfig.z_dim, 'e_latent_variable')
            return latent_variable

    def decoder(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Decoder'):
            d_Dense_1 = tf.nn.relu(self.Dense(x, SupervisedConfig.z_dim + SupervisedConfig.n_labels, SupervisedConfig.n_l2, 'd_Dense_1'))
            d_Dense_2 = tf.nn.relu(self.Dense(d_Dense_1, SupervisedConfig.n_l2, SupervisedConfig.n_l1, 'd_Dense_2'))
            output = tf.nn.sigmoid(self.Dense(d_Dense_2, SupervisedConfig.n_l1, SupervisedConfig.input_dim, 'd_output'))
            return output

    def discriminator(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Discriminator'):
            dc_Dense1 = tf.nn.relu(self.Dense(x, SupervisedConfig.z_dim, SupervisedConfig.n_l1, name='dc_Dense1'))
            dc_Dense2 = tf.nn.relu(self.Dense(dc_Dense1, SupervisedConfig.n_l1, SupervisedConfig.n_l2, name='dc_den2'))
            output = self.Dense(dc_Dense2, SupervisedConfig.n_l2, 1, name='dc_output')
            return output
        
    def fit(self):
        step = 0
        tensorboard_path = './tensorboard/'
        saved_model_path = './model/'
        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=self.sess.graph)
        for i in range(SupervisedConfig.n_epochs):
            n_batches = int(mnist.train.num_examples / SupervisedConfig.batch_size)
            print("Epoch {}/{}".format(i, SupervisedConfig.n_epochs))
            for batch in range(1, n_batches + 1):
                z_real_dist = np.random.randn(SupervisedConfig.batch_size, SupervisedConfig.z_dim) * 5.
                batch_x, batch_y = mnist.train.next_batch(SupervisedConfig.batch_size)
                self.sess.run(self.autoencoder_optimizer, feed_dict={self.x_ip: batch_x, self.x_op: batch_x, self.y_ip: batch_y})
                self.sess.run(self.discriminator_optimizer, feed_dict={self.x_ip: batch_x, self.x_op: batch_x, self.r_distribution: z_real_dist})
                self.sess.run(self.generator_optimizer, feed_dict={self.x_ip: batch_x})
                if batch % 50 == 0:
                    a_l, d_l, g_l = self.sess.run([self.autoencoder_loss, self.dc_loss, self.generator_loss], feed_dict={self.x_ip: batch_x, self.x_op: batch_x, self.r_distribution: z_real_dist, self.y_ip: batch_y})
                    print("Epoch: {}, iteration: {}".format(i, batch))
                    print("Autoencoder Loss: {}".format(a_l))
                    print("Discriminator Loss: {}".format(d_l))
                    print("Generator Loss: {}".format(g_l))
                step += 1
            
            self.saver.save(self.sess, save_path=saved_model_path, global_step=step)
			
    def generate(self, latent, label):
        latent, label = np.reshape(latent, (1, SupervisedConfig.z_dim)), np.reshape(label, (1, SupervisedConfig.n_labels))
        ip = np.concatenate((label, latent), 1)
        return self.sess.run(op, feed_dict={self.m_decoder_ip: ip})
    