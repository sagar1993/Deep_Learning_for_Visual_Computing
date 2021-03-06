import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import SemiSupervisedAdverserialAutoencoderConfig as Config

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class SemiSupervisedAdverserialAutoencoder(object):
    def __init__(self):
        
        self.x_ip = tf.placeholder(dtype=tf.float32, shape=[Config.batch_size, Config.input_dim], name='ip')
        self.x_ip_l = tf.placeholder(dtype=tf.float32, shape=[Config.batch_size, Config.input_dim], name='ip_label')
        self.y_ip_l = tf.placeholder(dtype=tf.float32, shape=[Config.batch_size, Config.n_labels], name='labels_op')
        self.x_op = tf.placeholder(dtype=tf.float32, shape=[Config.batch_size, Config.input_dim], name='op')
        self.r_distribution = tf.placeholder(dtype=tf.float32, shape=[Config.batch_size, Config.z_dim], name='r_distribution')
        self.cat_distribution = tf.placeholder(dtype=tf.float32, shape=[Config.batch_size, Config.n_labels],name='cat_distribution')
        self.m_decoder_ip = tf.placeholder(dtype=tf.float32, shape=[1, Config.z_dim + Config.n_labels], name='decoder_ip')
        
        with tf.variable_scope(tf.get_variable_scope()):
            self.encoder_op_l, self.encoder_output_latent = self.encoder(self.x_ip)
            self.decoder_input = tf.concat([self.encoder_op_l, self.encoder_output_latent], 1)
            self.decoder_output = self.decoder(self.decoder_input)

        with tf.variable_scope(tf.get_variable_scope()):
            self.d_g_real = self.discriminator_gauss(self.r_distribution)
            self.d_g_fake = self.discriminator_gauss(self.encoder_output_latent, reuse=True)

        with tf.variable_scope(tf.get_variable_scope()):
            self.d_c_real = self.discriminator_categorical(self.cat_distribution)
            self.d_c_fake = self.discriminator_categorical(self.encoder_op_l, reuse=True)

        with tf.variable_scope(tf.get_variable_scope()):
            self.encoder_op_l, self._ = self.encoder(self.x_ip_l, reuse=True)

        with tf.variable_scope(tf.get_variable_scope()):
            self.decoder_image = self.decoder(self.m_decoder_ip, reuse=True)

        self.prediction = tf.equal(tf.argmax(self.encoder_op_l, 1), tf.argmax(self.y_ip_l, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction, tf.float32))

        self.autoencoder_loss = tf.reduce_mean(tf.square(self.x_op - self.decoder_output))

        self.dc_g_l_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_g_real), logits=self.d_g_real))
        self.dc_g_l_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_g_fake), logits=self.d_g_fake))
        self.dc_g_l = self.dc_g_l_fake + self.dc_g_l_real

        self.dc_c_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_c_real), logits=self.d_c_real))
        self.dc_c_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_c_fake), logits=self.d_c_fake))
        self.dc_c_loss = self.dc_c_loss_fake + self.dc_c_loss_real

        self.g_g_l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_g_fake), logits=self.d_g_fake))
        self.g_c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_c_fake), logits=self.d_c_fake))
        self.g_l = self.g_c_loss + self.g_g_l

        self.encoder_training_l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ip_l, logits=self.encoder_op_l))

        self.all_variables = tf.trainable_variables()
        self.dc_g_var = [var for var in self.all_variables if 'dc_g_' in var.name]
        self.dc_c_var = [var for var in self.all_variables if 'dc_c_' in var.name]
        self.en_var = [var for var in self.all_variables if 'e_' in var.name]

        self.autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate, beta1=Config.beta1).minimize(self.autoencoder_loss)
        self.discriminator_g_optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate, beta1=Config.beta1).minimize(self.dc_g_l, var_list=self.dc_g_var)
        self.discriminator_c_optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate, beta1=Config.beta1).minimize(self.dc_c_loss, var_list=self.dc_c_var)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate, beta1=Config.beta1).minimize(self.g_l, var_list=self.en_var)
        self.supervised_encoder_optimizer = tf.train.AdamOptimizer(learning_rate=Config.learning_rate, beta1=Config.beta1).minimize(self.encoder_training_l, var_list=self.en_var)

        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver = tf.train.Saver() 
        
        
    def fit(self):
        step = 0
        tensorboard_path = './tensorboard/'
        saved_model_path = './model/'
        writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=self.sess.graph)
        X, Y = mnist.test.next_batch(Config.n_labeled)
        
        for i in range(Config.n_epochs):
            n_batches = int(Config.n_labeled / Config.batch_size)
            print("Epoch {}/{}".format(i, Config.n_epochs))
            for batch in range(1, n_batches + 1):
                z_real_dist = np.random.randn(Config.batch_size, Config.z_dim) * 5.
                real_cat_dist = np.random.randint(low=0, high=10, size=Config.batch_size)
                real_cat_dist = np.eye(Config.n_labels)[real_cat_dist]
                batch_x_ul, _ = mnist.train.next_batch(Config.batch_size)
                batch_X, batch_Y = self.next_batch(X, Y, batch_size=Config.batch_size)
                self.sess.run(self.autoencoder_optimizer, feed_dict={self.x_ip: batch_x_ul, self.x_op: batch_x_ul})
                self.sess.run(self.discriminator_g_optimizer, feed_dict={self.x_ip: batch_x_ul, self.x_op: batch_x_ul, self.r_distribution: z_real_dist})
                self.sess.run(self.discriminator_c_optimizer, feed_dict={self.x_ip: batch_x_ul, self.x_op: batch_x_ul, self.cat_distribution: real_cat_dist})
                self.sess.run(self.generator_optimizer, feed_dict={self.x_ip: batch_x_ul, self.x_op: batch_x_ul})
                self.sess.run(self.supervised_encoder_optimizer, feed_dict={self.x_ip_l: batch_X, self.y_ip_l: batch_Y})
                if batch % 5 == 0:
                    a_l, d_g_l, d_c_l, g_l = self.sess.run([self.autoencoder_loss, self.dc_g_l, self.dc_c_loss, self.g_l], feed_dict={self.x_ip: batch_x_ul, self.x_op: batch_x_ul, self.r_distribution: z_real_dist, self.y_ip_l: batch_Y, self.x_ip_l: batch_X, self.cat_distribution: real_cat_dist})
                    print("Epoch: {}, iteration: {}".format(i, batch))
                    print("Autoencoder Loss: {}".format(a_l))
                    print("Discriminator Gauss Loss: {}".format(d_g_l))
                    print("Discriminator Categorical Loss: {}".format(d_c_l))
                    print("Generator Loss: {}".format(g_l))
                step += 1
            acc = 0
            num_batches = int(mnist.validation.num_examples/Config.batch_size)
            for j in range(num_batches):
                batch_X, batch_Y = mnist.validation.next_batch(batch_size=Config.batch_size)
                encoder_acc = self.sess.run(self.accuracy, feed_dict={self.x_ip_l: batch_X, self.y_ip_l: batch_Y})
                acc += encoder_acc
            acc /= num_batches
            print("Classification Accuracy: {}".format(acc))
            self.saver.save(self.sess, save_path=saved_model_path, global_step=step)

    def generate(self):
        pass
    
    def next_batch(self, X, Y, batch_size):
        i = np.arange(Config.n_labeled)
        idx = np.random.permutation(i)[:batch_size]
        return X[idx], Y[idx]
    
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
            e_Dense_1 = tf.nn.relu(self.Dense(x, Config.input_dim, Config.n_l1, 'e_Dense_1'))
            e_Dense_2 = tf.nn.relu(self.Dense(e_Dense_1, Config.n_l1, Config.n_l2, 'e_Dense_2'))
            latent_variable = self.Dense(e_Dense_2, Config.n_l2, Config.z_dim, 'e_latent_variable')
            cat_op = self.Dense(e_Dense_2, Config.n_l2, Config.n_labels, 'e_label')
            output_label = tf.nn.softmax(logits=cat_op, name='e_op_label')
            return output_label, latent_variable

    def decoder(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Decoder'):
            d_Dense_1 = tf.nn.relu(self.Dense(x, Config.z_dim + Config.n_labels, Config.n_l2, 'd_Dense_1'))
            d_Dense_2 = tf.nn.relu(self.Dense(d_Dense_1, Config.n_l2, Config.n_l1, 'd_Dense_2'))
            output = tf.nn.sigmoid(self.Dense(d_Dense_2, Config.n_l1, Config.input_dim, 'd_op'))
            return output

    def discriminator_gauss(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Discriminator_Gauss'):
            dc_Dense1 = tf.nn.relu(self.Dense(x, Config.z_dim, Config.n_l1, name='dc_g_Dense1'))
            dc_Dense2 = tf.nn.relu(self.Dense(dc_Dense1, Config.n_l1, Config.n_l2, name='dc_g_Dense2'))
            output = self.Dense(dc_Dense2, Config.n_l2, 1, name='dc_g_op')
            return output

    def discriminator_categorical(self, x, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Discriminator_Categorial'):
            dc_Dense1 = tf.nn.relu(self.Dense(x, Config.n_labels, Config.n_l1, name='dc_c_Dense1'))
            dc_Dense2 = tf.nn.relu(self.Dense(dc_Dense1, Config.n_l1, Config.n_l2, name='dc_c_Dense2'))
            output = self.Dense(dc_Dense2, Config.n_l2, 1, name='dc_c_op')
            return output
			
	def generate(self, latent, label):
        latent, label = np.reshape(latent, (1, SupervisedConfig.z_dim)), np.reshape(label, (1, SupervisedConfig.n_labels))
        ip = np.concatenate((label, latent), 1)
        return self.sess.run(op, feed_dict={self.m_decoder_ip: ip})