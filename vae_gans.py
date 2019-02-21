import os, math

from functools import reduce

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from layers import RegDense, RegConv2D, RegConv2DTranspose


def l2_norm(x):

    return (tf.sqrt(tf.reduce_sum(x ** 2, axis=[1,2,3])))[..., None, None, None]


class Model():
    
    def __init__(self, logdir, lmda=1.0, zdim=100, learning_rate=1e-3, beta1=0.9, beta2=0.99, name=None):
        
        if not os.path.exists(logdir):
            
            os.makedirs(logdir)
        
        self.logdir = logdir
        self.lmda = lmda
        self.zdim = zdim
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.name = name
    
    def save(self, sess):

        self.saver.save(sess, self.logdir + 'model')

    def load(self, sess):

        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)

        if latest_checkpoint: 

            self.saver.restore(sess, latest_checkpoint)
    
    def reconstruct(self, sess, dataset, batch_size=100):
        
        n_itrs = math.ceil(len(dataset[0]) / batch_size)
        res = []
        
        for itr in range(n_itrs):
            
            batch_xs = dataset[0][itr * batch_size:(itr + 1) * batch_size]
            res.append(sess.run(self.X_tilde, feed_dict={self.X: batch_xs}))
        
        return np.concatenate(res, axis=0)
    
    def generate(self, sess, N, batch_size=100):
        
        n_itrs = math.ceil(N / batch_size)
        res = []
        
        xs = np.zeros(shape=[N] + self.X.get_shape().as_list()[1:])
        
        for itr in range(n_itrs):
            
            batch_xs = xs[itr * batch_size:(itr + 1) * batch_size]
            res.append(sess.run(self.X_fake, feed_dict={self.X: batch_xs}))
        
        return np.concatenate(res, axis=0)

    def latent(self, sess, dataset, batch_size=100):
        
        n_itrs = math.ceil(len(dataset[0]) / batch_size)
        res = []

        for itr in range(n_itrs):

            batch_xs = dataset[0][itr * batch_size:(itr + 1) * batch_size]
            res.append(sess.run(self.z_mean, feed_dict={self.X: batch_xs}))

        return np.concatenate(res, axis=0)
    
    def encode(self, inputs):
        
        raise NotImplementedError
    
    def decode(self, inputs):
        
        raise NotImplementedError
    
    def disc(self, inputs):

        raise NotImplementedError
    
    def build_model(self):
        
        with tf.variable_scope(self.name):
            
            # Reconstructions
            ep = tf.random_normal(shape=[tf.shape(self.X)[0], self.zdim])
            self.z_mean, self.z_sigma_log_sq = self.encode(self.X)
            z = self.z_mean + tf.sqrt(tf.exp(self.z_sigma_log_sq)) * ep
            self.X_tilde = self.decode(z)
            
            # Samples from random prior
            self.z_p = tf.random_normal(shape=[tf.shape(self.X)[0], self.zdim])
            self.X_fake = self.decode(self.z_p)
            
            # Discriminator logits
            d_real = self.disc(self.X)
            d_fake = self.disc(self.X_fake)
            d_dec = self.disc(self.X_tilde)
            
            # Losses and training graph
            with tf.name_scope('Losses'):
                
                with tf.name_scope('Reconst_Loss'):
                    
                    reconst_loss = tf.reduce_sum(tf.abs(self.X - self.X_tilde), axis=(1,2,3))
                
                with tf.name_scope('KL_Loss'):
                
                    kl_loss = -0.5 * tf.reduce_sum(1 + self.z_sigma_log_sq - tf.square(self.z_mean) - tf.exp(self.z_sigma_log_sq), axis=1)
                
                with tf.name_scope('Enc_Loss'):
                    
                    self.enc_loss = tf.reduce_mean(reconst_loss + kl_loss)
                
                with tf.name_scope('Dec_Loss'):
                    
                    self.dec_loss = tf.reduce_mean(self.lmda * reconst_loss - tf.log(d_dec + 1e-8) - tf.log(d_fake + 1e-8))
                
                with tf.name_scope('Disc_loss'):
                    
                    self.disc_loss = -tf.reduce_mean(tf.log(d_real + 1e-8) + tf.log(1 - d_dec + 1e-8) + tf.log(1 - d_fake + 1e-8))
                
                with tf.variable_scope('Enc_Train'):
                
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/Encoder')):

                        enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/Encoder')
                        self.enc_train = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2).minimize(self.enc_loss, var_list=enc_vars)
                
                with tf.variable_scope('Dec_Train'):
                    
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/Decoder')):

                        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/Decoder')
                        self.dec_train = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2).minimize(self.dec_loss, var_list=dec_vars)
                
                with tf.variable_scope('Disc_Train'):
                    
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name + '/Disc')):

                        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name + '/Disc')
                        self.disc_train = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2).minimize(self.disc_loss, var_list=disc_vars)
    
    def init_saver(self):
        
        self.saver = tf.train.Saver()


class MNIST_VAE_GAN(Model):
    
    def __init__(self, logdir, lmda=1.0, zdim=10, learning_rate=5e-4, beta1=0.5, beta2=0.9, name='MNIST_VAE_GAN'):
        
        super(MNIST_VAE_GAN, self).__init__(logdir, lmda, zdim, learning_rate, beta1, beta2, name)
        
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], 'X')
        self.training = tf.placeholder_with_default(False, [], 'training')
        
        # Encoder Layers
        enc_params1 = {'training': self.training, 'kernel_size': [4,4], 'strides': (2,2), \
                       'padding': 'SAME', 'activation': tf.nn.relu, 'use_batchnorm': True}
        enc_params2 = {'training': self.training, 'use_batchnorm': True}
        
        self.enc_layers = [RegConv2D(filters=64, name='conv1', **enc_params1), 
                           RegConv2D(filters=128, name='conv2', **enc_params1), 
                           RegConv2D(filters=256, name='conv3', **enc_params1), 
                           RegDense(units=self.zdim * 2, name='dense4', **enc_params2)]
        
        # Discriminator Layers
        disc_params1 = {'training': self.training, 'kernel_size': [4,4], 'strides': (2,2), \
                        'padding': 'SAME', 'activation': tf.nn.relu}
        disc_params2 = {'training': self.training, 'kernel_size': [4,4], 'strides': (2,2), \
                        'padding': 'SAME', 'activation': tf.nn.relu, 'use_batchnorm': True}
        disc_params3 = {'training': self.training, 'activation': tf.nn.relu, 'use_batchnorm': True}
        disc_params4 = {'training': self.training, 'activation': tf.nn.sigmoid}
        
        self.disc_layers = [RegConv2D(filters=32, name='conv1', **disc_params1), 
                            RegConv2D(filters=128, name='conv2', **disc_params2), 
                            RegConv2D(filters=256, name='conv3', **disc_params2), 
                            RegConv2D(filters=256, name='conv4', **disc_params2), 
                            RegDense(units=512, name='dense5', **disc_params3), 
                            RegDense(units=1, name='dense6', **disc_params4)]
        
        # Decoder / Generator Layers
        dec_params1 = {'training': self.training, 'activation': tf.nn.relu, 'use_batchnorm': True}
        dec_params2 = {'training': self.training, 'kernel_size': [4,4], 'activation': tf.nn.relu, 'use_batchnorm': True}
        dec_params3 = {'training': self.training, 'kernel_size': [4,4], 'activation': tf.nn.tanh}
        
        self.dec_layers = [RegDense(units=2 * 2 * 256, name='dense1', **dec_params1), 
                           RegConv2DTranspose(filters=256, strides=(2, 2), padding='SAME', name='deconv2', **dec_params2), 
                           RegConv2DTranspose(filters=128, strides=(1, 1), padding='VALID', name='deconv3', **dec_params2), 
                           RegConv2DTranspose(filters=32, strides=(2, 2), padding='SAME', name='deconv4', **dec_params2), 
                           RegConv2DTranspose(filters=1, strides=(2, 2), padding='SAME', name='deconv5', **dec_params3)]
        
        self.build_model()
        self.init_saver()
    
    def encode(self, inputs):
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):

            outputs = reduce((lambda x, y: y(x)), [inputs] + self.enc_layers[:3])
            outputs = tf.reshape(outputs, shape=[-1, 4 * 4 * 256], name='flat4')
            outputs = reduce((lambda x, y: y(x)), [outputs] + self.enc_layers[3:])
        
        return outputs[:,:self.zdim], outputs[:,self.zdim:]
    
    def decode(self, inputs):
        
        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):

            outputs = reduce((lambda x, y: y(x)), [inputs] + self.dec_layers[:1])
            outputs = tf.reshape(outputs, shape=[-1, 2, 2, 256], name='flat1')
            outputs = reduce((lambda x, y: y(x)), [outputs] + self.dec_layers[1:])
        
        return outputs
    
    def disc(self, inputs):
        
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):

            outputs = reduce((lambda x, y: y(x)), [inputs] + self.disc_layers[:4])
            outputs = tf.reshape(outputs, shape=[-1, 2 * 2 * 256], name='flat4')
            outputs = reduce((lambda x, y: y(x)), [outputs] + self.disc_layers[4:])
        
        return outputs


class CIFAR_VAE_GAN(Model):
    
    def __init__(self, logdir, lmda=1e-3, zdim=128, learning_rate=2e-4, beta1=0.0, beta2=0.9, name='CIFAR_VAE_GAN'):
        
        super(CIFAR_VAE_GAN, self).__init__(logdir, lmda, zdim, learning_rate, beta1, beta2, name)
        
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], 'X')
        self.training = tf.placeholder_with_default(False, [], 'training')
        
        # Encoder Layers
        enc_params1 = {'training': self.training, 'kernel_size': [5,5], 'strides': (2,2), \
                       'padding': 'SAME', 'activation': tf.nn.relu, 'use_batchnorm': True}
        enc_params2 = {'training': self.training, 'use_batchnorm': True}
        
        self.enc_layers = [RegConv2D(filters=64, name='conv1', **enc_params1), 
                           RegConv2D(filters=128, name='conv2', **enc_params1), 
                           RegConv2D(filters=256, name='conv3', **enc_params1), 
                           RegDense(units=self.zdim * 2, name='dense4', **enc_params2)]
        
        # Discriminator Layers
        leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.1)
        
        disc_params1 = {'training': self.training, 'kernel_size': [3,3], 'strides': (1,1), \
                        'padding': 'SAME', 'activation': leaky_relu, 'snbeta': 1.0}
        disc_params2 = {'training': self.training, 'kernel_size': [4,4], 'strides': (2,2), \
                        'padding': 'SAME', 'activation': leaky_relu, 'snbeta': 1.0}
        disc_params3 = {'training': self.training, 'activation': tf.nn.sigmoid}
        
        self.disc_layers = [RegConv2D(filters=64, name='conv1', **disc_params1), 
                            RegConv2D(filters=64, name='conv2', **disc_params2), 
                            RegConv2D(filters=128, name='conv3', **disc_params1), 
                            RegConv2D(filters=128, name='conv4', **disc_params2), 
                            RegConv2D(filters=256, name='conv5', **disc_params1), 
                            RegConv2D(filters=256, name='conv6', **disc_params2), 
                            RegConv2D(filters=512, name='conv7', **disc_params1), 
                            RegDense(units=1, name='dense8', **disc_params3)]
        
        # Decoder / Generator Layers
        dec_params1 = {'training': self.training}
        dec_params2 = {'training': self.training, 'kernel_size': [4,4], 'activation': tf.nn.relu, 'use_batchnorm': True}
        dec_params3 = {'training': self.training, 'kernel_size': [3,3], 'activation': tf.nn.tanh}
        
        self.dec_layers = [RegDense(units=4 * 4 * 512, name='dense1', **dec_params1),
                           RegConv2DTranspose(filters=256, strides=(2,2), padding='SAME', name='deconv2', **dec_params2), 
                           RegConv2DTranspose(filters=128, strides=(2,2), padding='SAME', name='deconv3', **dec_params2), 
                           RegConv2DTranspose(filters=64, strides=(2,2), padding='SAME', name='deconv4', **dec_params2), 
                           RegConv2D(filters=3, strides=(1,1), padding='SAME', name='conv5', **dec_params3)]
        
        self.build_model()
        self.init_saver()
    
    def encode(self, inputs):
        
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):

            outputs = reduce((lambda x, y: y(x)), [inputs] + self.enc_layers[:3])
            outputs = tf.reshape(outputs, shape=[-1, 4 * 4 * 256], name='flat4')
            outputs = reduce((lambda x, y: y(x)), [outputs] + self.enc_layers[3:])
        
        return outputs[:,:self.zdim], outputs[:,self.zdim:]
    
    def decode(self, inputs):
        
        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
            
            outputs = reduce((lambda x, y: y(x)), [inputs] + self.dec_layers[:1])
            outputs = tf.reshape(outputs, shape=[-1, 4, 4, 512], name='flat1')
            outputs = reduce((lambda x, y: y(x)), [outputs] + self.dec_layers[1:])
        
        return outputs
    
    def disc(self, inputs):
        
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            
            outputs = reduce((lambda x, y: y(x)), [inputs] + self.disc_layers[:7])
            outputs = tf.reshape(outputs, shape=[-1, 4 * 4 * 512], name='flat7')
            outputs = reduce((lambda x, y: y(x)), [outputs] + self.disc_layers[7:])
        
        return outputs