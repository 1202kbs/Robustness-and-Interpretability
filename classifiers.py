import os, math

from functools import reduce

import tensorflow as tf
import numpy as np

from layers import RegDense, RegConv2D, RegConv2DTranspose


class Model():
    
    def __init__(self, logdir, n_classes, activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, attack=None, attack_params=None, optimizer=tf.train.AdamOptimizer, learning_rate=1e-3, name=None):
        
        if not os.path.exists(logdir):
            
            os.makedirs(logdir)
        
        self.logdir = logdir
        self.n_classes = n_classes
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.droprate = droprate
        self.snbeta = snbeta
        self.l2rate = l2rate
        self.attack = attack
        self.attack_params = attack_params
        self.optimizer = optimizer(learning_rate)
        self.name = name

    def classify(self, inputs):
        
        raise NotImplementedError
    
    def _build_model(self):

        # If attack is given, instantiate attack
        if self.attack:

            self.attack = self.attack(self, **self.attack_params)
            self.logits = self.classify(self.attack.attacks)

        else:

            self.logits = self.classify(self.X)
        
        self.yi = tf.argmax(self.logits, 1, name='Prediction')
        self.yx = tf.nn.softmax(self.logits, name='Scores')
        self.yv = tf.reduce_max(self.logits, 1, name='MaxScore')
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.yi, self.Y), tf.float32), name='Accuracy')

        with tf.name_scope('Losses'):
            
            self.Y_hot = tf.one_hot(self.Y, depth=self.n_classes)
            
            with tf.name_scope('CELoss'):

                self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y_hot))
            
            with tf.name_scope('RegLoss'):
                
                if tf.get_collection('RegLosses'):
                    self.reg_loss = tf.add_n(tf.get_collection('RegLosses'))
                else:
                    self.reg_loss = 0.0

            self.loss = tf.add(self.cross_entropy, self.reg_loss, name='Loss')

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):

                self.train = self.optimizer.minimize(self.loss, var_list=self.vars)
    
    @property
    def vars(self):
        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
    def _init_saver(self):

        self.saver = tf.train.Saver()
    
    def save(self, sess):

        self.saver.save(sess, self.logdir + 'model')

    def load(self, sess):

        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)

        if latest_checkpoint:

            self.saver.restore(sess, latest_checkpoint)

    def evaluate(self, sess, dataset, batch_size=100):

        n_itrs = math.ceil(len(dataset[0]) / batch_size)
        avg_acc = 0

        for itr in range(n_itrs):

            batch_xs, batch_ys = dataset[0][itr * batch_size:(itr + 1) * batch_size], dataset[1][itr * batch_size:(itr + 1) * batch_size]

            feed_dict = {self.X: batch_xs, self.Y: batch_ys, self.training: False}
            acc = sess.run(self.accuracy, feed_dict=feed_dict)
            avg_acc += acc / n_itrs

        return avg_acc

    def inference(self, sess, dataset, batch_size=100):

        n_itrs = math.ceil(len(dataset[0]) / batch_size)
        res = []

        for itr in range(n_itrs):

            batch_xs = dataset[0][itr * batch_size:(itr + 1) * batch_size]

            feed_dict = {self.X: batch_xs, self.training: False}
            res.append(sess.run(self.yi, feed_dict=feed_dict))

        return np.concatenate(res, axis=0)


class MNORM_DNN(Model):
    
    def __init__(self, logdir, n_classes, activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, attack=None, attack_params=None, optimizer=tf.train.AdamOptimizer, learning_rate=1e-3, name='MNORM_DNN'):
        
        super(MNORM_DNN, self).__init__(logdir, n_classes, activation, use_batchnorm, droprate,snbeta, l2rate, attack, attack_params, optimizer, learning_rate, name)
        
        self.X = tf.placeholder(tf.float32, [None,2], 'X')
        self.Y = tf.placeholder(tf.int64, [None], 'Y')
        self.training = tf.placeholder_with_default(False, [], 'training')
        
        params = {'training': self.training, 'use_batchnorm': self.use_batchnorm, 'droprate': self.droprate, \
                  'snbeta': self.snbeta, 'l2rate': self.l2rate}
        
        self.layers = [RegDense(units=128, activation=self.activation, name='dense1', **params), 
                       RegDense(units=self.n_classes, name='dense2', **params)]
        
        self._build_model()
        self._init_saver()
    
    def classify(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            outputs = reduce((lambda x, y: y(x)), [inputs] + self.layers)
        
        return outputs


class MNIST_CNN(Model):
    
    def __init__(self, logdir, n_classes=10, activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, attack=None, attack_params=None, optimizer=tf.train.AdamOptimizer, learning_rate=1e-3, name='MNIST_CNN'):
        
        super(MNIST_CNN, self).__init__(logdir, n_classes, activation, use_batchnorm, droprate, snbeta, l2rate, attack, attack_params, optimizer, learning_rate, name)
        
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1], 'X')
        self.Y = tf.placeholder(tf.int64, [None], 'Y')
        self.training = tf.placeholder_with_default(False, [], 'training')
        
        # We follow the architecture in TensorFlow tutorial but do not use dropout to remove any external influence.
        params1 = {'training': self.training, 'kernel_size': [5,5], 'padding': 'SAME', 'activation': self.activation, \
                   'use_batchnorm': self.use_batchnorm, 'droprate': self.droprate, 'snbeta': self.snbeta, 'l2rate': self.l2rate}
        params2 = {'training': self.training, 'activation': self.activation, 'use_batchnorm': self.use_batchnorm, 'droprate': self.droprate, \
                   'snbeta': self.snbeta, 'l2rate': self.l2rate}
        params3 = {'training': self.training, 'snbeta': self.snbeta, 'l2rate': self.l2rate}
        
        n_filters = [32, 64]
        
        self.layers = [RegConv2D(filters=n_filters[i], name='conv{}'.format(i + 1), **params1) for i in range(2)] \
                    + [RegDense(units=1024, name='dense3', **params2), 
                       RegDense(units=self.n_classes, name='dense4', **params3)]
        
        self._build_model()
        self._init_saver()
    
    def classify(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            pooling_params = {'pool_size': [2,2], 'strides': [2,2], 'padding': 'SAME'}
            
            with tf.variable_scope('Block1'):
                
                outputs = tf.layers.max_pooling2d(reduce((lambda x, y: y(x)), [inputs] + self.layers[:2]), name='pool2', **pooling_params)

            with tf.variable_scope('Block2'):
                
                outputs = tf.reshape(outputs, [-1, 64 * 14 * 14], name='flat2')
                outputs = reduce((lambda x, y: y(x)), [outputs] + self.layers[2:])
        
        return outputs


class CIFAR_CNN(Model):
    
    def __init__(self, logdir, n_classes=10, activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, attack=None, attack_params=None, optimizer=tf.train.AdamOptimizer, learning_rate=1e-3, name='CIFAR_CNN'):
        
        super(CIFAR_CNN, self).__init__(logdir, n_classes, activation, use_batchnorm, droprate, snbeta, l2rate, attack, attack_params, optimizer, learning_rate, name)
        
        self.X = tf.placeholder(tf.float32, [None, 32, 32, 3], 'X')
        self.Y = tf.placeholder(tf.int64, [None], 'Y')
        self.training = tf.placeholder_with_default(False, [], 'training')
        
        params1 = {'training': self.training, 'kernel_size': [3,3], 'padding': 'SAME', 'activation': self.activation, \
                   'use_batchnorm': self.use_batchnorm, 'droprate': self.droprate, 'snbeta': self.snbeta, 'l2rate': self.l2rate}
        params2 = {'training': self.training, 'activation': self.activation, 'use_batchnorm': self.use_batchnorm, 'droprate': self.droprate, \
                   'snbeta': self.snbeta, 'l2rate': self.l2rate}
        params3 = {'training': self.training, 'snbeta': self.snbeta, 'l2rate': self.l2rate}
        
        n_filters = [32, 32, 64, 64]
        
        self.layers = [RegConv2D(filters=n_filters[i], name='conv{}'.format(i + 1), **params1) for i in range(4)] \
                    + [RegDense(units=512, name='dense5', **params2), 
                       RegDense(units=self.n_classes, name='dense6', **params3)]
        
        self._build_model()
        self._init_saver()
    
    def classify(self, inputs):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            pooling_params = {'pool_size': [2,2], 'strides': [2,2], 'padding': 'SAME'}
            
            with tf.variable_scope('Block1'):
                
                outputs = tf.layers.max_pooling2d(reduce((lambda x, y: y(x)), [inputs] + self.layers[:2]), name='pool2', **pooling_params)

            with tf.variable_scope('Block2'):
                
                outputs = tf.layers.max_pooling2d(reduce((lambda x, y: y(x)), [outputs] + self.layers[2:4]), name='pool4', **pooling_params)

            with tf.variable_scope('Block3'):
                
                outputs = tf.reshape(outputs, [-1, 8 * 8 * 64], name='flat4')
                outputs = reduce((lambda x, y: y(x)), [outputs] + self.layers[4:])
        
        return outputs
