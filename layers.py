import regularizers as regs
import tensorflow as tf
import numpy as np


class RegLayer():
    
    def __init__(self, training, activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, name=None):
        
        self.training = training
        self.activation = activation if activation else tf.identity
        self.use_batchnorm = use_batchnorm
        self.droprate = droprate
        self.snbeta = snbeta
        self.l2rate = l2rate
        self.name = name

    def _activation(self, inputs):
        
        raise NotImplementedError
    
    def __call__(self, inputs):
        
        with tf.variable_scope(self.name):
            
            outputs = self._activation(inputs)
            
            if self.use_batchnorm : outputs = tf.layers.batch_normalization(inputs=outputs, training=self.training, name='BatchNorm')
            if self.droprate : outputs = tf.layers.dropout(inputs=outputs, rate=self.droprate, training=self.training, name='Dropout')
        
        return outputs


class RegDense(RegLayer):
    
    def __init__(self, training, units, activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, name=None):
        
        super(RegDense, self).__init__(training, activation, use_batchnorm, droprate, snbeta, l2rate, name)
        self.units = units
        
    def _activation(self, inputs):
            
        self.kernel = tf.get_variable(name='kernel', shape=[inputs.shape[-1], self.units], initializer=tf.glorot_uniform_initializer())
        self.bias = tf.get_variable(name='bias', shape=[self.units], initializer=tf.constant_initializer(0.0))
        
        if self.snbeta : self.kernel = regs.SN_Dense(inputs, self.kernel, self.snbeta)
        if self.l2rate : tf.add_to_collection('RegLosses', self.l2rate * regs.L2(self.kernel))
        
        return self.activation(tf.matmul(inputs, self.kernel) + self.bias)


class RegConv2D(RegLayer):
    
    def __init__(self, training, filters, kernel_size, strides=(1,1), padding='VALID', activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, name=None):
        
        super(RegConv2D, self).__init__(training, activation, use_batchnorm, droprate, snbeta, l2rate, name)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
    
    def _activation(self, inputs):
        
        self.kernel = tf.get_variable(name='kernel', shape=[self.kernel_size[0], self.kernel_size[1], inputs.shape[-1], self.filters], initializer=tf.glorot_uniform_initializer())
        self.bias = tf.get_variable(name='bias', shape=[self.filters], initializer=tf.constant_initializer(0.0))
        
        if self.snbeta : self.kernel = regs.SN_Conv(inputs, self.kernel, self.strides, self.padding, self.snbeta)
        if self.l2rate : tf.add_to_collection('RegLosses', self.l2rate * regs.L2(self.kernel))
        
        return self.activation(tf.nn.conv2d(input=inputs, filter=self.kernel, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding) + self.bias)


class RegConv2DTranspose(RegLayer):
    
    def __init__(self, training, filters, kernel_size, strides=(1,1), padding='VALID', activation=None, use_batchnorm=False, droprate=None, snbeta=None, l2rate=None, name=None):
        
        super(RegConv2DTranspose, self).__init__(training, activation, use_batchnorm, droprate, snbeta, l2rate, name)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
    
    def _activation(self, inputs):
        
        self.kernel = tf.get_variable(name='kernel', shape=[self.kernel_size[0], self.kernel_size[1], self.filters, inputs.shape[-1]], initializer=tf.glorot_uniform_initializer())
        self.bias = tf.get_variable(name='bias', shape=[self.filters], initializer=tf.constant_initializer(0.0))
        
        if self.snbeta : self.kernel = regs.SN_Conv(inputs, self.kernel, self.strides, self.padding, self.snbeta)
        if self.l2rate : tf.add_to_collection('RegLosses', self.l2rate * regs.L2(self.kernel))
        
        inputs_shape = inputs.get_shape().as_list()
        
        if self.padding == 'SAME':
            output_shape = [tf.shape(inputs)[0], inputs_shape[1] * self.strides[0], inputs_shape[2] * self.strides[1], self.filters]
        else:
            h = (inputs_shape[1] - 1) * self.strides[0] + self.kernel_size[0]
            w = (inputs_shape[2] - 1) * self.strides[1] + self.kernel_size[1]
            output_shape = [tf.shape(inputs)[0], h, w, self.filters]
        
        return self.activation(tf.nn.conv2d_transpose(value=inputs, filter=self.kernel, output_shape=output_shape, strides=[1, self.strides[0], self.strides[1], 1], padding=self.padding) + self.bias)