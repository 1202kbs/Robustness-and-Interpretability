import tensorflow as tf


def L2(kernel, name='L2'):

    with tf.name_scope(name):

        loss = tf.nn.l2_loss(kernel)

    return loss


def SN_Dense(inputs, kernel, beta=1.0, Ip=1, reuse=False, name='SN_Dense'):
    
    def power_iteration(u, w, Ip):
        
        u_ = u
        
        for _ in range(Ip):
            
            v_ = tf.nn.l2_normalize(tf.matmul(u_, tf.transpose(w)))
            u_ = tf.nn.l2_normalize(tf.matmul(v_, w))
        
        return u_, v_
    
    x_shape = inputs.get_shape().as_list()
    w_shape = kernel.get_shape().as_list()
    
    with tf.variable_scope(name) as scope:
        
        if reuse:
            
            scope.reuse_variables()
            
        u = tf.get_variable('u', shape=[1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
        w = tf.reshape(kernel, [-1, w_shape[-1]])
        
        u_hat, v_hat = power_iteration(u, w, Ip)
        
        sigma = tf.maximum(tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat)) / beta, 1)
        
        with tf.control_dependencies([u.assign(u_hat)]):
            
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
        
    return w_norm


def SN_Conv(inputs, kernel, strides, padding='VALID', beta=1.0, Ip=1, reuse=False, name='SN_Conv'):
    
    x_shape = inputs.get_shape().as_list()
    w_shape = kernel.get_shape().as_list()
    
    u_width = x_shape[1]
    u_depth = w_shape[2]
    
    def power_iteration_conv(u, w, Ip):
        
        u_ = u
        
        for _ in range(Ip):
            
            v_ = tf.nn.l2_normalize(tf.nn.conv2d(u_, w, strides=[1, strides[0], strides[1], 1], padding=padding))
            u_ = tf.nn.l2_normalize(tf.nn.conv2d_transpose(v_, w, [1, u_width, u_width, u_depth], strides=[1, strides[0], strides[1], 1], padding=padding))
        
        return u_, v_
    
    with tf.variable_scope(name) as scope:
        
        if reuse:
            
            scope.reuse_variables()
        
        u = tf.get_variable('u', shape=[1, u_width, u_width, u_depth], initializer=tf.truncated_normal_initializer(), trainable=False)
        
        u_hat, v_hat = power_iteration_conv(u, kernel, Ip)
        
        z = tf.nn.conv2d(u_hat, kernel, strides=[1, strides[0], strides[1], 1], padding=padding)
        sigma = tf.maximum(tf.reduce_sum(tf.multiply(z, v_hat)) / beta, 1)
        
        with tf.control_dependencies([u.assign(u_hat)]):
            
            w_norm = kernel / sigma
        
    return w_norm