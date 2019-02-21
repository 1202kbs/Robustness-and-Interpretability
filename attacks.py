import math

from tqdm import tqdm

import tensorflow as tf
import numpy as np

from utils import save


def l2_norm(x):
    
    if len(x.shape.as_list()) > 2:
        return (tf.sqrt(tf.reduce_sum(x ** 2, axis=[1,2,3])))[..., None, None, None]
    else:
        return (tf.sqrt(tf.reduce_sum(x ** 2 ,axis=[1])))[..., None]

def l2_project(org_images, adv_images, eps):

    deltas = adv_images - org_images
    
    norms = l2_norm(deltas)
    delta_normalized = deltas / (norms + 1e-8) * tf.minimum(eps, norms)

    # Add normalized perturbation and clip to ensure valid pixel range
    adv_images = org_images + delta_normalized
    adv_images = tf.clip_by_value(adv_images, -1, 1)

    return adv_images

def linf_project(org_images, adv_images, eps):
    
    # Add l-inf normalized perturbation and clip to ensure valid pixel range
    adv_images = tf.clip_by_value(adv_images, org_images - eps, org_images + eps)
    adv_images = tf.clip_by_value(adv_images, -1, 1)
    
    return adv_images


class Attack():
    
    def __init__(self, model, eps, step_size, n_steps, norm='inf', loss_type='xent', random_start=False, name=None):
        
        self.model = model
        self.eps = float(eps)
        self.step_size = float(step_size)
        self.n_steps = n_steps
        self.norm = norm
        self.loss_type = loss_type
        self.random_start = random_start
        self.name = name

    def _get_loss(self, X):

        with tf.name_scope(self.name):

            if self.loss_type == 'xent':

                logits = self.model.classify(X)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.model.Y)

            elif self.loss_type == 'cw':

                logits = self.model.classify(X)
                label_mask = tf.one_hot(self.model.Y, depth=self.model.n_classes, on_value=1.0, off_value=0.0, dtype=tf.float32)
                c_logit = tf.reduce_sum(label_mask * logits, axis=1)
                w_logit = tf.reduce_max((1 - label_mask) * logits, axis=1)
                
                # Attack for analysis: Modify CW loss such that attack reaches end of epsilon-ball
                # loss = w_logit - c_logit
                
                # Attack for training: constrain the maximum loss
                loss = -tf.nn.relu(c_logit - w_logit + 500)

            else:

                raise Exception('{} loss is not available.'.format(self.loss_type))

        return loss

    # Build self.perturbation
    def _get_perturbation(self, X):
        
        raise NotImplementedError
    
    def _build_attack(self):

        init_attacks = self.model.X

        if self.random_start:

            if self.norm == '2':

                ps = tf.random_normal(shape=tf.shape(init_attacks))
                ps = ps / (l2_norm(ps) + 1e-8)
                init_attacks = l2_project(self.model.X, init_attacks + ps * self.eps, self.eps)

            elif self.norm == 'inf':

                ps = tf.random_uniform(minval=-self.eps, maxval=self.eps, shape=tf.shape(init_attacks))
                init_attacks = linf_project(self.model.X, init_attacks + ps, self.eps)

            else:

                raise Exception('L-{} norm bounded adversarial attack is not available.'.format(self.norm))

        def body(i, attacks):

            ps = self._get_perturbation(attacks)

            if self.norm == '2':

                ps = ps / (l2_norm(ps) + 1e-8)
                attacks = tf.stop_gradient(l2_project(self.model.X, attacks + ps * self.step_size, self.eps))

            elif self.norm == 'inf':

                ps = tf.sign(ps)
                attacks = tf.stop_gradient(linf_project(self.model.X, attacks + ps * self.step_size, self.eps))

            else:

                raise Exception('L-{} norm bounded adversarial attack is not available.'.format(self.norm))

            return i + 1, attacks

        i_0 = tf.constant(0)
        initial_vars = [i_0, init_attacks]
        cond = lambda i, _: tf.less(i, self.n_steps)
        _, attacks = tf.while_loop(cond, body, initial_vars, back_prop=False, parallel_iterations=1)

        self.attacks = tf.stop_gradient(attacks)
    
    def attack(self, sess, inputs, savedir=None, batch_size=100, show_progress=True):
        
        res = []
        n_itr = math.ceil(len(inputs[0]) / batch_size)
        
        iterator = tqdm(range(n_itr)) if show_progress else range(n_itr)
        
        for itr in iterator:
            
            batch_xs, batch_ys = inputs[0][itr * batch_size:(itr + 1) * batch_size], inputs[1][itr * batch_size:(itr + 1) * batch_size]
            res.append(sess.run(self.attacks, feed_dict={self.model.X: batch_xs, self.model.Y: batch_ys}))
        
        res = np.concatenate(res, axis=0)
        
        if savedir:
            
            save(res, savedir)
        
        return res


class GM(Attack):
    
    def __init__(self, model, eps, step_size, n_steps, norm='inf', loss_type='xent', random_start=False, name='GM'):
        
        super(GM, self).__init__(model, eps, step_size, n_steps, norm, loss_type, random_start, name)
        
        self._build_attack()
    
    # Gradient method uses vanilla gradient
    def _get_perturbation(self, X):

        with tf.name_scope(self.name):
            
            loss = self._get_loss(X)
            perturbation = tf.gradients(loss, X)[0]

        return perturbation
