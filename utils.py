import math, pickle, itertools

from skimage import feature, transform
from scipy.stats import gaussian_kde

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import cv2


def batch_run(function, images, batch_size=5000):
    '''
    function   : lambda function taking images with shape [N,H,W,C] as input
    images     : tensor of shape [N,H,W,C]
    batch_size : batch size
    '''
    
    res = []
    
    for i in range(math.ceil(len(images) / batch_size)):
        
        res.append(function(images[i*batch_size:(i+1)*batch_size]))
    
    return np.concatenate(res, axis=0)


def preprocess(attributions, q1, q2, use_abs=False):
    
    if use_abs:
        attributions = np.abs(attributions)
    
    attributions = np.sum(attributions, axis=-1)
    
    a_min = np.percentile(attributions, q1, axis=(1,2), keepdims=True)
    a_max = np.percentile(attributions, q2, axis=(1,2), keepdims=True)
    
    pos = np.tile(a_min > 0, [1,attributions.shape[1],attributions.shape[2]])
    ind = np.where(attributions < a_min)
    
    attributions = np.clip(attributions, a_min, a_max)
    attributions[ind] = (1 - pos[ind]) * attributions[ind]
    
    
    return attributions

def scale(x):
    
    return x / 127.5 - 1.0


def save(dataset, file):
    
    with open(file, 'wb') as fo:
        
        pickle.dump(dataset, fo)


def unpickle(file):
    
    with open(file, 'rb') as fo:
        
        dict = pickle.load(fo, encoding='bytes')
    
    return dict


def pixel_range(img):
    vmin, vmax = np.min(img), np.max(img)

    if vmin * vmax >= 0:
        
        v = np.maximum(np.abs(vmin), np.abs(vmax))
        
        return [-v, v], 'bwr'
    
    else:

        if -vmin > vmax:
            vmax = -vmin
        else:
            vmin = -vmax

        return [vmin, vmax], 'bwr'

    r = size[0] - ksize[0] + 1
    c = size[1] - ksize[1] + 1
    pool = [np.sum(temp[i:i+ksize[0], j:j+ksize[1]]) for i in range(r) for j in range(c)]

    return (np.argmax(pool) // c, np.argmax(pool) % c)


def hist_kde(data, xlim):
    
    density = gaussian_kde(data)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    
    xs = np.linspace(xlim[0], xlim[1], 500)
    ys = density(xs)
    
    return xs, ys


def params_maker(param_names, param_values):
    
    product = itertools.product(*param_values)
    params = [list(zip(param_names, p)) for p in product]
    
    return params