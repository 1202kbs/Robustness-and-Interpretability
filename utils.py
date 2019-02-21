import math, pickle

from skimage import feature, transform
from scipy.stats import gaussian_kde

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges
    
    data = np.clip(data, a_min=np.min(data), a_max=np.percentile(data, percentile))
    
    v, _ = pixel_range(data)

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=v[0], vmax=v[1])
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis


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


def convert(index_or_coord, width, height):
  
    if type(index_or_coord) is int:

        x = index_or_coord // width
        y = index_or_coord % width

        return [x,y]

    else:

        return x * width + y


def sample(scale, size, normal=True):
    
    if normal:
        
        return np.random.normal(scale=scale, size=size)
    
    else:
        
        return np.random.uniform(low=-scale, high=scale, size=size)


def ellipse(x, y, center_x=0, center_y=0, width_x=1, width_y=1, angle=0):
    
    return (((x - center_x) * np.cos(angle) + (y - center_y) * np.sin(angle)) / width_x) ** 2 + (((x - center_x) * np.sin(angle) - (y - center_y) * np.cos(angle)) / width_y) ** 2 <= 1


# def plot(samples, X_dim, channel):
# 	fig = plt.figure(figsize=(4, 4))
# 	gs = gridspec.GridSpec(4, 4)
# 	gs.update(wspace=0.05, hspace=0.05)

# 	dim1 = int(np.sqrt(X_dim))

# 	samples = (samples + 1) / 2

# 	for i, sample in enumerate(samples):
# 		ax = plt.subplot(gs[i])
# 		plt.axis('off')
# 		ax.set_xticklabels([])
# 		ax.set_yticklabels([])
# 		ax.set_aspect('equal')

# 		if channel == 1:
# 			plt.imshow(sample.reshape([dim1, dim1]), cmap=plt.get_cmap('gray'))
# 		else:
# 			plt.imshow(sample.reshape([dim1, dim1, channel]))

# 	return fig


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


def translate(img, x, y):
    """
    Translates the given image.

    :param x: distance to translate in the positive x-direction
    :param y: distance to translate in the positive y-direction
    :returns: the translated image as an numpy array
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img.reshape(28,28), M, (28, 28)).reshape(1,784)


def find_roi(img, ksize, coords):
    """
    Finds the feature with the largest relevance scores.

    :param img: the image to find the feature with the largest relevance score
    :param ksize: the size of the sliding window
    :param coords: the coordinates to ignore
    :returns: the coordinate of the feature with the largest relevance score. If the window size is larger than 1X1, function returns the position of the leftmost pixel.
    """
    size = np.shape(img)
    temp = np.copy(img)
    for coord in coords:
        temp[coord[0]:coord[0]+ksize[0], coord[1]:coord[1]+ksize[1]] = -np.infty

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