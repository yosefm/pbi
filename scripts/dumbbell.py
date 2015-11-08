# -*- coding: utf-8 -*-
"""
Guided recognition of dumbbell targets.

Loosely based on the Matlab version from Alex et al. here:
https://bitbucket.org/alexliberzonlab/ptv_postproc/src/106a97675479ad9afa95a1837bb5a6f9ff7de024/Matlab/?at=master
Files: tau_dumbbell_detection_db_y1.m and those called from within.
"""
import numpy as np
from scipy import signal
from math import floor, ceil

import matplotlib.pyplot as pl, matplotlib.cm as cm
from matplotlib.widgets import  RectangleSelector

def onselect(eclick, erelease, im, rects, tiles):
    xs = np.sort(np.r_[eclick.xdata, erelease.xdata])
    ys = np.sort(np.r_[eclick.ydata, erelease.ydata])
    rects.append(np.c_[xs, ys].flatten())
    print rects[-1]
    
    tiles.append(im[floor(ys[0]):ceil(ys[1]), floor(xs[0]):ceil(xs[1])].copy())
    if len(rects) == 2:
        pl.close()

def mark_image(image_path):
    rects = []
    tiles = []

    pl.figure()
    ax = pl.subplot('111')

    im = pl.imread(image_path)
    ax.imshow(im, cmap=cm.gray)
    pl.title(image_path)
    
    sel_style = {'edgecolor': 'red', 'fill': False}
    rs = RectangleSelector(ax, lambda clk, rls: onselect(clk, rls, im, rects, tiles), 
        rectprops=sel_style)
    rs.set_active(True)

    pl.show()
    return rects, tiles

def template_match(image, template):
    """
    Finds the center of image area matching a template.
    
    Arguments:
    image - grayscale image as read by pl.imread.
    template - a smaller image to be found in the larger.
    
    Returns:
    x, y - position of best match.
    """
    tmpl = (template - template.mean())[::-1, ::-1]
    corr = signal.fftconvolve(image - image.mean(), tmpl, mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    return x, y
    
# First mark each in turn:
num_cams = 4
templates = []
for cam in xrange(num_cams):
    r, t = mark_image('data/20150714/dumbbell_end/cam%d.3136' % (cam + 1))
    templates.append(t)

# Then show all marks in one plot.
pl.figure(figsize=(15,15))
for cam in xrange(num_cams):
    pl.subplot(2, 2, cam + 1)
    im = pl.imread('data/20150714/dumbbell_end/cam%d.3136' % (cam + 1))
    pl.imshow(im, cmap=cm.gray)
    
    x, y = template_match(im, templates[cam][0])
    pl.plot(x, y, 'ro')
    print cam, x, y
    
    # Blank out the first find in case it is similar to the second template.
    h, w = templates[cam][0].shape
    im_blanked = im.copy()
    im_blanked[y - h/2 : y + h/2, x - w/2 : x + w/2] = 0
    
    x, y = template_match(im_blanked, templates[cam][1])
    pl.plot(x, y, 'ro')
    print cam, x, y
    
    pl.axis('tight')
    pl.axis('off')
    
pl.show()
