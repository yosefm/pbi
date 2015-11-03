# -*- coding: utf-8 -*-
"""
Guided recognition of dumbbell targets.
"""
import numpy as np
from math import floor, ceil

import matplotlib.pyplot as pl, matplotlib.cm as cm
from matplotlib.widgets import  RectangleSelector

def onselect(eclick, erelease, im, rects, tiles):
    xs = np.sort(np.r_[eclick.xdata, erelease.xdata])
    ys = np.sort(np.r_[eclick.ydata, erelease.ydata])
    rects.append(np.c_[xs, ys].flatten())
    print rects[-1]
    tiles.append(im[floor(ys[0]):ceil(ys[1]), floor(xs[0]):ceil(xs[1])])
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

for cam in xrange(4):
    r, t = mark_image('data/20150714/dumbbell_end/cam%d.3136' % (cam + 1))
    print cam
    print r[0]
    print r[1]

    