# -*- coding: utf-8 -*-
"""
Guided recognition of dumbbell targets.
"""
import numpy as np
from math import floor, ceil

import matplotlib.pyplot as pl, matplotlib.cm as cm
from matplotlib.widgets import  RectangleSelector

rects = []
tiles = []
def onselect(eclick, erelease):
    xs = np.sort(np.r_[eclick.xdata, erelease.xdata])
    ys = np.sort(np.r_[eclick.ydata, erelease.ydata])
    rects.append(np.c_[xs, ys].flatten())
    tiles.append(im[floor(ys[0]):ceil(ys[1]), floor(xs[0]):ceil(xs[1])])
    
fig = pl.figure()
ax = pl.subplot('111')

im = pl.imread('data/20150714/dumbbell_end/20150714_Scene15_1-DVR Express CLFC_3136.TIF')
ax.imshow(im, cmap=cm.gray)

sel_style = {'edgecolor': 'red', 'fill': False}
rs = RectangleSelector(ax, onselect, rectprops=sel_style)
rs.set_active(True)

pl.show()

pl.figure()
pl.imshow(tiles[-1], cmap=cm.gray)
pl.show()
