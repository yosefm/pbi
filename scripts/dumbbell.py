# -*- coding: utf-8 -*-
"""
Guided recognition of dumbbell targets.

Loosely based on the Matlab version from Alex et al. here:
https://bitbucket.org/alexliberzonlab/ptv_postproc/src/106a97675479ad9afa95a1837bb5a6f9ff7de024/Matlab/?at=master
Files: tau_dumbbell_detection_db_y1.m and those called from within.
"""
import numpy as np
from scipy import signal
from math import floor, ceil, sqrt

import matplotlib.pyplot as pl, matplotlib.cm as cm
from matplotlib.widgets import  RectangleSelector

from optv.tracking_framebuf import TargetArray, read_targets

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

def record_target(targ, targ_num, tmpl, centroid):
    """
    Put all template-matching results into a Target object.
    
    Arguments:
    targ - a Target object having the storage space.
    targ_num - target number in frame.
    tmpl - the matched template, a 2D grey image array.
    centroid - the position of center of match.
    """
    nx, ny = tmpl.shape
    
    targ.set_pos(centroid)
    targ.set_pnr(0)
    targ.set_tnr(-1)
    targ.set_pixel_counts(nx*ny, nx, ny)
    targ.set_sum_grey_value(tmpl.sum())

def process_image(image_path, templates, targets_path, targets_frame):
    """
    Do double template matching for one image, with the necessary recording
    and I/O necessary.
    
    Arguments:
    image_path - path to processed image file.
    templates - a sequence of two images, each serves as one template to match
        against the opened image.
    targets_path - base path of output targets file. Will be used to compose 
        the output file name.
    targets_frame - the output file name is <targets_path><targets_frame>_targets.
    """
    im = pl.imread(image_path)
    targs = TargetArray(2)
    
    pos = template_match(im, templates[0])
    record_target(targs[0], 0, templates[0], pos)
    
    # Blank out the first find in case it is similar to the second template.
    h, w = templates[0].shape
    im_blanked = im.copy()
    im_blanked[pos[1] - h/2 : pos[1] + h/2, pos[0] - w/2 : pos[0] + w/2] = 0
    
    pos = template_match(im_blanked, templates[1])
    record_target(targs[1], 1, templates[1], pos)
    
    targs.write(targets_path, targets_frame)

def process_frame(image_tmpl, templates, targets_tmpl, frame_num, cam_count):
    """
    Write targets found by template-matching for all cameras in a frame.
    
    Arguments:
    image_tmpl - a format string with two integer places. The first is replaced
        with camera number, the second with frame number.
    templates - a list, For each camera, a sequence of two images, each serves 
        as one template to match against the opened image.
    targets_tmpl - Format string with one int specifier. base path of output 
        targets file. Will be used to compose the output file name. The int 
        specifier is replaced with camera number.
    frame_num - frame number.
    cam_count - number of cameras in scene.
    """
    for cam in xrange(cam_count):
        tpath = targets_tmpl % (cam + 1)
        ipath = image_tmpl % (cam + 1, frame_num)
        process_image(ipath, templates[cam], tpath, frame_num)

def show_frame(image_tmpl, targets_tmpl, frame_num, cam_count):
    """
    Creates a figure with subplots for each camera, showing the targets read 
    for that camera on top of the seen image.
    
    Arguments:
    image_tmpl - a format string with two integer places. The first is replaced
        with camera number, the second with frame number.
    targets_tmpl - Format string with one int specifier. base path of input 
        targets file. Will be used to compose the output file name. The int 
        specifier is replaced with camera number.
    frame_num - frame number.
    cam_count - number of cameras in scene.
    """
    pl.figure(figsize=(15,15))
    vert_plots = floor(sqrt(cam_count))
    horz_plots = cam_count/vert_plots
    
    for cam in xrange(cam_count):
        pl.subplot(vert_plots, horz_plots, cam + 1)
        im = pl.imread(image_tmpl % (cam + 1, frame_num))
        pl.imshow(im, cmap=cm.gray)
        
        targs = read_targets(targets_tmpl % (cam + 1), frame_num)
        pl.plot(targs[0].pos()[0], targs[0].pos()[1], 'ro')
        pl.plot(targs[1].pos()[0], targs[1].pos()[1], 'ro')
    
    pl.axis('tight')
    pl.axis('off')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('tmpl', type=str, 
        help="Name template - a format string with two integer places, " + \
        "the first for camera number and the second for frame number.")
    parser.add_argument('targ_tmpl', type=str, 
        help="Targets template - a format string with one integer place, " + \
        "for camera number. Used to make output file names")    
    parser.add_argument('first', type=int, help="First frame number")
    parser.add_argument('last', type=int, help="Last frame number")
    parser.add_argument('--cams', '-c', type=int, default=4, 
        help="Number of cameras in scene")
    args = parser.parse_args()
    
    # First mark each in turn:
    templates = []
    for cam in xrange(args.cams):
        r, t = mark_image(args.tmpl % (cam + 1, args.first))
        templates.append(t)
    show_frame(args.tmpl, args.targ_tmpl, args.first, args.cams)
    pl.show()
    
    for frame in xrange(args.first, args.last + 1):
        process_frame(args.tmpl, templates, args.targ_tmpl, frame, args.cams)
    
