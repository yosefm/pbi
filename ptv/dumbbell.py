# -*- coding: utf-8 -*-
"""
Guided recognition of dumbbell targets.

Loosely based on the Matlab version from Alex et al. here:
https://bitbucket.org/alexliberzonlab/ptv_postproc/src/106a97675479ad9afa95a1837bb5a6f9ff7de024/Matlab/?at=master
Files: tau_dumbbell_detection_db_y1.m and those called from within.

Innovations in this code:
1. Template matching is done on a search region around the marked target rather
   than on the whole image. The search region is continuously updated to follow
   the found targets.
2. Multithreaded processing reduces real time, although due to Python's GIL 
   it's not as fast as it could be.

Author: Yosef Meller.
"""
import numpy as np
from scipy import signal
from math import floor, ceil, sqrt
import os

import matplotlib.pyplot as pl, matplotlib.cm as cm
from matplotlib.widgets import  RectangleSelector

from optv.tracking_framebuf import TargetArray, read_targets

def onselect(eclick, erelease, im, rects, tiles):
    xs = np.sort(np.r_[eclick.xdata, erelease.xdata])
    ys = np.sort(np.r_[eclick.ydata, erelease.ydata])
    rects.append(np.c_[xs, ys].flatten())
    print((rects[-1]))
    
    ybounds = list(map(int, [floor(ys[0]), ceil(ys[1])]))
    xbounds = list(map(int, [floor(xs[0]), ceil(xs[1])]))
    tiles.append(im[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]].copy())
    if len(rects) == 2:
        pl.close()

def mark_image(image_path):
    rects = []
    tiles = []

    pl.figure()
    ax = pl.subplot(1,1,1)

    im = pl.imread(image_path)
    ax.imshow(im, cmap=cm.gray)
    pl.title(image_path)
    
    sel_style = {'edgecolor': 'red', 'fill': False}
    rs = RectangleSelector(ax, lambda clk, rls: onselect(clk, rls, im, rects, tiles), 
        props=sel_style)
    rs.set_active(True)

    pl.show()
    return rects, tiles

def template_match(image, template, rect=None, margin_factor=1.5):
    """
    Finds the center of image area matching a template. Optionally only 
    searches an area around a given rect. In this case, the rect is updated in
    place to mark the next-frame search region. 
    
    There is a possible race condition when using rect in a multithreaded run,
    when two threads update the rect of the same frame. It is bearable because 
    of the assumption that the rect moves slowly (and frames are started in 
    order), the margin is wide enough, and because its size is always 
    recalculated to the template size.
    
    Arguments:
    image - grayscale image as read by pl.imread.
    template - a smaller image to be found in the larger.
    rect - optionsl 2x2 array, [[left, top], [right, bottom]]
    margin_factor - multiply rect width/height by that much.
    
    Returns:
    x, y - position of best match.
    """
    tmpl = (template - template.mean())[::-1, ::-1]
    if rect is not None:
        xys = rect.reshape(2,2)
        lim = np.array(image.shape)[::-1]
        
        search_center = 0.5*xys.sum(axis=0)
        cent_rect = np.array(template.shape)[::-1]/2 * np.c_[[-1, 1]]
        
        sr = np.int_(
            np.clip(margin_factor * cent_rect + search_center, [0, 0], lim))
        clipped_center = 0.5*sr.sum(axis=0)
        image = image[sr[0,1]:sr[1,1], sr[0,0]:sr[1,0]]
    
    corr = signal.fftconvolve(image - image.mean(), tmpl, mode='same')
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    
    if rect is not None:
        y += clipped_center[1] - image.shape[0]/2
        x += clipped_center[0] - image.shape[1]/2
        rect[:] = (cent_rect + np.r_[x, y]).flatten()
    
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

def process_image(image_path, templates, targets_path, targets_frame, 
    rects=None):
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
    rects - the rectangles circumscribing each template, each a 2x2 array, 
        [[left, top], [right, bottom]]. If None (default), matches whole image.
    """
    im = pl.imread(image_path)
    targs = TargetArray(2)
    
    pos = template_match(im, templates[0], 
        rects[0] if rects is not None else None)
    record_target(targs[0], 0, templates[0], pos)
    
    if rects is None:
        # Blank out the first find in case it is similar to the second template.
        h, w = templates[0].shape
        im_blanked = im.copy()
        im_blanked[pos[1] - h/2 : pos[1] + h/2, pos[0] - w/2 : pos[0] + w/2] = 0
    else: 
        im_blanked = im
    
    pos = template_match(im_blanked, templates[1], 
        rects[1] if rects is not None else None)
    record_target(targs[1], 1, templates[1], pos)
    
    # Alex: on windows provide full path
    # print(os.path.abspath(targets_path))
    targs.write(targets_path.encode(), targets_frame)
    

def process_frame(image_tmpl, templates, targets_tmpl, frame_num, cam_count,
    rects=None):
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
    rects - the rectangles circumscribing each template, a list of cam_count
        arrays each 2x2, [[left, top], [right, bottom]]. If None (default), 
        matches whole image.
    """
    for cam in range(cam_count):
        tpath = targets_tmpl % (cam)
        ipath = image_tmpl % (cam, frame_num)
        process_image(ipath, templates[cam], tpath, frame_num,
            rects[cam] if rects is not None else None)

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
    vert_plots = int(floor(sqrt(cam_count)))
    horz_plots = int(cam_count/vert_plots)
    
    for cam in range(cam_count):
        pl.subplot(vert_plots, horz_plots, cam + 1)
        im = pl.imread(image_tmpl % (cam, frame_num))
        pl.imshow(im, cmap=cm.gray)
        
        targs = read_targets(targets_tmpl % (cam), frame_num)
        pl.plot(targs[0].pos()[0], targs[0].pos()[1], 'ro')
        pl.plot(targs[1].pos()[0], targs[1].pos()[1], 'ro')
    
    pl.axis('tight')
    pl.axis('off')

if __name__ == "__main__":
    import threading
    
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
    parser.add_argument('--full', action='store_true', default=False,
        help="Match full image rather than the area following the target.")
    parser.add_argument('--threads', type=int, default=6, 
        help="Max. number of concurrent worker threads.")
    
    parser.add_argument('--cache-file', '-f', type=str, 
        help="Save initial positions to this file.")
    parser.add_argument('--positions', '-p', type=str,
        help="Load premarked positions from this file.")
    args = parser.parse_args()
    
    # Initialize search regions, either by loading or graphically marking:
    templates = []
    rects = []
    if args.positions is not None:
        premarks = np.loadtxt(args.positions)
        
    for cam in range(args.cams):
        if args.positions is None:
            r, t = mark_image(args.tmpl % (cam, args.first))
            # r, t = mark_image(args.tmpl % (cam, args.first))
        else:
            r = premarks[2*cam:2*(cam)]
            r[:,:2] = np.floor(r[:,:2])
            r[:,2:] = np.floor(r[:,2:])
            r= np.int_(r)
            
            image = pl.imread(args.tmpl % (cam, args.first))
            t = [
                image[r[0,1]:r[0,3], r[0,0]:r[0,2]],
                image[r[1,1]:r[1,3], r[1,0]:r[1,2]],
            ]
        
        templates.append(t)
        # `rects` is a list of lists.
        rects.append([r[0], r[1]])
    
    if args.cache_file is not None:
        np.savetxt(args.cache_file, np.array(rects).reshape(-1,4))
    
    if args.positions is None:
        process_frame(args.tmpl, templates, args.targ_tmpl, args.first, args.cams,
            None if args.full else rects)
        show_frame(args.tmpl, args.targ_tmpl, args.first, args.cams)
        pl.show()
    
    # Start one thread per frame. Keep a maximum of live threads and add one
    # new job each time a job is finished.
    active = []
    for frame in range(args.first, args.last + 1):
        arglist = (args.tmpl, templates, args.targ_tmpl, frame, args.cams)
        if not args.full:
            arglist = arglist + (rects,)
        
        t = threading.Thread(target=process_frame, args=arglist)
        t.start()
        active.append(t)
        
        if len(active) == args.threads:
            active[0].join()
            active.pop(0)
    
    for t in active:
        t.join()
