# -*- coding: utf-8 -*-
"""
Detect inertial particles.

Created on Wed Jan  6 15:46:13 2016

@author: yosef
"""
from itertools import izip
from optv.tracking_framebuf import TargetArray

import numpy as np
from skimage.feature import match_template, peak_local_max
from skimage.morphology import disk

def detect_large_particles(image, approx_size=15, peak_thresh=0.5):
    """
    A particle detection method based on template matching followed by peak 
    fitting. It is needed when particles are large, because the other methods
    assume that particles are small clumps of particles and can find multiple
    targets per large particle or find an inconsistent centre.
    
    Arguments:
    image - the image to search for particles.
    approx_size - search for particles whose pixel radius is around this value.
    peak_thresh - minimum grey value for a peak to be recognized.
    
    Returns:
    a TargetArray with the detections.
    """
    sel = disk(approx_size)
    matched = match_template(image, sel, pad_input=True)
    peaks = np.c_[peak_local_max(matched, threshold_abs=peak_thresh)][:,::-1]
    targs = TargetArray(len(peaks))
    
    tnum = 0
    for t, pos in izip(targs, peaks):
        t.set_pos(pos)
        t.set_pnr(tnum)
        t.set_sum_grey_value(10) # whatever
        t.set_pixel_counts(approx_size**2 * 4, approx_size*2, approx_size*2)
        t.set_tnr(-1) # The official "correspondence not found" that the rest
                      # of the code expects.
        tnum += 1
    
    return targs