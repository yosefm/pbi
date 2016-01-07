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

def detect_large_particles(image, approx_size=15):
    """
    A particle detection method based on template matching followed by peak 
    fitting. It is needed when particles are large, because the other methods
    assume that particles are small clumps of particles and can find multiple
    targets per large particle or find an inconsistent centre.
    
    Arguments:
    image - the image to search for particles.
    approx_size - search for particles whose pixel radius is around this value.
    
    Returns:
    a TargetArray with the detections.
    """
    sel = disk(approx_size)
    matched = match_template(image, sel, pad_input=True)
    matched[matched < 0.5] = 0
    peaks = np.c_[peak_local_max(matched)][:,::-1]
    targs = TargetArray(len(peaks))
    
    tnum = 0
    for t, pos in izip(targs, peaks):
        t.set_pos(pos)
        t.set_pnr(tnum)
        tnum += 1
    
    return targs