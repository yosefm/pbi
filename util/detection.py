# -*- coding: utf-8 -*-
"""
Detect inertial particles.

Created on Wed Jan  6 15:46:13 2016

@author: yosef
"""

from optv.tracking_framebuf import TargetArray, CORRES_NONE

import numpy as np
from skimage.feature import match_template, peak_local_max, blob_dog
from skimage.morphology import disk

def targetize(detects, approx_size, sumg=10):
    """
    Creates a correct TargetArray object with the detected positions and some
    placeholder values for target parameters that I don't use.
    
    Arguments:
    detects - (n,2) array, pixel coordinates of a detected target.
    approx_size - a value to use for the pixel size placeholders.
    sumg - a value to use for the sum of grey values placeholder.
        Default: 10.
    """
    targs = TargetArray(len(detects))
    
    tnum = 0
    for t, pos in zip(targs, detects):
        t.set_pos(pos)
        t.set_pnr(tnum)
        t.set_sum_grey_value(sumg) # whatever
        t.set_pixel_counts(approx_size**2 * 4, approx_size*2, approx_size*2)
        t.set_tnr(CORRES_NONE) # The official "correspondence not found" that 
                               # the rest of the code expects.
        tnum += 1
    
    return targs
    
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
    return targetize(peaks, approx_size)

def detect_blobs(image, approx_size=15, thresh=0.1):
    """
    A particle detection method based on the Difference of Gaussians algorithm.
    It is possibly more consistent than the method used in 
    ``detect_large_particles()``. 
    
    Arguments:
    image - the image to search for particles.
    approx_size - just a placeholder and there's a default so don't worry 
        about it.
    thresh - minimum grey value for blob pixels.
    
    Returns:
    a TargetArray with the detections.
    """
    blobs = blob_dog(image.T, max_sigma=5, threshold=thresh)
    return targetize(blobs, approx_size)
