# -*- coding: utf-8 -*-
"""
Basic operations for evolutionary algorithms. 

Created on Wed Mar  9 10:56:55 2016

@author: yosef
"""

import numpy as np
import numpy.random as rnd
from optv.calibration import Calibration

def get_pos(inters, R, angs):
    # Transpose of http://planning.cs.uiuc.edu/node102.html
    # Also consider the angles are reversed when moving from camera frame to
    # global frame.
    s = np.sin(angs)
    c = np.cos(angs)
    pos = inters + R*np.r_[ s[1], -c[1]*s[0], c[1]*c[0] ]
    return pos
    
def gen_calib(inters, R, angs, glass_vec, prim_point, radial_dist, decent):
    pos = get_pos(inters, R, angs)
    cal = Calibration()
    cal.set_pos(pos)
    cal.set_angles(angs)
    cal.set_primary_point(prim_point)
    cal.set_radial_distortion(radial_dist)
    cal.set_decentering(decent)
    cal.set_affine_trans(np.r_[1,0])
    cal.set_glass_vec(glass_vec)

    return cal

def mutation(solution, bounds, chance):
    genes = rnd.rand(len(solution)) < chance
    for gix in xrange(len(solution)):
        if genes[gix]:
            minb, maxb = bounds[gix]
            solution[gix] = rnd.rand()*(maxb - minb) + minb

def recombination(sol1, sol2):
    genes = rnd.random_integers(0, 1, len(sol1))
    ret = sol1.copy()
    ret[genes == 1] = sol2[genes == 1]
    return ret
