# -*- coding: utf-8 -*-
"""
Basic operations for evolutionary algorithms. 

Created on Wed Mar  9 10:56:55 2016

@author: yosef

references:
[1] Koenig A.C, A Study of Mutation Methods for Evolutionary Algorithms; 2002
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

def get_polar_rep(pos, angs):
    """
    Returns the point of intersection with zero Z plane, and distance from it.
    """
    s = np.sin(angs)
    c = np.cos(angs)
    zdir = -np.r_[ s[1], -c[1]*s[0], c[1]*c[0] ]
    
    c = -pos[2]/zdir[2]
    inters = pos + c*zdir
    R = np.linalg.norm(inters - pos)
    
    return inters[:2], R
    
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

def choose_breeders(fits, minimize=True):
    """
    Chooses two breeers, with chance to breed based on fitness. The fitter 
    you are, the more chance to breed. If 'minimize' is True, fitness chance
    is calculated s.t. lower 'fitness' has higher chance.
    
    Returns:
    Two indixes into the fits array.
    """
    ranking = np.argsort(fits)
    fit_range = np.add.reduce(fits) - fits.min()
    fits_normed = (fits - fits.min())/fit_range
    
    if minimize:
        fits_normed = fits_normed.max()*1.05 - fits_normed[ranking]
    
    ranked_fit = np.add.accumulate(fits_normed)
    if not ranked_fit.any():
        print ranked_fit
        return None

    breeding_dice = rnd.rand(2) * ranked_fit[-1]
    breeders = ranking[np.digitize(breeding_dice, ranked_fit)]
    
    return breeders

def cauchy_mutation(solution, bounds, chance, stds_in_range=5):
    """
    The Cauchy mutation operator prefers small mutations, unlike the naive
    uniform-distribution ``mutation()``. However, it still has heavy tails, 
    making it more globally oriented than the often used Normal distribution.
    
    See also: [1]
    
    Arguments:
    solution - a vector of v decision variables.
    bounds - (v,2) array, fir min, max bound of each variable.
    chance - the chance of any gene in the chromosome to undergo mutation.
    stds_in_range - scales each mutation by (variable range / stds_in_range).
    """
    genes = rnd.rand(len(solution)) < chance
    mute = solution[genes]
    bdg = bounds[genes]
    
    mute += (bdg[:,1] - bdg[:,0]) / stds_in_range * \
        rnd.standard_cauchy(genes.sum())
    
    under = mute < bdg[:,0]
    over = mute > bdg[:,1]
    mute[under] = bdg[under,0]
    mute[over] = bdg[over,0]
    
    solution[genes] = mute
    
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
