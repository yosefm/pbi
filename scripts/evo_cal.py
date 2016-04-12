# -*- coding: utf-8 -*-
"""
Evolutionary solution to the camera position.\

Created on Thu Jun  4 10:06:30 2015

@author: yosef
"""
import signal

import numpy as np, matplotlib.pyplot as pl
import numpy.random as rnd
from matplotlib import cm

from calib import pixel_2D_coords, simple_highpass, detect_ref_points
from mixintel.evolution import gen_calib, get_pos, mutation, recombination, choose_breeders

wrap_it_up = False
def interrupt(signum, frame):
    global wrap_it_up
    wrap_it_up = True
    
signal.signal(signal.SIGINT, interrupt)
    
def fitness(solution, calib_targs, calib_detect, glass_vec, cpar):
    """
    Checks the fitness of an evolutionary solution of calibration values to 
    target points. Fitness is the sum of squares of the distance from each 
    guessed point to the closest neighbor.
    
    Arguments:
    solution - array, concatenated: position of intersection with Z=0 plane; 3 
        angles of exterior calibration; primary point (xh,yh,cc); 3 radial
        distortion parameters; 2 decentering parameters.
    calib_targs - a (p,3) array of p known points on the calibration target.
    calib_detect - a (d,2) array of d detected points in the calibration 
        target.
    cpar - a ControlParams object with image data.
    """
    # Breakdown of of agregate solution vector:
    inters = np.zeros(3)
    inters[:2] = solution[:2]
    R = solution[2]
    angs = solution[3:6] 
    prim_point = solution[6:9]
    rad_dist = solution[9:12]
    decent = solution[12:14]
        
    # Compare known points' projections to detections:
    cal = gen_calib(inters, R, angs, glass_vec, prim_point, rad_dist, decent)
    known_2d = pixel_2D_coords(cal, calib_targs, cpar)
    dists = np.linalg.norm(
        known_2d[None,:,:] - calib_detect[:,None,:], axis=2).min(axis=0)
    
    return np.sum(dists**2)

def show_current(signum, frame):
    """
    Takes the best-fit current to call time, and displays the current 
    calibration that produces the fit, and graphs the known/detected points
    for visual match check.
    """
    import __main__
    
    fits = __main__.fits
    cal_points = __main__.cal_points
    hp = __main__.hp
    targs = __main__.targs
    
    best_fit = np.argmin(fits)
    inters = np.zeros(3)
    inters[:2] = init_sols[best_fit][:2]
    R = init_sols[best_fit][2]
    angs = init_sols[best_fit][3:6]
    pos = get_pos(inters, R, angs)
    prim = init_sols[best_fit][6:9]
    rad = init_sols[best_fit][9:12]
    decent = init_sols[best_fit][12:14]

    print
    print fits[best_fit]
    print "pos/ang:"
    print "%.8f %.8f %.8f" % tuple(pos)
    print "%.8f %.8f %.8f" % tuple(angs)
    print
    print "internal: %.8f %.8f %.8f" % tuple(prim)
    print "radial distortion: %.8f %.8f %.8f" % tuple(rad)
    print "decentering: %.8f %.8f" % tuple(decent)
    
    cal = gen_calib(inters, R, angs, glass_vec, prim, rad, decent)
    init_xs, init_ys = pixel_2D_coords(cal, cal_points, cpar).T
    
    pl.imshow(hp, cmap=cm.gray)
    pl.scatter(targs[:,0], targs[:,1])
    pl.scatter(init_xs, init_ys, c='r')
    pl.scatter(init_xs[[0,-1]], init_ys[[0,-1]], c='y')

    pl.show()

# Main part
import sys, yaml
from mixintel.openptv import control_params

yaml_file = sys.argv[1]
yaml_args = yaml.load(file(yaml_file))
control_args = yaml_args['scene']
cpar = control_params(**control_args)
cam = yaml_args['target']['number']

fname = yaml_args['target']['image']
calblock_name = yaml_args['target']['known_points']
glass_vec = np.r_[yaml_args['target']['glass_vec']]

image = pl.imread(fname)
hp = simple_highpass(image, cpar)

tarr = detect_ref_points(hp, cam, cpar)
targs = np.array([t.pos() for t in tarr])

pop_size = 2500
#bounds = [(100., 200.), (-120, 0), (-150, -400), (-1, 0), (-1, 0), (-0.1, 0.1)]
#bounds = [(100., 200.), (-120, 0), (150, 400), (np.pi - 1, np.pi + 1), (-1, 0), (-0.1, 0.1)]
##bounds = [(100., 200.), (-120, 0), (-150, -400), (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
if cam == 0:
    bounds = [(-20.,20.), (-20.,20.), # offset
              (210.,300.), # R
              (-0., 0.6), (-0.6, 0.), (-0.5, 0.5), # angles
              (-0.05, 0.05), (-0.05, 0.05), (60, 100), # primary point
              (-1e-5, 1e-5), (-1e-5, 1e-5), (-1e-5, 1e-5), # radial distortion
              (-1e-6, 1e-6), (-1e-6, 1e-6) # decentering
    ]
    cal_points = np.loadtxt(calblock_name)[:,1:]
elif cam == 1:
    bounds = [(-20.,20.), (-20.,20.), # offset
              (210.,300.), # R
              (-0., 0.6), (-0., 0.6), (-0.5, 0.5), # angles
              (-0.05, 0.05), (-0.05, 0.05), (60, 100), # primary point
              (-1e-5, 1e-5), (-1e-5, 1e-5), (-1e-5, 1e-5), # radial distortion
              (-1e-6, 1e-6), (-1e-6, 1e-6) # decentering
    ]
    cal_points = np.loadtxt(calblock_name)[:,1:]
elif cam == 2:
    bounds = [(-20.,20.), (-20.,20.), # offset
              (210.,300.), # R
              (-0.6, 0.), (-0.6, 0.), (-0.5, 0.5), # angles
              (-10., 10.), (-10., 10.), (60, 100), # primary point
              (-2e-4, 2e-4), (-1e-4, 1e-4), (-1e-4, 1e-4), # radial distortion
              (-1e-4, 1e-4), (-1e-4, 1e-4) # decentering
    ]
    cal_points = np.loadtxt(calblock_name)[:,1:]
elif cam == 3:
    bounds = [(-20.,20.), (-20.,20.), # offset
              (210.,300.), # R
              (-0.6, 0.), (-0., 0.6), (-0.5, 0.5), # angles
              (-0.05, 0.05), (-0.05, 0.05), (60, 100), # primary point
              (-1e-5, 1e-5), (-1e-5, 1e-5), (-1e-5, 1e-5), # radial distortion
              (-1e-6, 1e-6), (-1e-6, 1e-6) # decentering
    ]
    cal_points = np.loadtxt(calblock_name)[:,1:]

ranges = np.r_[[(maxb - minb) for minb, maxb in bounds]]

init_sols = np.array([
    np.r_[[rnd.rand()*(maxb - minb) + minb for minb, maxb in bounds]
    ] for s in xrange(pop_size)])

fits = []
for sol in init_sols:
    fits.append( fitness(sol, cal_points, targs, glass_vec, cpar) )
fits = np.array(fits)
print fits

signal.signal(signal.SIGTSTP, show_current)

mutation_chance = 0.05
niche_size = len(bounds) / 2.
niche_penalty = 2.
num_iters = 1000000
for it in xrange(num_iters):
    if it % 100 == 0:
        niche_size *= 0.996
        niche_penalty = niche_penalty**0.9995
    if it % 500 == 0:
        print fits.min(), fits.max()
        print niche_size, niche_penalty
    
    # Check if Ctrl-C event happened during previous iteration:
    if wrap_it_up:
        break

    # Choose breeders, chance to be chosen weighted by inverse fitness
    breeders = choose_breeders(fits)
    if breeders is None:
        break
        
    # breed into losers
    loser = fits.argmax()
    newsol = recombination(*init_sols[breeders])
    mutation(newsol, bounds, mutation_chance)
    
    newfit = fitness(newsol, cal_points, targs, glass_vec, cpar)
    
    # Niching to avoid early convergence:
    dist = np.linalg.norm((newsol - init_sols)/ranges, axis=1).min()
    if dist < niche_size:
        newfit *= niche_penalty
        #print "niching", newfit
    
    if newfit > fits[loser]:
        continue
    
    init_sols[loser] = newsol
    fits[loser] = newfit

import inspect
show_current(0, inspect.currentframe())
