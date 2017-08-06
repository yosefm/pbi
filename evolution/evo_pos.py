# -*- coding: utf-8 -*-
"""
Evolutionary solution to the camera position. Based on camparing 3D points 
from converging rays to known positions. Spatial distances are weighted to 
favor low Ray Convergence Measure (RCM).

Created on Tue May 10 09:41:11 2016

@author: yosef
"""
from __future__ import print_function

import sys, numpy as np
from util.evolution import gen_calib, get_pos
from optv.transforms import convert_arr_pixel_to_metric, correct_arr_brown_affine
from optv.orientation import point_positions
from calib import correspondences

def fitness(solution, detections, glass_vecs, known_points, vpar, cpar, 
    search_radius=3, penalty=250.):
    """
    Checks the fitness of an evolutionary solution of calibration values to 
    target points. Fitness is the sum of squares of the distance from each 
    calculated point to the closest known point.
    
    To favor more correspondences while avoiding a solution between points, 
    pairs whose distance is more than the search radius are penalized.
    
    Arguments:
    solution - array, flattened rows for each camera. Each row concatenates: 
        position of intersection with Z=0 plane; 3 angles of exterior 
        calibration; primary point (xh,yh,cc); 3 radial distortion parameters; 
        2 decentering parameters.
    detections - a list of TargetArray objects, one per camera.
    glass_vecs - (c,3) array, one glass vector for each of c cameras. This is 
        the only part of the calibration struct that remains constant 
        throughout the iteration.
    known_points - (n,3) array, positions of known points seen by all cameras
        (may be composed by intersecting the per-camera lists) to compare 
        against.
    vpar - a VolumeParams object with observed volume size and criteria for 
        correspondence. Don't ask me why the forefathers put those in the same 
        place.
    cpar - a ControlParams object with image data.
    search_radius - acceptable distance between points to consider them a 
        valid pair.
    penalty - value added to fitness for each invalid pair.
    """
    num_cams = glass_vecs.shape[0]
    solution = solution.reshape(num_cams, -1)
    unfound_cost = penalty*1.5
    known_indices = np.r_[:known_points.shape[0]]
    
    # Transform solution vector to calibration objects:
    cals = []
    for cam in xrange(num_cams):
        # Breakdown of of agregate solution vector:
        inters = np.zeros(3)
        inters[:2] = solution[cam, :2]
        R = solution[cam, 2]
        angs = solution[cam, 3:6]
        prim_point = solution[cam, 6:9]
        rad_dist = solution[cam, 9:12]
        decent = solution[cam, 12:14]
        
        cal = gen_calib(inters, R, angs, glass_vecs[cam], prim_point, 
            rad_dist, decent)
        cals.append(cal)
    
    # Find 3D positions implied by targets:
    sets = correspondences(detections, cals, vpar, cpar)[0]
    if np.all([s.shape[1] == 0 for s in sets]):
        return known.shape[0] * unfound_cost
    
    points = []
    quality = []
    
    for qual, pset in enumerate(sets):
        if pset.shape[1] == 0:
            continue
        
        flat = []
        for cam, cal in enumerate(cals):
            cam_cent = cal.get_primary_point()[:2]
            
            unused = (pset[cam] == -999)
            metric = convert_arr_pixel_to_metric(pset[cam], cpar) - cam_cent
            flat.append(correct_arr_brown_affine(metric, cal))
            flat[-1][unused] = -999
        
        flat = np.array(flat)
        pos, rcm = point_positions(flat.transpose(1,0,2), cpar, cals)
        
        # Favor better correspondences:
        if qual > 0:
            rcm *= 3.*qual
        
        points.append(pos)
        quality.append(rcm)
    
    points = np.vstack(points)
    quality = np.hstack(quality)
        
    # Compare to known points. result is length n.
    dists = np.linalg.norm(points[None,:,:] - known_points[:,None,:], axis=-1)
    closest_found = np.argmin(dists, axis=1)
    closest_known = np.argmin(dists, axis=0)
    close_dists = np.full(known_points.shape[0], unfound_cost)
    matching_pairs = np.nonzero(closest_known == known_indices)[0]
    close_dists[matching_pairs] = dists[matching_pairs, closest_found[matching_pairs]]
    penalize = close_dists > search_radius
    quality = quality[closest_found]
        
    fits = close_dists * (1 + quality)
    fits[penalize] += penalty
    
    # Favor quads:
    
    return np.linalg.norm(fits)

def eprint(string):
    print(string, file = sys.stderr)

def show_current(signum, frame):
    num_cams = glass_vecs.shape[0]
    solution = init_sols[np.argmin(fits)].reshape(num_cams, -1)
    
    # Transform solution vector to calibration objects:
    cals = []
    for cam in xrange(num_cams):
        # Breakdown of of agregate solution vector:
        inters = np.zeros(3)
        inters[:2] = solution[cam, :2]
        R = solution[cam, 2]
        angs = solution[cam, 3:6]
        prim_point = solution[cam, 6:9]
        rad = solution[cam, 9:12]
        decent = solution[cam, 12:14]
        pos = get_pos(inters, R, angs)
        
        # Compare known points' projections to detections:
        cal = gen_calib(inters, R, angs, glass_vecs[cam], prim_point, 
            rad, decent)
        cals.append(cal)
    
        eprint("")
        eprint(solution)
        eprint("pos/ang:")
        eprint("%.8f %.8f %.8f" % tuple(pos))
        eprint("%.8f %.8f %.8f" % tuple(angs))
        eprint("")
        eprint("internal: %.8f %.8f %.8f" % tuple(prim_point))
        eprint("radial distortion: %.8f %.8f %.8f" % tuple(rad))
        eprint("decentering: %.8f %.8f" % tuple(decent))
        
    # Find 3D positions implied by targets:
    sets = correspondences(detections, cals, vpar, cpar)[0]
    if np.all([s.shape[1] == 0 for s in sets]):
        return 
    
    for qual, pset in enumerate(sets):
        if pset.shape[1] == 0:
            continue
        
        flat = []
        for cam, cal in enumerate(cals):
            cam_cent = cal.get_primary_point()[:2]
            
            unused = (pset[cam] == -999)
            metric = convert_arr_pixel_to_metric(pset[cam], cpar)
            flat.append(correct_arr_brown_affine(metric, cal))
            flat[-1][unused] = -999
        
        flat = np.array(flat)
        pos, rcm = point_positions(flat.transpose(1,0,2), cpar, cals)
        sort = np.argsort(pos[:,1])
        eprint(str(pos[sort]))
        eprint(str(rcm[sort]))
    

def intersect_known_points(point_list):
    """
    Return an array of points appearing in all lists.
    
    Arguments:
    point_lists - a list of 2D arrays whose second dim is 3.
    """
    current_set = point_list[0]
    for points in point_list[1:]:
        in_both = np.all(current_set[None,:,:] == points[:,None,:], axis=2)
        current_set = current_set[np.any(in_both, axis=0)]
    return current_set
    
if __name__ == "__main__":
    import argparse, yaml, matplotlib.pyplot as pl
    import numpy.random as rnd
    
    from optv.parameters import VolumeParams, ControlParams, TargetParams
    from optv.segmentation import target_recognition
    from util.evolution import cauchy_mutation, recombination, choose_breeders
    from util.openptv import simple_highpass

    import signal
    signal.signal(signal.SIGTSTP, show_current)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', 
        help="A YAML file with calibration and image properties.")
    args = parser.parse_args()
    
    yaml_args = yaml.load(file(args.config))
    
    control_args = yaml_args['scene']
    cam_args = yaml_args['cameras']
    control_args['cams'] = len(cam_args)
    cpar = ControlParams(**control_args)
    vpar = VolumeParams()
    vpar.read_volume_par(yaml_args['volume_params'])
    targ_par = TargetParams(**yaml_args['detection'])
    
    # Per-camera constant data (detections, gene bounds, glass vector).
    glass_vecs = []
    bounds = []
    detections = []
    known = []
    for cix, cam_spec in enumerate(cam_args):
        img = pl.imread(cam_spec['image'])
        hp = simple_highpass(img, cpar)
        targs = target_recognition(hp, targ_par, cix, cpar)
        
        known.append(np.loadtxt(cam_spec['known_points'])[:,1:])
        glass_vecs.append(np.r_[cam_spec['glass_vec']])
        bounds.extend(cam_spec['bounds'])
        detections.append(targs)
    
    glass_vecs = np.array(glass_vecs)
    bounds = np.array(bounds)
    known = intersect_known_points(known)
    #print known.shape
    #print known
    
    # Initialize population.
    pop_size = 1000
    ranges = np.r_[[(maxb - minb) for minb, maxb in bounds]]

    init_sols = np.array([
        np.r_[[rnd.rand()*(maxb - minb) + minb for minb, maxb in bounds]
        ] for s in xrange(pop_size)])
    
    fits = []
    for sol in init_sols:
        fits.append(fitness(sol, detections, glass_vecs, known, vpar, cpar) )
    fits = np.array(fits)
    eprint(fits)
    
    mutation_chance = 0.01
    num_iters = 2000000
    for it in xrange(num_iters):
        if it % 500 == 0:
            sys.stderr.write("min %g, max %g\n" % (fits.min(), fits.max()))
            
        # Choose breeders, chance to be chosen weighted by inverse fitness
        breeders = choose_breeders(fits)
        if breeders is None:
            break
            
        # breed into losers
        loser = fits.argmax()
        newsol = recombination(*init_sols[breeders])
        cauchy_mutation(newsol, bounds, mutation_chance)
        
        newfit = fitness(newsol, detections, glass_vecs, known, vpar, cpar)
        
        # Niching to avoid early convergence:
        dist = np.linalg.norm((newsol - init_sols)/ranges, axis=1).min()
        if newfit > fits[loser]:
            continue
        
        init_sols[loser] = newsol
        fits[loser] = newfit
    
    show_current(0, 0)
