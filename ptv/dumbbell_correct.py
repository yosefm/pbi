# -*- coding: utf-8 -*-
"""
Perform dumbbell calibration from existing target files, using a subset of
the camera set, assuming some cameras are known to have moved and some to have
remained relatively static (but we can alternate on subsequent runs).

Created on Tue Dec 15 13:39:40 2015

@author: yosef
"""

# These readers should go in a nice module, but I wait on Max to finish the 
# proper bindings.

import os
from optv.orientation import dumbbell_target_func
from optv.parameters import ControlParams

def calib_convergence(calib_vec, targets, calibs, active_cams, cpar,
    db_length, db_weight):
    """
    Mediated the ray_convergence function and the parameter format used by 
    SciPy optimization routines, by taking a vector of variable calibration
    parameters and pouring it into the Calibration objects understood by 
    OpenPTV.
    
    Arguments:
    calib_vec - 1D array. 3 elements: camera 1 position, 3 element: camera 1 
        angles, next 6 for camera 2 etc.
    targets - a (c,t,2) array, for t target metric positions in each of c 
        cameras.
    calibs - an array of per-camera Calibration objects. The permanent fields 
        are retained, the variable fields get overwritten.
    active_cams - a sequence of True/False values stating whether the 
        corresponding camera is free to move or just a parameter.
    cpar - a ControlParams object describing the overall setting.
    db_length - expected distance between two dumbbell points.
    db_weight - weight of the distance error in the target function.
    
    Returns:
    The weighted ray convergence + length error measure.
    """
    calib_pars = calib_vec.reshape(-1, 2, 3)
    
    for cam, cal in enumerate(calibs):
        if not active_cams[cam]:
            continue
        
        # Pop a parameters line:
        pars = calib_pars[0]
        calib_pars = calib_pars[1:]
        
        cal.set_pos(pars[0])
        cal.set_angles(pars[1])
    
    return dumbbell_target_func(targets, cpar, calibs, db_length, db_weight)

if __name__ == "__main__":
    from optv.calibration import Calibration
    from optv.tracking_framebuf import read_targets
    from optv.transforms import convert_arr_pixel_to_metric
    
    import yaml, numpy as np
    from scipy.optimize import minimize
    
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, default="dumbbell.yaml",
        help="path to YAML configuration file.")
    parser.add_argument('--clobber', action='store_true', default=False,
        help="Replace original ori files with result.")
    cli_args = parser.parse_args()
    yaml_args = yaml.load(open(cli_args.config,'r'),yaml.CLoader)
    
    # Generate initial-guess calibration objects. These get overwritten by
    # the optimizer's target function.
    cal_args = yaml_args['initial']
    calibs = []
    active = []
    
    for cam_data in cal_args:
        cl = Calibration()
        cl.from_file(cam_data['ori_file'].encode(), cam_data['addpar_file'].encode())
        
        calibs.append(cl)
        active.append(cam_data['free'])
    
    scene_args = yaml_args['scene']
    scene_args['cams'] = len(cal_args)
    cpar = ControlParams(**scene_args)
    
    db_length = yaml_args['dumbbell']['length']
    db_weight = yaml_args['dumbbell']['weight']
    
    # Soak up all targets to memory. Not perfect but how OpenPTV wants it.
    # Well, use a limited clip, ok?
    num_frames = yaml_args['last'] - yaml_args['first'] + 1
    all_targs = [[] for pt in range(num_frames*2)] # 2 targets per fram
    
    for cam in range(len(cal_args)):
        for frame in range(num_frames):
            targ_file = yaml_args['template'] % (cam)
            print(os.path.abspath(targ_file))
            targs = read_targets(targ_file, yaml_args['first'] + frame)
            
            for tix, targ in enumerate(targs):
                all_targs[frame*2 + tix].append(targ.pos())
    
    all_targs = np.array([convert_arr_pixel_to_metric(np.array(targs), cpar) \
        for targs in all_targs])
    assert(all_targs.shape[1] == len(cal_args) and all_targs.shape[2] == 2)
    
    # Generate initial guess vector and bounds for optimization:
    num_active = np.sum(active)
    calib_vec = np.empty((num_active, 2, 3))
    active_ptr = 0
    for cam in range(len(cal_args)):
        if active[cam]:
            calib_vec[active_ptr,0] = calibs[cam].get_pos()
            calib_vec[active_ptr,1] = calibs[cam].get_angles()
            active_ptr += 1
        
        # Positions within a neighbourhood of the initial guess, so we don't 
        # converge to the trivial solution where all cameras are in the same 
        # place.
    calib_vec = calib_vec.flatten()
    
    # Test optimizer-ready target function:
    print("Initial values (1 row per camera, pos, then angle):")
    print(calib_vec.reshape(len(cal_args),-1))
    print("Current target function (to minimize):", end=' ')
    print(calib_convergence(calib_vec, all_targs, calibs, active, cpar,
        db_length, db_weight))
    
    # Optimization:
    res = minimize(calib_convergence, calib_vec, 
                   args=(all_targs, calibs, active, cpar, db_length, db_weight),
                   tol=1, options={'maxiter': 1000})
    
    print("Result of minimize:")
    print(res.x.reshape(len(cal_args),-1))
    print("Success:", res.success, res.message)
    print("Final target function:", end=' ')
    print(calib_convergence(res.x, all_targs, calibs, active, cpar,
        db_length, db_weight))
    
    if cli_args.clobber:
        x = res.x.reshape(-1,2,3)
        for cam in range(len(cal_args)):
            if active[cam]:
                # Make sure 'minimize' didn't play around:
                calibs[cam].set_pos(x[0,0])
                calibs[cam].set_angles(x[0,1])
                calibs[cam].write(cal_args[cam]['ori_file'].encode(), 
                    cal_args[cam]['addpar_file'].encode())
                x = x[1:]
