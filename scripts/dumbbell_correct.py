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

from optv.parameters import ControlParams
from calib import dumbbell_target_func

def control_params(**control_args):
    """
    Generates an OpenPTV ControlParams object from its field values.
    
    Arguments:
    control_args - a dictionary with the recognized keys. Currently these are:
        flags, image_size, pixel_size, cam_side_n, wall_ns, object_side_n, 
        wall_thicks, cams
    """
    control_args.setdefault('cams', 1)
    control = ControlParams( control_args['cams'])
    
    control.set_hp_flag( 1 if 'hp' in control_args['flags'] else 0)
    control.set_allCam_flag( 1 if 'allcam' in control_args['flags'] else 0)
    control.set_tiff_flag( 1 if 'headers' in control_args['flags'] else 0)
    control.set_imx(control_args['image_size'][0])
    control.set_imy(control_args['image_size'][1])
    control.set_pix_x(control_args['pixel_size'][0])
    control.set_pix_y(control_args['pixel_size'][1])
    control.set_chfield(0)
    
    layers = control.get_multimedia_params()
    layers.set_lut(0)
    layers.set_n1(control_args['cam_side_n'])
    layers.set_n2(control_args['wall_ns'])
    layers.set_n3(control_args['object_side_n'])
    layers.set_d(control_args['wall_thicks'])
    layers.set_nlay(len(control_args['wall_ns']))
    
    return control

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
    from calib import image_coords_metric
    
    import yaml, numpy as np
    from scipy.optimize import minimize
    
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config', type=str, default="dumbbell.yaml",
        help="path to YAML configuration file.")
    cli_args = parser.parse_args()
    yaml_args = yaml.load(file(cli_args.config))
    
    # Generate initial-guess calibration objects. These get overwritten by
    # the optimizer's target function.
    cal_args = yaml_args['initial']
    calibs = []
    active = []
    
    for cam_data in cal_args:
        cl = Calibration()
        cl.from_file(cam_data['ori_file'], cam_data['addpar_file'])
        
        calibs.append(cl)
        active.append(cam_data['free'])
    
    scene_args = yaml_args['scene']
    scene_args['cam'] = len(cal_args)
    cpar = control_params(**scene_args)
    
    db_length = yaml_args['dumbbell']['length']
    db_weight = yaml_args['dumbbell']['weight']
    
    # Soak up all targets to memory. Not perfect but how OpenPTV wants it.
    # Well, use a limited clip, ok?
    num_frames = yaml_args['last'] - yaml_args['first'] + 1
    all_targs = [[] for pt in xrange(num_frames*2)] # 2 targets per fram
    
    for cam in xrange(len(cal_args)):
        for frame in xrange(num_frames):
            targ_file = yaml_args['template'] % (cam + 1)
            targs = read_targets(targ_file, yaml_args['first'] + frame)
            
            for tix, targ in enumerate(targs):
                all_targs[frame*2 + tix].append(targ.pos())
    
    all_targs = np.array([image_coords_metric(np.array(targs), cpar) \
        for targs in all_targs])
    assert(all_targs.shape[1] == 4 and all_targs.shape[2] == 2)
    
    # Generate initial guess vector and bounds for optimization:
    num_active = np.sum(active)
    calib_vec = np.empty((num_active, 2, 3))
    active_ptr = 0
    for cam in xrange(len(cal_args)):
        if active[cam]:
            calib_vec[active_ptr,0] = calibs[cam].get_pos()
            calib_vec[active_ptr,1] = calibs[cam].get_angles()
        
        # Positions within a neighbourhood of the initial guess, so we don't 
        # converge to the trivial solution where all cameras are in the same 
        # place.
    calib_vec = calib_vec.flatten()
    
    # Test optimizer-ready target function:
    print calib_vec
    print calib_convergence(calib_vec, all_targs, calibs, active, cpar,
        db_length, db_weight)
    
    # Optimization:
    res = minimize(calib_convergence, calib_vec, 
                   args=(all_targs, calibs, active, cpar, db_length, db_weight),
                   tol=1, options={'maxiter': 500})
    print res.x, res.success, res.message
    print calib_convergence(res.x, all_targs, calibs, active, cpar,
        db_length, db_weight)
