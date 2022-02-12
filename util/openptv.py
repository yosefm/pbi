# -*- coding: utf-8 -*-
"""
Code for interacting with liboptv that is pure Python and requires no 
compilation. Counterpart of calib.pyx in that it could be in liboptv but for
the bureaucratic burden.

Created on Sun Jan 31 14:06:13 2016

@author: yosef
"""
import yaml
import numpy as np
from imageio import imread

from optv.parameters import ControlParams, TargetParams
from optv.calibration import Calibration
from optv.image_processing import preprocess_image
from optv.segmentation import target_recognition
from optv.tracking_framebuf import CORRES_NONE

def simple_highpass(img, cpar):
    return preprocess_image(img, 0, cpar, 12)
    
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

def count_unused_targets(targs):
    """
    Just go over all targets in all cameras and count the number that have no
    correspondence number.
    
    Arguments:
    targs - list of per-camera TargetArray objects.
    
    Returns:
    the total number of unused targets.
    """
    count = 0
    for cam in targs:
        for t in cam:
            if t.tnr() == CORRES_NONE:
                count += 1
    return count
    # yes, it's that stupid.
    
def read_scene_config(fname):
    """
    Extract a YAML configuration file into regular constituents.
    
    Arguments:
    fname - path to the config YAML file.
    
    Returns:
    yaml_args - the raw dict read from file. Added keys: 'targ_par' for the
        target detection parameters (TargetParams object).
    cam_args - a list of per-camera dicts, as found in the raw YAML with keys 
        added for per-camera derived data structures. Added keys
        are 'image' for the array with image data; 'hp' for the image 
        resulting from a simple highpass filter; 'known' for the (n,3) array
        of 3D points seen by the camera; 'targs' for detected targets; 
        `glass_vec' for glass vector (transformed from list to array).
        If camera orientation files are found, also creates a Calibration 
        object under the 'calib' key.
    cpar - a ControlParameters object from the YANL 'scene' part.
    """
    yaml_args = yaml.load(open(fname,'r'),yaml.CLoader)
    cam_args = yaml_args['cameras']
    
    yaml_args['scene']['cams'] = len(cam_args)
    cpar = ControlParams(**yaml_args['scene'])
    targ_par = TargetParams(**yaml_args['detection'])
    yaml_args['targ_par'] = targ_par
    
    # Load them from spec and results of curve generator.
    for cix, cam_spec in enumerate(cam_args):
        # Detection and known points.
        cam_spec['image_data'] = imread(cam_spec['image'])
        cam_spec['hp'] = simple_highpass(cam_spec['image_data'], cpar)
        cam_spec['targs'] = target_recognition(
            cam_spec['hp'], targ_par, cix, cpar)
        
        if 'glass_vec' in cam_spec:
            cam_spec['glass_vec'] = np.r_[cam_spec['glass_vec']]
        
        if 'known_points' in cam_spec:
            cam_spec['known'] = np.loadtxt(cam_spec['known_points'])[:,1:]
        
        if 'ori_file' in cam_spec:
            cam_spec.setdefault('addpar_file', None)
            cal = Calibration()
            cal.from_file(cam_spec['ori_file'], cam_spec['addpar_file'])
            cam_spec['calib'] = cal
    
    return yaml_args, cam_args, cpar