# -*- coding: utf-8 -*-
"""
Code for interacting with liboptv that is pure Python and requires no 
compilation. Counterpart of calib.pyx in that it could be in liboptv but for
the bureocratic burden.

Created on Sun Jan 31 14:06:13 2016

@author: yosef
"""
import yaml
import numpy as np
from scipy.misc import imread

from optv.parameters import ControlParams
from optv.calibration import Calibration
from optv.image_processing import preprocess_image
from calib import detect_ref_points

def simple_highpass(img, cpar):
    preprocess_image(img, 0, cpar, 12)

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
    
    control.set_hp_flag('hp' in control_args['flags'])
    control.set_allCam_flag('allcam' in control_args['flags'])
    control.set_tiff_flag('headers' in control_args['flags'])
    control.set_image_size(control_args['image_size'])
    control.set_pixel_size(control_args['pixel_size'])
    control.set_chfield(0)
    
    layers = control.get_multimedia_params()
    layers.set_n1(control_args['cam_side_n'])
    layers.set_layers(control_args['wall_ns'], control_args['wall_thicks'])
    layers.set_n3(control_args['object_side_n'])
    
    return control
    
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

def read_scene_config(fname):
    """
    Extract a YAML configuration file into regular constituents.
    
    Arguments:
    fname - path to the config YAML file.
    
    Returns:
    yaml_args - the raw dict read from file.
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
    yaml_args = yaml.load(file(fname))
    cam_args = yaml_args['cameras']
    
    yaml_args['scene']['cams'] = len(cam_args)
    cpar = ControlParams(**yaml_args['scene'])
        
    # Load them from spec and results of curve generator.
    for cix, cam_spec in enumerate(cam_args):
        # Detection and known points.
        cam_spec['image_data'] = imread(cam_spec['image'])
        cam_spec['hp'] = simple_highpass(cam_spec['image_data'], cpar)
        cam_spec['targs'] = detect_ref_points(cam_spec['hp'], cix, cpar, 
            detection_pars=yaml_args['detection_params'])
        print() # misbehaved liboptv
        
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