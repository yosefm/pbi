# -*- coding: utf-8 -*-
"""
Code for interacting with liboptv that is pure Python and requires no 
compilation. Counterpart of calib.pyx in that it could be in liboptv but for
the bureocratic burden.

Created on Sun Jan 31 14:06:13 2016

@author: yosef
"""
from optv.parameters import ControlParams

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