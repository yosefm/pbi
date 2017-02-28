# -*- coding: utf-8 -*-
"""
Draw epipolar lines using calibrated cameras.

Created on Wed Sep  9 11:10:21 2015

@author: yosef
"""
import numpy as np

from PyQt4 import QtCore, QtGui
from optv.calibration import Calibration
from scene_window_base import Ui_Scene
from epi_panel import CamPanelEpi

class SceneWindow(QtGui.QWidget, Ui_Scene):
    """
    Holds 4 CamPanelEpi panels in a grid, and coordinates the drawing of 
    epipolar lines from a point selected in one camera on all other cameras.
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
                
        # Switchboard:
        self._marking_bus = QtCore.QSignalMapper(self)
    
    def init_cams(self, cpar, ov_file, image_dicts, **det_pars):
        """
        Initializes each camera panel in turn. 
        
        Arguments:
        cpar - dictionary of common scene data such as image size, as needed
            by ControlParams()
        ov_file - path to .par file holding observed volume parameters.
        det_pars - passed on directly to base.
        """
        cam_panels = self.findChildren(CamPanelEpi)
        cam_nums = range(len(cam_panels))
        cpar.setdefault('cams', len(cam_nums))
        
        large = False
        if 'detection_method' in det_pars \
                and det_pars['detection_method'] != 'default':
            large = True
        
        for cam_num, cam_dict, cam_panel in zip(cam_nums, image_dicts, cam_panels):
            cal = Calibration()
            cal.from_file(cam_dict['ori_file'], cam_dict['addpar_file'])
            
            if 'peak_threshold' in cam_dict:
                det_pars['peak_threshold'] = cam_dict['peak_threshold']
            else:
                det_pars['peak_threshold'] = 0.5
            
            cam_panel.reset(cpar, ov_file, cam_num, cal=cal, **det_pars)
            cam_panel.set_image(cam_dict['image'], hp_vis=large)
            cam_panel.set_highpass_visibility(False)
            cam_panel.point_marked.connect(self.point_marked)
    
    def point_marked(self, point, calibration, marked_cam_num):
        """
        Responds to a point marked on one camera by drawing epipolar curves on
        the other cameras.
        
        Arguments:
        point - a length-2 array with the point position on the image plane of
            the marked camera.
        calibration - a Calibration instance with camera position and 
            distortion.
        marked_cam_num - clicked camera identifier.
        """
        num_pts = 20
        cam_panels = self.findChildren(CamPanelEpi)
        for cam_num, cam_panel in enumerate(cam_panels):
            if cam_num == marked_cam_num:
                continue
            cam_panel.draw_epipolar_curve(point, calibration, 
                num_pts, marked_cam_num)
        
if __name__ == "__main__":
    import sys
    from mixintel.openptv import read_scene_config, count_unused_targets
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method', '-m', default='default', 
        choices=['default', 'large', 'dog'],
        help="Change detection method to one suitable for large particles.")
    parser.add_argument('scene_args', type=str, 
        help="Path to 4-camera scene parameters (yaml)")
    parser.add_argument('--corresp', action='store_true', default=False)
    args = parser.parse_args()
    
    yaml_args, cal_args, cpar = read_scene_config(args.scene_args)
    control_args = yaml_args['scene']
    yaml_args.setdefault('detection_params', "parameters/targ_rec.par")
    
    app = QtGui.QApplication([])
    window = SceneWindow()
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 50, 900, 900)
    window.show()

    if args.method == 'large':
        if 'sequence' in yaml_args:
            radius = yaml_args['sequence']['radius']
        else:
            radius = 20
        det_pars = {'radius': radius}
    elif args.method == 'dog':
        # This is new enough that sequence is always in yaml_args
        det_pars = {'threshold': yaml_args['sequence']['threshold']}
    elif args.method == 'default':
        det_pars = {'target_pars': yaml_args['targ_par']}
    else:
        raise ValueError('Detection method not recognized')
    
    det_pars['detection_method'] = args.method
    window.init_cams(
        control_args, yaml_args['correspondences'], cal_args, **det_pars)
    
    if args.corresp:
        from calib import correspondences
        from optv.orientation import point_positions
        from optv.transforms import convert_arr_pixel_to_metric, \
            distorted_to_flat
        #from mixintel.evolution import get_polar_rep
        
        cals = []
        targs = []
        for cam in window.findChildren(CamPanelEpi):
            cals.append(cam.calibration())
            targs.append(cam._targets)
            #print cals[-1].get_angles()
            #print get_polar_rep(cals[-1].get_pos(), cals[-1].get_angles())
        
        sets = correspondences(targs, cals, cam._vpar, cam._cpar)[0]
        print "Unused: %d" % count_unused_targets(targs)
        
        names = ['quads', 'triplets', 'pairs']
        colors = ['red', 'green', 'orange']
        for pset, clique_name, cross_color in zip(sets, names, colors):
            if pset.shape[1] == 0:
                continue
            
            flat = []
            for cam in window.findChildren(CamPanelEpi):
                cam_cent = cam.calibration().get_primary_point()[:2]
                
                unused = pset[cam.cam_id()] == -999
                metric = convert_arr_pixel_to_metric(pset[cam.cam_id()], 
                    cam._cpar)
                flat.append(distorted_to_flat(metric, cam.calibration()))
                flat[-1][unused] = -999
                
                cam.add_correspondence_set(
                    pset[cam.cam_id()], clique_name, cross_color)
            
            flat = np.array(flat)
            print point_positions(flat.transpose(1,0,2), cam._cpar, cals)[0]
            
    sys.exit(app.exec_())
