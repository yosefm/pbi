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
    
    def init_cams(self, cpar, ov_file, detect_file, image_dicts, large=False):
        """
        Initializes each camera panel in turn. 
        
        Arguments:
        cpar - dictionary of common scene data such as image size, as needed
            by mixintel.openptv.control_params()
        ov_file - path to .par file holding observed volume parameters.
        image_dicts - a list of dicts, one per camera. The dict contains the 
            following keys: image (path to image file); ori_file (path to 
            corresponding calibration information .ori file); addpar_file 
            (path to camera distortion parameters file).
        large - change detection method to template matching.
        """
        cam_panels = self.findChildren(CamPanelEpi)
        cam_nums = range(len(cam_panels))
        method = 'large' if large else 'default'
        cpar.setdefault('cams', len(cam_nums))
        
        for cam_num, cam_dict, cam_panel in zip(cam_nums, image_dicts, cam_panels):
            cal = Calibration()
            cal.from_file(cam_dict['ori_file'], cam_dict['addpar_file'])
            cam_panel.reset(cpar, ov_file, cam_num, cal=cal, 
                detection_file=detect_file, detection_method=method)
            cam_panel.set_image(cam_dict['image'])
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
            pts_epi = cam_panel.draw_epipolar_curve(point, calibration, 
                num_pts, marked_cam_num)
            #pts_linear = np.c_[np.linspace(pts_epi[0,0], pts_epi[-1,0], num_pts),
            #                   np.linspace(pts_epi[0,1], pts_epi[-1,1], num_pts)
            #]
            print pts_epi
            #print pts_linear
        
if __name__ == "__main__":
    import sys, yaml
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ov_file', type=str,
        help="Path to observed volume parameters")
    parser.add_argument('-d', '--detection-file', type=str, 
        default="parameters/targ_rec.par", help="Path to detection parameters")
    parser.add_argument('--large', '-l', action='store_true', default=False,
        help="Change detection method to one suitable for large particles.")
    parser.add_argument('scene_args', type=str, 
        help="Path to 4-camera scene parameters (yaml)")
    parser.add_argument('--corresp', action='store_true', default=False)
    args = parser.parse_args()
    
    yaml_args = yaml.load(file(args.scene_args))
    cal_args = yaml_args['images']
    control_args = yaml_args['scene']
    
    app = QtGui.QApplication([])
    window = SceneWindow()
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 50, 900, 900)
    
    window.show()
    window.init_cams(control_args, args.ov_file, args.detection_file, cal_args, 
        large=args.large)
    
    if args.corresp:
        from calib import correspondences, point_positions
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
