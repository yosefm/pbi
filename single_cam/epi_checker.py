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
    
    def init_cams(self, par_file, ov_file, detect_file, image_dicts):
        """
        Initializes each camera panel in turn. 
        
        Arguments:
        par_file - path to .par file holding common scene data such as image 
            size.
        ov_file - path to .par file holding observed volume parameters.
        image_dicts - a list of dicts, one per camera. The dict contains the 
            following keys: image (path to image file); ori_file (path to 
            corresponding calibration information .ori file); addpar_file 
            (path to camera distortion parameters file).
        """
        cam_panels = self.findChildren(CamPanelEpi)
        cam_nums = range(len(cam_panels))
        
        for cam_num, cam_dict, cam_panel in zip(cam_nums, image_dicts, cam_panels):
            cal = Calibration()
            cal.from_file(cam_dict['ori_file'], cam_dict['addpar_file'])
            cam_panel.reset(par_file, ov_file, cam_num, cal=cal, 
                detection_file=detect_file)
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
    parser.add_argument('par_file', type=str, help="Path to main parameters")
    parser.add_argument('ov_file', type=str,
        help="Path to observed volume parameters")
    parser.add_argument('-d', '--detection-file', type=str, 
        default="parameters/targ_rec.par", help="Path to detection parameters")
    parser.add_argument('scene_args', type=str, 
        help="Path to 4-camera scene parameters (yaml)")
    args = parser.parse_args()
    
    cal_args = yaml.load(file(args.scene_args))
    
    app = QtGui.QApplication([])
    window = SceneWindow()
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 50, 900, 900)
    
    window.show()
    window.init_cams(args.par_file, args.ov_file, args.detection_file, cal_args)
    
    sys.exit(app.exec_())