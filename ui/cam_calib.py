# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:46:14 2015

@author: yosef
"""

import numpy as np

from PyQt5 import QtGui, QtWidgets  
from cam_calib_base import Ui_CameraCalibration

from optv.calibration import Calibration
from optv.parameters import ControlParams, TargetParams

class SingleCameraCalibration(QtWidgets.QWidget, Ui_CameraCalibration):
    def __init__(self, control_args, cam, ori, addpar, man_file, known_points,
                 manual_detection_numbers, detection_args, tune_flags=[]):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)
        
        self._man = man_file

        # Generate OpenPTV configuration objects:
        cal = Calibration()
        cal.from_file(ori.encode(), addpar.encode())
        
        control_args.setdefault('cams', 1)
        control = ControlParams(**control_args)
        targ_par = TargetParams(**detection_args)
        
        # Subordinate widgets setup:
        self.cam.reset(control, cam, manual_detection_numbers, cal, 
            target_pars=targ_par)
        self.calpars.set_free_vars(tune_flags)
        self.calpars.set_calibration_obj(cal)
        
        self.txt_ori.setText(ori)
        self.txt_addpar.setText(addpar)
        
        img_name = ori[:ori.rfind('.')]
        self.txt_detected.setText(img_name + '.crd')
        self.txt_matched.setText(img_name + '.fix')
        
        # Reference points preprocessing:
        self._cp = np.loadtxt(known_points, usecols=(1,2,3))
        self._match_manual = np.zeros(len(self._cp), dtype=np.bool)
        self._match_manual[np.r_[manual_detection_numbers] - 1] = True
        
        # Signal/slot routing table:
        self.show_hp.stateChanged.connect(self.cam.set_highpass_visibility)
        self.show_detect.stateChanged.connect(self.cam.set_detection_visibility)
        self.show_project.stateChanged.connect(self.cam.set_projection_visibility)
        self.show_project.stateChanged.connect(self.reproject_points)
        self.show_resids.stateChanged.connect(self.cam.set_residuals_visibility)
        
        self.btn_load_man.clicked.connect(self.load_man_detection)
        self.btn_save_man.clicked.connect(self.save_man_detection)
        
        self.btn_detect.clicked.connect(self.cam.detect_targets)
        self.btn_detect.clicked.connect(lambda: self.show_detect.setChecked(True))
        
        self.btn_raw.clicked.connect(
            lambda: self.cam.tune_external_calibration(self._cp, self._match_manual))
        self.btn_raw.clicked.connect(lambda: self.show_project.setChecked(True))
        
        self.btn_number.clicked.connect(lambda: self.cam.number_detections(self._cp))
        
        self.btn_full_calib.clicked.connect(self.call_full_calibration_fith_flags)
        self.btn_full_calib.clicked.connect(lambda: self.show_resids.setChecked(True))
        
        self.btn_save_cal.clicked.connect(self.save_calibration)
        self.btn_dump_multi.clicked.connect(self.save_point_sets)
        self.calpars.cal_changed.connect(self.reproject_points)
        
        self.cam.cal_changed.connect(self.reproject_points)
        self.cam.cal_changed.connect(self.calpars.update_all_fields)
        
    def set_image(self, image_name):
        """
        Replaces the scene with a new one, holding the unadorned base image.
        
        Arguments:
        image_name - path to background image.
        """
        self.cam.set_image(image_name)
    
    def load_man_detection(self):
        """
        Reads the manual detection file whose name was given at construction
        time, and replaces clicked points with its content. The file is a 
        simple two-column text table of 2D scene-coordinates points.
        """
        pts = np.loadtxt(self._man)
        self.cam.set_manual_detection_points(pts)
    
    def save_man_detection(self):
        """
        Saves manual detection points for this camera as a simple text table,
        in the file name given at construction time.
        """
        pts = self.cam.get_manual_detection_points()
        np.savetxt(self._man, pts)
    
    def save_calibration(self):
        self.calpars.calibration().write(
            str(self.txt_ori.text()), str(self.txt_addpar.text()))
    
    def reproject_points(self, cal=None):
        if self.show_project.isChecked():
            self.cam.project_cal_points(self._cp)
    
    def save_point_sets(self):
        self.cam.save_point_sets(str(self.txt_detected.text()),
            str(self.txt_matched.text()), self._cp)
    
    def call_full_calibration_fith_flags(self):
        self.cam.tune_calibration(self._cp, self.calpars.get_free_vars())
        
if __name__ == "__main__":
    import sys
    import yaml
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('calib_par_file', type=str, 
        help="Path to calibration parameters of this camera (yaml)")
    args = parser.parse_args()
    
    yaml_args = yaml.load(open(args.calib_par_file,'r'), yaml.CLoader)
    cal_args = yaml_args['target']
    scene_args = yaml_args['scene']
    
    app = QtWidgets.QApplication([])
    
    cal_args.setdefault('detection_par_file', None)
    conf_args = (scene_args, cal_args['number'], 
        cal_args['ori_file'], cal_args['addpar_file'], 
        cal_args['manual_detection_file'], cal_args['known_points'],
        cal_args['manual_detection_points'], yaml_args['detection'],
        yaml_args['default_free_vars'])
    if 'detection_method' in cal_args:
        conf_args = conf_args + (cal_args['detection_method'], )
        
    window = SingleCameraCalibration(*conf_args)
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 50, 1200, 900)
    
    window.show()
    window.set_image(cal_args['image'])
    
    sys.exit(app.exec_())
