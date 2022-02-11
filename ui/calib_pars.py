# -*- coding: utf-8 -*-
"""
Implementation of Calibration object interface. Holds a single Calibration
instance which is rewritten with changes made by the GUI or otherwise.

Created on Tue Aug  4 10:23:01 2015

@author: yosef
"""

from PyQt4 import QtCore, QtGui
from .calib_pars_base import Ui_calibPars

import numpy as np
from optv.calibration import Calibration

class CalibParameters(QtGui.QWidget, Ui_calibPars):
    pos_changed = QtCore.pyqtSignal(np.ndarray, name="positionChanged")
    ang_changed = QtCore.pyqtSignal(np.ndarray, name="orientationChanged")
    primary_point_changed = QtCore.pyqtSignal(np.ndarray, name="primaryPointChanged")
    distortion_changed = QtCore.pyqtSignal(np.ndarray, name="distortionChanged")
    cal_changed = QtCore.pyqtSignal(Calibration, name="calibrationChanged")
    
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
    
    def set_calibration_obj(self, cal_obj):
        """
        Call this once and only once, after construction.
        """
        self._cal = cal_obj
        self.update_all_fields()
        
        # Routing table:
        self.posx.valueChanged.connect(self.pos_spinbox_changed)
        self.posy.valueChanged.connect(self.pos_spinbox_changed)
        self.posz.valueChanged.connect(self.pos_spinbox_changed)
        
        self.intpar_cc.valueChanged.connect(self.int_spinbox_changed)
        self.intpar_xh.valueChanged.connect(self.int_spinbox_changed)
        self.intpar_yh.valueChanged.connect(self.int_spinbox_changed)
        
        self.ang_omega.valueChanged.connect(self.ang_spinbox_changed)
        self.ang_phi.valueChanged.connect(self.ang_spinbox_changed)
        self.ang_kappa.valueChanged.connect(self.ang_spinbox_changed)
        
        self.radial_k1.valueChanged.connect(self.distortion_sb_changed)
        self.radial_k2.valueChanged.connect(self.distortion_sb_changed)
        self.radial_k3.valueChanged.connect(self.distortion_sb_changed)
        self.decent_p1.valueChanged.connect(self.distortion_sb_changed)
        self.decent_p2.valueChanged.connect(self.distortion_sb_changed)
        
    def calibration(self):
        return self._cal
            
    def update_all_fields(self):
        x, y, z = self._cal.get_pos()
        self.posx.setValue(x)
        self.posy.setValue(y)
        self.posz.setValue(z)
        
        omega, phi, kappa = self._cal.get_angles()
        self.ang_omega.setValue(omega)
        self.ang_phi.setValue(phi)
        self.ang_kappa.setValue(kappa)
        
        xh, yh, cc = self._cal.get_primary_point()
        self.intpar_cc.setValue(cc)
        self.intpar_xh.setValue(xh)
        self.intpar_yh.setValue(yh)
        
        k1, k2, k3 = self._cal.get_radial_distortion()
        self.radial_k1.setValue(k1)
        self.radial_k2.setValue(k2)
        self.radial_k3.setValue(k3)
        
        p1, p2 = self._cal.get_decentering()
        self.decent_p1.setValue(p1)
        self.decent_p2.setValue(p2)
    
    def set_free_vars(self, set_names):
        controls = [self.use_cc, self.use_xh, self.use_yh, self.use_k1,
                    self.use_k2, self.use_k3, self.use_p1, self.use_p2]
        names = ['cc', 'xh', 'yh', 'k1', 'k2', 'k3', 'p1', 'p2', 
            'scale', 'shear']
        
        for name, checkbox in zip(names, controls):
            if name in set_names:
                checkbox.setChecked(True)
        
    def get_free_vars(self):
        controls = [self.use_cc, self.use_xh, self.use_yh, self.use_k1,
                    self.use_k2, self.use_k3, self.use_p1, self.use_p2]
        names = ['cc', 'xh', 'yh', 'k1', 'k2', 'k3', 'p1', 'p2', 
            'scale', 'shear']
        return [name for name, control in zip(names, controls) \
            if control.isChecked()]
    
    def pos_spinbox_changed(self, newval):
        newpos = np.r_[self.posx.value(), self.posy.value(), self.posz.value()]
        self._cal.set_pos(newpos)
        self.pos_changed.emit(newpos)
        self.cal_changed.emit(self._cal)
    
    def ang_spinbox_changed(self, newval):
        newangs = np.r_[
            self.ang_omega.value(),
            self.ang_phi.value(),
            self.ang_kappa.value(),
        ]
        self._cal.set_angles(newangs)
        self.ang_changed.emit(newangs)
        self.cal_changed.emit(self._cal)

    def int_spinbox_changed(self, newval):
        newpp = np.r_[
            self.intpar_xh.value(),
            self.intpar_yh.value(),
            self.intpar_cc.value(),
        ]
        self._cal.set_primary_point(newpp)
        self.primary_point_changed.emit(newpp)
        self.cal_changed.emit(self._cal)
    
    def distortion_sb_changed(self, newval):
        newrad = np.r_[
            self.radial_k1.value(),
            self.radial_k2.value(),
            self.radial_k3.value(),
        ]
        newdec = np.r_[
            self.decent_p1.value(),
            self.decent_p2.value()
        ]
        
        self._cal.set_radial_distortion(newrad)
        self._cal.set_decentering(newdec)
        self.distortion_changed.emit(np.r_[newrad, newdec])
        self.cal_changed.emit(self._cal)
        
if __name__ == "__main__":
    import sys
    
    cal = Calibration()
    cal.read_calibration(sys.argv[1], sys.argv[2])
    
    def print_pos():
        print((cal.get_pos()))
    
    def print_angs():
        print((cal.get_angles()))
    
    app = QtGui.QApplication([])
    window = CalibParameters()
    window.set_calibration_obj(cal)
    window.pos_changed.connect(lambda x: print_pos())
    window.ang_changed.connect(lambda x: print_angs())
    
    window.show()
    
    sys.exit(app.exec_())