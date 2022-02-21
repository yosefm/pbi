#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Image panel for showing calibration data.

Created on Wed Mar 29 13:03:43 2017

@author: yosef
"""

from PyQt5 import QtCore, QtGui
import numpy as np

from cam_panel import CameraPanel

from optv.orientation import external_calibration, full_calibration
from optv.imgcoord import image_coordinates
from optv.transforms import convert_arr_metric_to_pixel
from optv.orientation import match_detection_to_ref

class CalibPanel(CameraPanel):
    def __init__(self, parent=None):
        CameraPanel.__init__(self, parent)
    
    def reset(self, control, cam_num, manual_detection_numbers=None, cal=None,
        detection_method="default", **detection_pars):
        """
        This function must be called before the widget is usable. It sets the
        needed configuration for future interactions. Mainly calls the base 
        class with amall added setup.
        
        Arguments:
        control - a ControlParams object holding general scene information.
        cam_num - camera number, a unique ID used for identifying the panel 
            when there are several of those in a UI.
        manual_detection_numbers - numbers of points in the known-points list
            that are to be used for manual (external) calibration.
        cal - a Calibration object holding initial camera orientation. If None,
            a default calibration will be created, but don't use this unless
            you know what you're doing.
        detection_method - either 'default' to use OpenPTV detection, or 
            'large' to use a template-matching algorithm.
        detection_pars - parameters specific to each detection method, as 
            specified below.
        
        Parameters for 'default' detection:
        target_pars - a TargetParams object.
        
        Parameters for 'large' method:
        peak_threshold - minimum grey value for a peak to be recognized.
        radius - the expected radius of particles, in pixels.
        
        Parameters for 'dog' method:
        threshold - minimum grey value for blob pixels.
        """
        CameraPanel.reset(
            self, control, cam_num, cal, detection_method, **detection_pars)
        
        self.add_patchset('manual', ['pos'])
        self._manual_detection_nums = manual_detection_numbers
        self._next_manual = 0
        
        self.add_patchset('resids')
        self.add_patchset('projected')
        self.add_patchset('matching')
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.add_manual_detection(self.mapToScene(event.pos()))
        elif event.button() == QtCore.Qt.RightButton:
            self.rem_last_manual_detection()
        else: 
            CameraPanel.mousePressEvent(self, event)
    
    def add_manual_detection(self, pos):
        """
        Adds a position clicked by the user to the list of manual detection 
        points.
        
        Arguments:
        pos - a QPointF with the scene coordinates of the clicked point.
        """
        if self._manual_detection_nums is not None:
            if self._next_manual >= len(self._manual_detection_nums):
                return
            num = self._manual_detection_nums[self._next_manual]
            self._next_manual += 1
        
        rad = 5
        red_pen = QtGui.QPen(QtGui.QColor("red"))
        patch = self._scene.addEllipse(pos.x() - rad, pos.y() - rad, 
            2*rad, 2*rad, pen=red_pen)
        
        text_size = 20
        font = QtGui.QFont()
        font.setPointSize(text_size)
        num = self._scene.addSimpleText(str(num), font=font)
        num.setPos(pos.x(), pos.y() - (text_size + 5 + rad))
        num.setPen(red_pen)
        
        self._patch_sets['manual'].push((patch, num), pos=(pos.x(), pos.y()))
        
    
    def get_manual_detection_points(self):
        """
        Returns an (n,2) array for n 2D scene-coordinates points, the current
        set of manually-detected points.
        """
        return np.array(self._patch_sets['manual'].get_prop('pos'))
    
    def set_manual_detection_points(self, points):
        """
        Clears the current manual detection and replaces it with given points.
        
        Arguments:
        points - an (n,2) array for n 2D scene-coordinates points.
        """
        self.clear_patchset('manual')
        for point in points:
            self.add_manual_detection(QtCore.QPointF(*point))
        
    def rem_last_manual_detection(self):
        """
        Removes last manually-detected point from the scene.
        """
        if len(self._patch_sets['manual']) == 0:
            return
        
        patch, num = self._patch_sets['manual'].pop()
        self._scene.removeItem(patch)
        self._scene.removeItem(num)
        
        if self._manual_detection_nums is not None:
            self._next_manual -= 1
    
    def project_cal_points(self, cal_points):
        """
        Calculate image-plane positions of given 3D points according to current
        calibration parameters, and show them on the panel.
        
        Arguments:
        cal_points - (n,3) array, the 3D points to project
        """
        self.clear_patchset('projected')
        
        img_coords = image_coordinates(cal_points, self._cal, 
            self._cpar.get_multimedia_params())
        img_plane_pos = convert_arr_metric_to_pixel(img_coords, self._cpar)
        
        # Now draw it:
        pen = QtGui.QPen(QtGui.QColor("yellow"))
        rad = 5
        for x, y in img_plane_pos:
            p = self._scene.addEllipse(x - rad, y - rad, 2*rad, 2*rad, pen=pen)
            self._patch_sets['projected'].push(p)
    
    def set_projection_visibility(self, vis):
        """
        Change visibility of detected targets.
        """
        self._patch_sets['projected'].set_visibility(vis)
    
    def number_detections(self, cal_points):
        if len(cal_points) > len( self._targets):
            raise ValueError("Insufficient detected points, need at least as"
                "many as fixed points")
        
        sorted_targs = match_detection_to_ref(self._cal, cal_points, 
            self._targets, self._cpar, eps=51)
        self.set_targets(sorted_targs)
        
        self.clear_patchset('projected')
        rad = 5
        text_size = 20
        font = QtGui.QFont()
        font.setPointSize(text_size)
        pen = QtGui.QPen(QtGui.QColor("cyan"))
        
        for t in self._targets:
            if t.pnr() < len(cal_points):
                text_size = 20
                num = self._scene.addSimpleText(str(t.pnr() + 1), font=font)
                x, y = t.pos()
                num.setPos(x, y - (text_size + 5 + rad))
                num.setPen(pen)
                self._patch_sets['matching'].push(num)
    
    def tune_external_calibration(self, cal_points, manual_matching):
        """
        update the external calibration with results of raw orientation, i.e.
        the iterative process that adjust the initial guess' external 
        parameters (position and angle of cameras) without internal or
        distortions.
    
        Arguments:
        cal_points - (n,3) array, the 3D calibration points to project.
        manual_matching - n-length boolean array, True where the corresponding
            row of cal_points represents the position of a manual detection 
            point. Should be in the same order that manual points are entered,
            i.e. the order of manual detection numbers given at construction.
        """
        success = external_calibration(
            self.calibration(), cal_points[manual_matching], 
            self.get_manual_detection_points(), self._cpar)
        
        if success is False:
            print("y u no good initial guess?")
        else:
            self.project_cal_points(cal_points)
            self.cal_changed.emit(self.calibration())
    
    def tune_calibration(self, cal_points, flags):
        """
        update the calibration with results of Gauss-Markov least squares 
        iteration, i.e. the iterative process that adjust the calibration 
        parameters starting with the current initial guess.
    
        Arguments:
        cal_points - (n,3) array, the 3D calibration points to project.
        """
        targs = self.get_target_array()
        if targs is None:
            raise ValueError("Detection must be performed first.")
        
        # Clear previous residuals:
        self.clear_patchset('resids')
        
        residuals, targ_ix, err_est = full_calibration(
            self._cal, cal_points, targs, self._cpar, flags)
        self.report_orientation(err_est)
        
        # Quiver plot of the residuals, scaled to arbitrary size.
        scale = 5000
        pen = QtGui.QPen(QtGui.QColor("red"))
        
        print(residuals)
        print(targ_ix)

        for r, t in zip(residuals, targ_ix):
            if t > -999: 
                pos = targs[t].pos()
                rpos = pos + r*scale
                self._patch_sets['resids'].push(self._scene.addLine(
                    pos[0], pos[1], rpos[0], rpos[1], pen=pen))
        
        self.cal_changed.emit(self.calibration())
    
    def report_orientation(self, err_est):
        """
        Terminal output of the current calibration, +- errors as estimated by
        the orientation algorithm.
        
        Arguments:
        err_est - an array with the respective error estimates for each 
            calibration parameter.
        """
        from scipy.constants import degree
        
        print("\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print("\nResults after iteration:\n")
        print(("sigma0 = %6.2f micron" % (err_est[-1]*1000)))
        
        cal = self.calibration()
        x0, y0, z0 = cal.get_pos()
        print(("X0 =    %8.3f   +/- %8.3f" % (x0, err_est[0])))
        print(("Y0 =    %8.3f   +/- %8.3f" % (y0, err_est[1])))
        print(("Z0 =    %8.3f   +/- %8.3f" % (z0, err_est[2])))
        
        omega, phi, kappa = cal.get_angles()
        print(("omega = %8.4f   +/- %8.4f degrees" % \
            (omega/degree, err_est[3]/degree)))
        print(("phi   = %8.4f   +/- %8.4f degrees" % \
            (phi/degree, err_est[4]/degree)))
        print(("kappa = %8.4f   +/- %8.4f degrees" % \
            (kappa/degree, err_est[5]/degree)))
        
        cc, xh, yh = cal.get_primary_point()
        print(("camera const  = %8.5f   +/- %8.5f" % (cc, err_est[6])))
        print(("xh            = %8.5f   +/- %8.5f" % (xh, err_est[7])))
        print(("yh            = %8.5f   +/- %8.5f" % (xh, err_est[8])))
        
        k1, k2, k3 = cal.get_radial_distortion()
        print(("k1            = %8.5f   +/- %8.5f" % (k1, err_est[9])))
        print(("k2            = %8.5f   +/- %8.5f" % (k2, err_est[10])))
        print(("k3            = %8.5f   +/- %8.5f" % (k3, err_est[11])))
        
        p1, p2 = cal.get_decentering()
        print(("p1            = %8.5f   +/- %8.5f" % (p1, err_est[12])))
        print(("p2            = %8.5f   +/- %8.5f" % (p2, err_est[13])))
        
        scx, she = cal.get_affine()
        print(("scale for x'  = %8.5f   +/- %8.5f" % (scx, err_est[14])))
        print(("shearing      = %8.5f   +/- %8.5f degrees" % \
            (she/degree, err_est[15]/degree)))
        
    def set_residuals_visibility(self, vis):
        """
        Change visibility of residuals targets.
        """
        self._patch_sets['resids'].set_visibility(vis)
