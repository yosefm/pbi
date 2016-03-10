# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:07:11 2015

@author: yosef
"""

from PyQt4 import QtCore, QtGui
from itertools import izip
import numpy as np, matplotlib.pyplot as pl

from calib import simple_highpass, detect_ref_points, \
    pixel_2D_coords, external_calibration, match_detection_to_ref, \
    full_calibration

from optv.calibration import Calibration
from mixintel.detection import detect_large_particles

def gray2qimage(gray):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.
    """
    gray = np.require(gray, np.uint8, 'C')
    h, w = gray.shape
    result = QtGui.QImage(gray.data, w, h, QtGui.QImage.Format_Indexed8)
    
    for i in range(256):
        result.setColor(i, QtGui.QColor(i, i, i).rgb())
    return result
 
class CameraPanel(QtGui.QGraphicsView):
    cal_changed = QtCore.pyqtSignal(Calibration, name="calibrationChanged")
    
    def __init__(self, parent=None):
        QtGui.QGraphicsView.__init__(self, parent)
    
    def reset(self, control, cam_num, manual_detection_numbers=None, cal=None,
              detection_file=None, detection_method="default"):
        """
        This function must be called before the widget is usable. It sets the
        needed configuration for future interactions.
        
        Arguments:
        control - a ControlParams object holding general scene information.
        cam_num - camera number, a unique ID used for identifying the panel 
            when thgee are several of those in a UI.
        manual_detection_numbers - numbers of points in the known-points list
            that are to be used for manual (external) calibration.
        cal - a Calibration object holding initial camera orientation. If None,
            a default calibration will be created, but don't use this unless
            you know what you're doing.
        detection_file - if not None, a .par file with overrides to the default
            detection file.
        detection_method - either "default" to use OpenPTV detection, or 
            'large' to use a template-matching algorithm.
        """
        self._manual_detection_pts = []
        self._manual_detection_nums = manual_detection_numbers
        self._next_manual = 0
        self._manual_patches = []
        
        self._cpar = control
        self._num = cam_num
        
        if cal is None:
            self._cal = Calibration()
        else:
            self._cal = cal
        
        self._detect_method = detection_method
        self._detect_path = detection_file
        self._targets = None
        self._detected_patches = []
        self._projected_patches = []
        self._residual_patches = []
    
    def clear(self):
        self._manual_detection_pts = []
        while len(self._manual_patches):
             patch, num = self._manual_patches.pop()
             self._scene.removeItem(patch)
             self._scene.removeItem(num)
    
    def set_image(self, image_name):
        """
        Replaces the scene with a new one, holding the unadorned base image.
        
        Arguments:
        image_name - path to background image.
        """
        self._scene = QtGui.QGraphicsScene(self)
        
        # Vanilla image:
        self._orig_img = pl.imread(image_name)
        pm = QtGui.QPixmap.fromImage(gray2qimage(self._orig_img))
        self._orig_pixmap = self._scene.addPixmap(pm)
        w = pm.width()
        h = pm.height()
        
        self.setScene(self._scene)
        self.scale(0.99*self.width()/w, 0.99*self.height()/h)
        self.clear()
        
        # High-pass image:
        self._hp_img = simple_highpass(self._orig_img, self._cpar)
        pm = QtGui.QPixmap.fromImage(gray2qimage(self._hp_img))
        self._hp_pixmap = self._scene.addPixmap(pm)
        self._hp_pixmap.setVisible(False)
    
    def cam_id(self):
        return self._num
    
    def calibration(self):
        return self._cal
    
    def set_highpass_visibility(self, vis):
        """
        If True, sets the background to show the high-passed image, otherwise 
        the vanilla image.
        """
        self._orig_pixmap.setVisible(not vis)
        self._hp_pixmap.setVisible(vis)
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.add_manual_detection(self.mapToScene(event.pos()))
        else:
            self.rem_last_manual_detection()
    
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
            
        self._manual_detection_pts.append((pos.x(), pos.y()))
        
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
        
        self._manual_patches.append((patch, num))
    
    def get_manual_detection_points(self):
        """
        Returns an (n,2) array for n 2D scene-coordinates points, the current
        set of manually-detected points.
        """
        return np.array(self._manual_detection_pts)
    
    def set_manual_detection_points(self, points):
        """
        Clears the current manual detection and replaces it with given points.
        
        Arguments:
        points - an (n,2) array for n 2D scene-coordinates points.
        """
        self.clear()
        for point in points:
            self.add_manual_detection(QtCore.QPointF(*point))
        
    def rem_last_manual_detection(self):
        """
        Removes last manually-detected point from the scene.
        """
        if len(self._manual_detection_pts) == 0:
            return
        
        self._manual_detection_pts.pop()
        patch, num = self._manual_patches.pop()
        self._scene.removeItem(patch)
        self._scene.removeItem(num)
        
        if self._manual_detection_nums is not None:
            self._next_manual -= 1
    
    def detect_targets(self):
        # Clear previous detections:
        while len(self._detected_patches):
            patch = self._detected_patches.pop()
            self._scene.removeItem(patch)
        
        # New detection from C:
        if self._detect_method == 'large':
            self._targets = detect_large_particles(self._orig_img)
        elif self._detect_path is None:
            self._targets = detect_ref_points(self._hp_img, self._num, self._cpar)
        else:
            self._targets = detect_ref_points(self._hp_img, self._num,
                self._cpar, self._detect_path)
        
        # Now draw it:
        blue = QtGui.QPen(QtGui.QColor("blue"))
        rad = 5
        for targ in self._targets:
            x, y = targ.pos()
            p = self._scene.addEllipse(x - rad, y - rad, 2*rad, 2*rad, pen=blue)
            self._detected_patches.append(p)
    
    def get_detections(self):
        """
        Returns the detected points from the last call to ``detect_targets()``
        as a (2,n) numpy array.
        """
        targs = np.empty((len(self._targets), 2))
        
        for tix, targ in enumerate(self._targets):
            targs[tix] = targ.pos()
        
        return targs
    
    def set_detection_visibility(self, vis):
        """
        Change visibility of detected targets.
        """
        for patch in self._detected_patches:
            patch.setVisible(vis)
        
    def project_cal_points(self, cal_points):
        """
        Calculate image-plane positions of given 3D points according to current
        calibration parameters, and show them on the panel.
        
        Arguments:
        cal_points - (n,3) array, the 3D points to project
        """
        # Clear previous projections:
        while len(self._projected_patches):
            patch = self._projected_patches.pop()
            self._scene.removeItem(patch)
        
        img_plane_pos = pixel_2D_coords(self._cal, cal_points, self._cpar)
        
        # Now draw it:
        pen = QtGui.QPen(QtGui.QColor("yellow"))
        rad = 5
        for x, y in img_plane_pos:
            p = self._scene.addEllipse(x - rad, y - rad, 2*rad, 2*rad, pen=pen)
            self._projected_patches.append(p)
    
    def set_projection_visibility(self, vis):
        """
        Change visibility of detected targets.
        """
        for patch in self._projected_patches:
            patch.setVisible(vis)
    
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
        success = external_calibration(self._cal, cal_points[manual_matching], 
            self.get_manual_detection_points(), self._cpar)
        
        if success is False:
            print "y u no good initial guess?"
        else:
            self.project_cal_points(cal_points)
            self.cal_changed.emit(self._cal)
    
    def number_detections(self, cal_points):
        if len(cal_points) > len( self._targets):
            raise ValueError("Insufficient detected points, need at least as"
                "many as fixed points")
        match_detection_to_ref(self._cal, cal_points, self._targets, self._cpar)
        
        rad = 5
        text_size = 20
        font = QtGui.QFont()
        font.setPointSize(text_size)
        pen = QtGui.QPen(QtGui.QColor("cyan"))
        self._sortgrid_patches = []
        
        for t in self._targets:
            if t.pnr() < len(cal_points):  # Number in sortgrid.c, TODO: need a constant.
                text_size = 20
                num = self._scene.addSimpleText(str(t.pnr() + 1), font=font)
                x, y = t.pos()
                num.setPos(x, y - (text_size + 5 + rad))
                num.setPen(pen)
                self._sortgrid_patches.append(num)
    
    def tune_calibration(self, cal_points):
        """
        update the calibration with results of Gauss-Markov least squares 
        iteration, i.e. the iterative process that adjust the calibration 
        parameters starting with the current initial guess.
    
        Arguments:
        cal_points - (n,3) array, the 3D calibration points to project.
        """
        if self._targets is None:
            raise ValueError("Detection must be performed first.")
        
        # Clear previous residuals:
        while len(self._residual_patches):
            patch = self._residual_patches.pop()
            self._scene.removeItem(patch)
        
        residuals, targ_ix = full_calibration(self._cal, cal_points, self._targets, 
            self._cpar)
        
        # Quiver plot of the residuals, scaled to arbitrary size.
        scale = 5000
        pen = QtGui.QPen(QtGui.QColor("red"))
        
        for r, t in zip(residuals, targ_ix):
            pos = self._targets[t].pos()
            rpos = pos + r*scale
            self._residual_patches.append(self._scene.addLine(
                pos[0], pos[1], rpos[0], rpos[1], pen=pen))
        
        self.cal_changed.emit(self._cal)
        
    def set_residuals_visibility(self, vis):
        """
        Change visibility of residuals targets.
        """
        for patch in self._residual_patches:
            patch.setVisible(vis)
    
    def save_point_sets(self, detected_file, matched_file, cal_points):
        """
        Save text files of the intermediate results of detection and of 
        matching the detections to the known positions. This may later be used
        for multiplane calibration.
        
        Currently assumes that the first targets up to the number of known 
        points are the matched points.
        
        Arguments:
        detected_file - will contain the detected points in a 3 column format -
            point number, x, y.
        matched_file - same for the known positions that were matched, with 
            columns for point number, x, y, z.
        cal_points - the known points array, as in other methods.
        """
        detected = np.empty((len(cal_points), 2))
       
        nums = np.arange(len(cal_points))
        for pnr in nums:
            detected[pnr] = self._targets[pnr].pos()
        
        detected = np.hstack((nums[:,None], detected))
        known = np.hstack((nums[:,None], cal_points))
        
        # Formats from jw_ptv.c, until we can rationalize this stuff.        
        np.savetxt(detected_file, detected, fmt="%9.5f")
        np.savetxt(matched_file, known, fmt="%10.5f")
        
if __name__ == "__main__":
    import sys
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cal_img', type=str, help="Path to calibration image")
    parser.add_argument('par_file', type=str, help="Path to main parameters")
    parser.add_argument('cam', type=int, help="Camera number")
    args = parser.parse_args()
    
    app = QtGui.QApplication([])
    window = CameraPanel(args.par_file, args.cam)
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 100, 500, 500)
    
    window.show()
    window.set_image(args.cal_img)
    window.detect_targets()
    
    sys.exit(app.exec_())