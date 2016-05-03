# -*- coding: utf-8 -*-
"""
A panel showing an image with particles detected, with added ability to show 
different types of epipolar currves from other cameras. 

Derives from the CamPanel used for calibration.
"""

import numpy as np

from PyQt4 import QtCore, QtGui
from cam_panel import CameraPanel
from optv.calibration import Calibration
from optv.parameters import VolumeParams, ControlParams
from calib import epipolar_curve

class CamPanelEpi(CameraPanel):
    point_marked = QtCore.pyqtSignal(np.ndarray, Calibration, int,
        name="pointMarked")
    _epi_colours = ['red', 'green', 'cyan', 'magenta']

    
    def __init__(self, parent=None):
        CameraPanel.__init__(self, parent)
        self._marked_pts = []
        self._marked_patches = []
    
    def reset(self, cpar_file, vpar_file, cam_num, cal=None, 
        detection_file=None, detection_method='default'):
        """
        Set up the necessary state for analysing an image.
        
        Arguments:
        cpar_file - path to control parameters file (e.g. ptv.par) or a dict
            as read from YAML configuration.
        vpar_file - path to observed volume parameters (e.g. criteria.par)
        cam_num - identifier for this camera.
        cal - a Calibration object with camera parameters. If None, one will be 
            created.
        detection_file - optional path to target detection parameters.
        """
        if type(cpar_file) is str:
            cpar = ControlParams(4)
            cpar.read_control_par(cpar_file)
        else: # assume dict
            cpar = ControlParams(**cpar_file)
            
        CameraPanel.reset(self, cpar, cam_num, cal=cal, 
            detection_file=detection_file, detection_method=detection_method)
        self._vpar = VolumeParams()
        self._vpar.read_volume_par(vpar_file)
        
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            # Finds the closest detected point and informs others that its 
            # epipolar line must be drawn.
            click_pos = self.mapToScene(event.pos())
            cam_points = self.get_detections()
            closest_point = np.argmin(np.linalg.norm(
                cam_points - np.r_[click_pos.x(), click_pos.y()], axis=1))
            closest_coord = cam_points[closest_point]
            
            self.point_marked.emit(closest_coord, self._cal, self._num)
            self.mark_point(closest_coord)
        elif event.button() == QtCore.Qt.MiddleButton:
            CameraPanel.mousePressEvent(self, event)
    
    def set_image(self, image_name):
        """
        Assumes "reset" has been called.
        """
        CameraPanel.set_image(self, image_name)
        self.detect_targets()
        
    def mark_point(self, coord):
        """
        Record a point on the scene as marked
        """
        self._marked_pts.append(coord)
        
        rad = 5
        red_pen = QtGui.QPen(QtGui.QColor(self._epi_colours[self.cam_id()]))
        patch = self._scene.addEllipse(coord[0] - rad, coord[1] - rad, 
            2*rad, 2*rad, pen=red_pen)
        
        text_size = 20
        font = QtGui.QFont()
        font.setPointSize(text_size)
        num = self._scene.addSimpleText(str(len(self._marked_pts)), font=font)
        num.setPos(coord[0], coord[1] - (text_size + 5 + rad))
        num.setPen(red_pen)
        
        self._marked_patches.append((patch, num))
    
    def draw_epipolar_curve(self, point, origin_cam, num_points, cam_id):
        """
        Shows a line in camera color code corresponding to a point on another 
        camera's view plane.
        
        Arguments:
        point - a length-2 array with the point position on the image plane of
            the marked camera.
        origin_cam - a Calibration instance with marked-camera position and 
            distortion.
        num_points - number of points comprising the curve, minimum 2 for 
            endpoints.
        cam_id - a unique number for the origin camera that stays constant 
            across the program's run.
        
        Returns:
        a (num_points,2) array with pixel coordinates of point along the 
            epipolar curve.
        """
        pts = epipolar_curve(point, origin_cam, self._cal, num_points,
            self._cpar, self._vpar)
        
        pen = QtGui.QPen(QtGui.QColor(self._epi_colours[cam_id]))
        rad = 3
        for pt in xrange(len(pts) - 1):
            self._scene.addEllipse(pts[pt,0] - rad, pts[pt,1] - rad, 
                2*rad, 2*rad, pen=pen)
            self._scene.addLine(
                pts[pt,0], pts[pt,1], pts[pt + 1,0], pts[pt + 1,1], pen)
        return pts
        
if __name__ == "__main__":
    import sys
    
    app = QtGui.QApplication([])
    window = CamPanelEpi()
    
    cal = Calibration()
    cal.read("cal/cam1_fe.tif.ori", "cal/cam1_fe.tif.addpar")
    window.reset("parameters/ptv.par", "parameters/criteria.par", 0, cal)
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 50, 400, 500)
    
    window.show()
    window.set_image("cal/cam1_fe.tif")
    window.detect_targets()
    
    sys.exit(app.exec_())
