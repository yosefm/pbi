# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:07:11 2015

@author: yosef
"""

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np, matplotlib.pyplot as pl

from optv.calibration import Calibration
from optv.segmentation import target_recognition
from util.detection import detect_large_particles, detect_blobs
from util.openptv import simple_highpass

def gray2qimage(gray):
    """Convert the 2D numpy array `gray` into a 8-bit QImage with a gray
    colormap.  The first dimension represents the vertical image axis.
    """
    gray = np.require(gray, np.uint8, 'C')
    h, w = gray.shape
    result = QtGui.QImage(gray.data, w, h, QtGui.QImage.Format_Indexed8)
    
    ## Stretch the histogram over all the range.
    #crange = gray.min() - gray.max()
    colors = np.linspace(0, 255, 256)
    for i in colors:
        c = int(colors[int(i)])
        result.setColor(c, QtGui.QColor(c, c, c).rgb())
    return result

class PatchSet(object):
    """
    Manage a set of graphic items on the camera panel, with possible added 
    info.
    """
    def __init__(self, props):
        """
        Arguments:
        props - a list of names of properties attached to each point in the 
            set. A list is created for them.
        """
        self._patches = []
        self._props = {}
        
        for prop in props:
            self._props[prop] = []
    
    def push(self, patch, **props):
        """
        Registers a patch with its required properties as defined in PatchSet
        construction. Any missing or extraneous properties will cause a 
        ValueError.
        
        Arguments:
        patch - a QGraphicsItem or a sequence thereof (e.g. in case of a 
            numbered point).
        """
        self._patches.append(patch)
        
        for prop in list(self._props.keys()):
            if prop not in props:
                raise ValueError(
                    "Required property %s not given for patch." % prop)
        
        for prop, val in list(props.items()):
            if prop not in self._props:
                raise ValueError("Unrecognized property %s for patch." % prop)
            self._props[prop].append(val)
    
    def pop(self):
        """
        Remove the last inserted patch from the registry and return it so that
        the scene could remove it.
        """
        ret = self._patches.pop()
        for prop in list(self._props.values()):
            prop.pop()
        return ret
    
    def get_prop(self, name):
        return self._props[name]
    
    def __len__(self):
        return len(self._patches)
    
    def set_visibility(self, vis):
        """
        Sets a consistent visibility on all patches and subpatches.
        """
        for patch in self._patches:
            if isinstance(patch, QtWidgets.QGraphicsItem):
                patch.setVisible(vis)
            else: # assume sequence
                for subpatch in patch:
                    subpatch.setVisible(vis)
    
class CameraPanel(QtWidgets.QGraphicsView):
    cal_changed = QtCore.pyqtSignal(Calibration, name="calibrationChanged")
    
    def __init__(self, parent=None):
        QtWidgets.QGraphicsView.__init__(self, parent)
    
    def add_patchset(self, name, props=[]):
        self._patch_sets[name] = PatchSet(props)
    
    def clear_patchset(self, name):
        pset = self._patch_sets[name]
        for pnum in range(len(pset)):
            patch = pset.pop()
            if isinstance(patch, QtWidgets.QGraphicsItem):
                patch = [patch]
            for subpatch in patch:
                self._scene.removeItem(subpatch)
    
    def clear_patches(self):
        for pset in list(self._patch_sets.keys()):
            self.clear_patchset(pset)
    
    def reset(self, control, cam_num, cal=None,
        detection_method="default", **detection_pars):
        """
        This function must be called before the widget is usable. It sets the
        needed configuration for future interactions.
        
        Arguments:
        control - a ControlParams object holding general scene information.
        cam_num - camera number, a unique ID used for identifying the panel 
            when there are several of those in a UI.
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
        if detection_pars is None:
            detection_pars = {}
        detection_pars.setdefault('peak_threshold', 0.5)
        detection_pars.setdefault('radius', 20)
        
        if detection_method == "default" \
                and 'target_pars' not in detection_pars:
            raise ValueError("Selected detection method requires a " \
                "TargetParams object.")
        
        self._zoom = 1
        self._dragging = False
        self._patch_sets = {}
        
        self._cpar = control
        self._num = cam_num
        
        if cal is None:
            self._cal = Calibration()
        else:
            self._cal = cal
        
        self._detect_method = detection_method
        self._detect_par = detection_pars
        self._targets = None
        
        self.add_patchset('detected')
    
    def set_image(self, image_name, hp_vis=False):
        """
        Replaces the scene with a new one, holding the unadorned base image.
        
        Arguments:
        image_name - path to background image.
        hp_vis - whether the highpass version is visible or the original.
        """
        self._scene = QtWidgets.QGraphicsScene(self)
        
        # Vanilla image:
        self._orig_img = pl.imread(image_name)
        pm = QtGui.QPixmap.fromImage(gray2qimage(self._orig_img))
        self._orig_pixmap = self._scene.addPixmap(pm)
        w = pm.width()
        h = pm.height()
        
        self.setScene(self._scene)
        self.scale(0.99*self.width()/w, 0.99*self.height()/h)
        self.clear_patches()
        
        # High-pass image:
        #self._hp_img = preprocess_image(self._orig_img, 0, self._cpar, 12)
        self._hp_img = simple_highpass(self._orig_img, self._cpar)
        pm = QtGui.QPixmap.fromImage(gray2qimage(self._hp_img))
        self._hp_pixmap = self._scene.addPixmap(pm)
        
        self._hp_pixmap.setVisible(hp_vis)
        self._orig_pixmap.setVisible(not hp_vis)
        
    
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
        if event.button() == QtCore.Qt.MidButton: # middle button used for dragging
            self._dragging = True
            self._last_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self._dragging = False
    
    def mouseMoveEvent(self, event):
        if self._dragging:
            dx = event.pos().x() - self._last_pos.x()
            hb = self.horizontalScrollBar()
            hb.setValue(hb.value() - dx)
            
            dy = event.pos().y() - self._last_pos.y()
            vb = self.verticalScrollBar()
            vb.setValue(vb.value() - dy)

            self._last_pos = event.pos()
    
    def wheelEvent(self, event):
        """
        Implements zooming/unzooming.
        """
        numDegrees = event.delta() / 8;
        numSteps = numDegrees / 15;
        
        self.scale(1./self._zoom, 1./self._zoom)
        self._zoom += (numSteps * 0.1)
        self.scale(self._zoom, self._zoom)
        
        event.accept()
    
    def set_targets(self, targs):
        """
        Removes existing detected target set and replaces it with a new one.
        
        Arguments:
        targs - a TargetArray holding the new set.
        """
        self._targets = targs
        self.clear_patchset('detected')
        
        # Now draw new set:
        blue = QtGui.QPen(QtGui.QColor("blue"))
        rad = 5
        for targ in self._targets:
            x, y = targ.pos()
            p = self._scene.addEllipse(x - rad, y - rad, 2*rad, 2*rad, pen=blue)
            self._patch_sets['detected'].push(p)
    
    def get_target_array(self):
        """
        Return detected targets as a TargetArray object. cf. get_detections()
        for the data in a numpy array.
        """
        return self._targets
        
    def detect_targets(self):
        # New detection from C:
        if self._detect_method == 'large':
            targs = detect_large_particles(
                self._orig_img, approx_size=self._detect_par['radius'], 
                peak_thresh=self._detect_par['peak_threshold'])
        elif self._detect_method == 'dog':
            targs = detect_blobs(
                self._orig_img, thresh=self._detect_par['threshold'])
        else:
            targs = target_recognition(
                self._hp_img, self._detect_par['target_pars'], self._num, 
                self._cpar)
        
        self.set_targets(targs)
    
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
        self._patch_sets['detected'].set_visibility(vis)
    
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
    
    app = QtWidgets.QApplication([])
    window = CameraPanel(args.par_file, args.cam)
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 100, 500, 500)
    
    window.show()
    window.set_image(args.cal_img)
    window.detect_targets()
    
    sys.exit(app.exec_())
