# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 14:07:11 2015

@author: yosef
"""

from PyQt4 import QtCore, QtGui
import numpy as np, matplotlib.pyplot as pl

from calib import simple_highpass, detect_ref_points, \
    external_calibration, match_detection_to_ref, full_calibration

from optv.calibration import Calibration
from optv.imgcoord import image_coordinates
from optv.transforms import convert_arr_metric_to_pixel
from mixintel.detection import detect_large_particles

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
        c = int(colors[i])
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
        
        for prop in self._props.keys():
            if prop not in props:
                raise ValueError(
                    "Required property %s not given for patch." % prop)
        
        for prop, val in props.iteritems():
            if prop not in self._props:
                raise ValueError("Unrecognized property %s for patch." % prop)
            self._props[prop].append(val)
    
    def pop(self):
        """
        Remove the last inserted patch from the registry and return it so that
        the scene could remove it.
        """
        ret = self._patches.pop()
        for prop in self._props.itervalues():
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
            if isinstance(patch, QtGui.QGraphicsItem):
                patch.setVisible(vis)
            else: # assume sequence
                for subpatch in patch:
                    subpatch.setVisible(vis)
    
class CameraPanel(QtGui.QGraphicsView):
    cal_changed = QtCore.pyqtSignal(Calibration, name="calibrationChanged")
    
    def __init__(self, parent=None):
        QtGui.QGraphicsView.__init__(self, parent)
    
    def add_patchset(self, name, props=[]):
        self._patch_sets[name] = PatchSet(props)
    
    def clear_patchset(self, name):
        pset = self._patch_sets[name]
        for pnum in xrange(len(pset)):
            patch = pset.pop()
            if isinstance(patch, QtGui.QGraphicsItem):
                patch = [patch]
            for subpatch in patch:
                self._scene.removeItem(subpatch)
    
    def clear_patches(self):
        for pset in self._patch_sets.iterkeys():
            self.clear_patchset(pset)
    
    def reset(self, control, cam_num, manual_detection_numbers=None, cal=None,
              detection_file=None, detection_method="default",
              peak_threshold=0.5):
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
        peak_threshold - for the 'large' method, the minimum grey value for a
            peak to be recognized.
        """
        self._zoom = 1
        self._dragging = False
        self._patch_sets = {}
        
        self.add_patchset('manual', ['pos'])
        self._manual_detection_nums = manual_detection_numbers
        self._next_manual = 0
        
        self._cpar = control
        self._num = cam_num
        
        if cal is None:
            self._cal = Calibration()
        else:
            self._cal = cal
        
        self._detect_method = detection_method
        self._detect_path = detection_file
        self._detect_thresh = peak_threshold
        self._targets = None
        
        self.add_patchset('detected')
        self.add_patchset('projected')
        self.add_patchset('matching')
        self._residual_patches = []
    
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
        self.clear_patches()
        
        # High-pass image:
        self._hp_img = simple_highpass(self._orig_img, self._cpar)
        pm = QtGui.QPixmap.fromImage(gray2qimage(self._hp_img))
        self._hp_pixmap = self._scene.addPixmap(pm)
        
        hp_vis = False
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
        if event.button() == QtCore.Qt.LeftButton:
            self.add_manual_detection(self.mapToScene(event.pos()))
        elif event.button() == QtCore.Qt.RightButton:
            self.rem_last_manual_detection()
        else: # middle button used for dragging
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
        
    def detect_targets(self):
        # New detection from C:
        if self._detect_method == 'large':
            targs = detect_large_particles(self._orig_img, 
                peak_thresh=self._detect_thresh)
        elif self._detect_path is None:
            targs = detect_ref_points(self._hp_img, self._num, self._cpar)
        else:
            targs = detect_ref_points(self._hp_img, self._num,
                self._cpar, self._detect_path)
        
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
        
        sorted_targs = match_detection_to_ref(self._cal, cal_points, 
            self._targets, self._cpar)
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
        
        residuals, targ_ix, err_est = full_calibration(self._cal, cal_points, 
            self._targets, self._cpar)
        self.report_orientation(err_est)
        
        # Quiver plot of the residuals, scaled to arbitrary size.
        scale = 5000
        pen = QtGui.QPen(QtGui.QColor("red"))
        
        for r, t in zip(residuals, targ_ix):
            pos = self._targets[t].pos()
            rpos = pos + r*scale
            self._residual_patches.append(self._scene.addLine(
                pos[0], pos[1], rpos[0], rpos[1], pen=pen))
        
        self.cal_changed.emit(self._cal)
    
    def report_orientation(self, err_est):
        """
        Terminal output of the current calibration, +- errors as estimated by
        the orientation algorithm.
        
        Arguments:
        err_est - an array with the respective error estimates for each 
            calibration parameter.
        """
        from scipy.constants import degree
        
        print "\n|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
        print "\nResults after iteration:\n"
        print "sigma0 = %6.2f micron" % (err_est[-1]*1000)
        
        x0, y0, z0 = self._cal.get_pos()
        print "X0 =    %8.3f   +/- %8.3f" % (x0, err_est[0])
        print "Y0 =    %8.3f   +/- %8.3f" % (y0, err_est[1])
        print "Z0 =    %8.3f   +/- %8.3f" % (z0, err_est[2])
        
        omega, phi, kappa = self._cal.get_angles()
        print "omega = %8.4f   +/- %8.4f degrees" % \
            (omega/degree, err_est[3]/degree)
        print "phi   = %8.4f   +/- %8.4f degrees" % \
            (phi/degree, err_est[4]/degree)
        print "kappa = %8.4f   +/- %8.4f degrees" % \
            (kappa/degree, err_est[5]/degree)
        
        cc, xh, yh = self._cal.get_primary_point()
        print "camera const  = %8.5f   +/- %8.5f" % (cc, err_est[6])
        print "xh            = %8.5f   +/- %8.5f" % (xh, err_est[7])
        print "yh            = %8.5f   +/- %8.5f" % (xh, err_est[8])
        
        k1, k2, k3 = self._cal.get_radial_distortion()
        print "k1            = %8.5f   +/- %8.5f" % (k1, err_est[9])
        print "k2            = %8.5f   +/- %8.5f" % (k2, err_est[10])
        print "k3            = %8.5f   +/- %8.5f" % (k3, err_est[11])
        
        p1, p2 = self._cal.get_decentering()
        print "p1            = %8.5f   +/- %8.5f" % (p1, err_est[12])
        print "p2            = %8.5f   +/- %8.5f" % (p2, err_est[13])
        
        scx, she = self._cal.get_affine()
        print "scale for x'  = %8.5f   +/- %8.5f" % (scx, err_est[14])
        print "shearing      = %8.5f   +/- %8.5f degrees" % \
            (she/degree, err_est[15]/degree)
        
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