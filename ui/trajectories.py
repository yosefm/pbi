# -*- coding: utf-8 -*-
"""
Show trajectories in a 4-camera view. Allow highlighting trajectories based
on user selection.

Trajectories will be shown by projecting their 3D points, for now. this 
will save us the headache of loading targets for a full scene. Fingers Crossed.

Created on Sun Oct  2 12:36:21 2016

@author: yosef
"""

#import numpy as np

from PyQt5 import QtCore, QtGui
from .trajectories_base import Ui_TrajectoriesSelector
from .cam_panel import CameraPanel

from optv.imgcoord import image_coordinates
from optv.transforms import convert_arr_metric_to_pixel
from flowtracks.scene import Scene
from flowtracks.io import save_particles_table

class TrajectoriesWindow(QtGui.QWidget, Ui_TrajectoriesSelector):
    """
    Shows trajectories in a 4-camera grid, and coordinates the selection and
    highlighting activities.
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
        
        self._traj_cache = {}
        self.btn_export.clicked.connect(self.export_selected)
    
    def set_trajectories(self, filename):
        """
        Fill out the selection table and initialize trajectory display.
        """
        self._scn = Scene(filename)
        self.traj_table.setColumnCount(3)
        self.traj_table.setColumnWidth(0, 30)
        self.traj_table.setHorizontalHeaderLabels(["Show", "ID", "length"])
        
        max_length = 0
        for trix, traj in enumerate(self._scn.iter_trajectories()):
            self.traj_table.setRowCount(trix + 1)
            
            # Checkbox deciding visibility of trajectory.
            item = QtGui.QTableWidgetItem()
            self.traj_table.setItem( trix, 0, item)
            item.setCheckState(False)
            item.setFlags(QtCore.Qt.ItemIsUserCheckable | item.flags())
            
            # Trajectory ID
            self.traj_table.setItem( trix, 1, 
                QtGui.QTableWidgetItem(str(traj.trajid())) )
            
            # Trajectory length.
            self.traj_table.setItem( trix, 2, QtGui.QTableWidgetItem())
            self.traj_table.item(trix, 2).setData(
                QtCore.Qt.DisplayRole, len(traj))
            
            # Update data range:
            if len(traj) > max_length:
                max_length = len(traj)
        
        self.traj_table.sortItems(2, QtCore.Qt.DescendingOrder)
        self.length_from.setRange(0, max_length)
        self.length_to.setRange(0, max_length)
        self.length_to.setValue(max_length)
        
        # Connections: each checkbox toggles one item, the buttons toggle
        # all trajectories.
        self.traj_table.itemChanged.connect(
            lambda it: self.toggle_trajectory(
                self.traj_table.item(it.row(), 1).data(
                    QtCore.Qt.DisplayRole).toInt()[0], 
                self.traj_table.item(it.row(), 0).checkState() == QtCore.Qt.Checked)
        )
        self.mark_all.clicked.connect(self.mark_all_trajects)
        self.mark_none.clicked.connect(self.mark_no_trajects)
        self.mark_invert.clicked.connect(self.invert_marks)
    
    def init_cams(self, cals, im_names, cpar, targ_par):
        """
        Initializes each camera panel in turn. 
        
        Arguments:
        cals - a list of Calibration objects, one per camera.
        img_names - for each camera, path to the image file it displays.
        cpar - a ControlParams instance holding common scene data such as image
            size.
        targ_par - a TargetParams object, not needed for us but needed for the 
            base class.
        """
        self._cpar = cpar
        
        cam_panels = self.findChildren(CameraPanel)
        cam_nums = list(range(len(cam_panels)))
        
        for num, cal, img, panel in zip(cam_nums, cals, im_names, cam_panels):
            panel.reset(cpar, num, cal=cal, target_pars=targ_par)
            panel.set_image(img)
    
    def toggle_trajectory(self, trajid, visibility):
        if trajid in self._traj_cache:
            for panel in self.findChildren(CameraPanel):
                panel._patch_sets[trajid].set_visibility(visibility)
        else:
            self._traj_cache[trajid] = True
            self.add_trajectory(trajid)
    
    def mark_all_trajects(self):
        for trj_ix in range(self.traj_table.rowCount()):
            length = self.traj_table.item(trj_ix, 2).data(
                QtCore.Qt.DisplayRole).toInt()[0] 
            
            if (length >= self.length_from.value()) and \
                (length <= self.length_to.value()):
                self.traj_table.item(trj_ix, 0).setCheckState(QtCore.Qt.Checked)
    
    def mark_no_trajects(self):
        for trj_ix in range(self.traj_table.rowCount()):
            length = self.traj_table.item(trj_ix, 2).data(
                QtCore.Qt.DisplayRole).toInt()[0] 
            
            if (length >= self.length_from.value()) and \
                (length <= self.length_to.value()):
                self.traj_table.item(trj_ix, 0).setCheckState(QtCore.Qt.Unchecked)
    
    def invert_marks(self):
        for trj_ix in range(self.traj_table.rowCount()):
            if self.traj_table.item(trj_ix, 0).checkState() == QtCore.Qt.Unchecked:
                self.traj_table.item(trj_ix, 0).setCheckState(
                    QtCore.Qt.Checked)
            else:
                self.traj_table.item(trj_ix, 0).setCheckState(
                    QtCore.Qt.Unchecked)
    
    def add_trajectory(self, trajid):
        """
        Creates on each camera a patch-set for the trajectory of the given ID. 
        The points are generated by projecting the 3D positions on each sensor
        in the same way it would be done for calibration points. No target data
        is needed.
        
        Arguments:
        trajid - the trajectory identifier number.
        """
        points = self._scn.trajectory_by_id(trajid).pos() * 1000 # expects mm
        cam_panels = self.findChildren(CameraPanel)
        
        pen = QtGui.QPen(QtGui.QColor("yellow"))
        bpen = QtGui.QPen(QtGui.QColor("cyan"))
        rad = 5
        
        for panel in cam_panels:
            proj = convert_arr_metric_to_pixel(image_coordinates(
                points, panel.calibration(), 
                self._cpar.get_multimedia_params()), self._cpar)
            
            # Now draw it:
            panel.add_patchset(trajid)
            for x, y in proj:
                p = panel._scene.addEllipse(x - rad, y - rad, 2*rad, 2*rad, 
                    pen=pen)
                panel._patch_sets[trajid].push(p)
            
            # Mark beginning of trajectory:
            p = panel._scene.addEllipse(
                proj[0,0] - rad, proj[0,1] - rad, 2*rad, 2*rad, pen=bpen)
            panel._patch_sets[trajid].push(p)
            
    
    def _iter_selected_trajectories(self):
        """
        Iterates over trajectories, yielding the selected ones.
        """
        for trj_ix in range(self.traj_table.rowCount()):
            chk_item = self.traj_table.item(trj_ix, 0)
            
            if chk_item.checkState() == QtCore.Qt.Checked:
                trid = self.traj_table.item(trj_ix, 1).data(
                    QtCore.Qt.DisplayRole).toInt()[0]
                yield self._scn.trajectory_by_id(trid)
        
    def export_selected(self):
        """
        Writes out the only trajectories selected in the table into a new FHDF
        file, leaving all the chaff in the old file. Will ask for a filename.
        """
        fname = QtGui.QFileDialog.getSaveFileName(
            caption="Clean scene file save", filter="Flowtracks data (*.h5)")
        save_particles_table(str(fname), self._iter_selected_trajectories())
        
        
    def __del__(self):
        self._scn = None

if __name__ == "__main__":
    import sys, argparse
    from util.openptv import read_scene_config
    
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="The FHDF file with trajectory data")
    parser.add_argument('config', help="A scene-parameters YAML.")
    args = parser.parse_args()
    
    yaml_args, cam_args, cpar = read_scene_config(args.config)
    cals = [ca['calib'] for ca in cam_args]
    imgs = [ca['image'] for ca in cam_args]
    
    app = QtGui.QApplication([])
    window = TrajectoriesWindow()
    
    window.setGeometry(100, 50, 1200, 900)
    window.show()
    
    window.init_cams(cals, imgs, cpar, yaml_args['targ_par'])
    window.set_trajectories(args.data)
    
    # Proper destruction sequence to get rid of pytables' "closing remaining 
    # open files" message.
    state = app.exec_()
    window = None
    sys.exit(state)
