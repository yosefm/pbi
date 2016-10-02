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

from PyQt4 import QtCore, QtGui
#from optv.calibration import Calibration
from trajectories_base import Ui_TrajectoriesSelector
#from cam_panel import CameraPanel

from flowtracks.scene import Scene

class TrajectoriesWindow(QtGui.QWidget, Ui_TrajectoriesSelector):
    """
    Shows trajectories in a 4-camera grid, and coordinates the selection and
    highlighting activities.
    """
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.setupUi(self)
    
    def set_trajectories(self, filename):
        """
        Fill out the selection table and initialize trajectory display.
        """
        self._scn = Scene(filename)
        self.traj_table.setColumnCount(2)
        self.traj_table.setHorizontalHeaderLabels(["ID", "length"])
        
        for trix, traj in enumerate(self._scn.iter_trajectories()):
            self.traj_table.setRowCount(trix + 1)
            self.traj_table.setItem( trix, 0, 
                QtGui.QTableWidgetItem(str(traj.trajid())) )
            self.traj_table.setItem( trix, 1, QtGui.QTableWidgetItem())
            self.traj_table.item(trix, 1).setData(
                QtCore.Qt.DisplayRole,len(traj))
        
        self.traj_table.sortItems(1, QtCore.Qt.DescendingOrder)

if __name__ == "__main__":
    import sys
    
    app = QtGui.QApplication([])
    window = TrajectoriesWindow()
    
    #br = window._scene.itemsBoundingRect()
    window.setGeometry(100, 50, 1200, 900)
    window.show()
    
    window.set_trajectories(sys.argv[1])
    sys.exit(app.exec_())