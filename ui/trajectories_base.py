# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'trajectories_base.ui'
#
# Created by: PyQt5 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_TrajectoriesSelector(object):
    def setupUi(self, TrajectoriesSelector):
        TrajectoriesSelector.setObjectName(_fromUtf8("TrajectoriesSelector"))
        TrajectoriesSelector.resize(1052, 690)
        self.horizontalLayout = QtGui.QHBoxLayout(TrajectoriesSelector)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.widget = CameraPanel(TrajectoriesSelector)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setObjectName(_fromUtf8("widget"))
        self.gridLayout.addWidget(self.widget, 0, 1, 1, 1)
        self.widget_2 = CameraPanel(TrajectoriesSelector)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setObjectName(_fromUtf8("widget_2"))
        self.gridLayout.addWidget(self.widget_2, 0, 0, 1, 1)
        self.widget_3 = CameraPanel(TrajectoriesSelector)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName(_fromUtf8("widget_3"))
        self.gridLayout.addWidget(self.widget_3, 1, 1, 1, 1)
        self.widget_4 = CameraPanel(TrajectoriesSelector)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setObjectName(_fromUtf8("widget_4"))
        self.gridLayout.addWidget(self.widget_4, 1, 0, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label = QtGui.QLabel(TrajectoriesSelector)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMinimumSize(QtCore.QSize(0, 0))
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_2.addWidget(self.label)
        self.mark_all = QtGui.QPushButton(TrajectoriesSelector)
        self.mark_all.setObjectName(_fromUtf8("mark_all"))
        self.horizontalLayout_2.addWidget(self.mark_all)
        self.mark_none = QtGui.QPushButton(TrajectoriesSelector)
        self.mark_none.setObjectName(_fromUtf8("mark_none"))
        self.horizontalLayout_2.addWidget(self.mark_none)
        self.mark_invert = QtGui.QPushButton(TrajectoriesSelector)
        self.mark_invert.setObjectName(_fromUtf8("mark_invert"))
        self.horizontalLayout_2.addWidget(self.mark_invert)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_2 = QtGui.QLabel(TrajectoriesSelector)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_4.addWidget(self.label_2)
        self.length_from = QtGui.QSpinBox(TrajectoriesSelector)
        self.length_from.setObjectName(_fromUtf8("length_from"))
        self.horizontalLayout_4.addWidget(self.length_from)
        self.label_3 = QtGui.QLabel(TrajectoriesSelector)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_4.addWidget(self.label_3)
        self.length_to = QtGui.QSpinBox(TrajectoriesSelector)
        self.length_to.setObjectName(_fromUtf8("length_to"))
        self.horizontalLayout_4.addWidget(self.length_to)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.traj_table = QtGui.QTableWidget(TrajectoriesSelector)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.traj_table.sizePolicy().hasHeightForWidth())
        self.traj_table.setSizePolicy(sizePolicy)
        self.traj_table.setMinimumSize(QtCore.QSize(180, 0))
        self.traj_table.setObjectName(_fromUtf8("traj_table"))
        self.traj_table.setColumnCount(0)
        self.traj_table.setRowCount(0)
        self.verticalLayout.addWidget(self.traj_table)
        self.btn_export = QtGui.QPushButton(TrajectoriesSelector)
        self.btn_export.setObjectName(_fromUtf8("btn_export"))
        self.verticalLayout.addWidget(self.btn_export)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(TrajectoriesSelector)
        QtCore.QMetaObject.connectSlotsByName(TrajectoriesSelector)

    def retranslateUi(self, TrajectoriesSelector):
        TrajectoriesSelector.setWindowTitle(_translate("TrajectoriesSelector", "Form", None))
        self.label.setText(_translate("TrajectoriesSelector", "Mark:", None))
        self.mark_all.setText(_translate("TrajectoriesSelector", "All", None))
        self.mark_none.setText(_translate("TrajectoriesSelector", "None", None))
        self.mark_invert.setText(_translate("TrajectoriesSelector", "Invert", None))
        self.label_2.setText(_translate("TrajectoriesSelector", "Length from", None))
        self.label_3.setText(_translate("TrajectoriesSelector", "to", None))
        self.btn_export.setText(_translate("TrajectoriesSelector", "Export", None))

from .cam_panel import CameraPanel
