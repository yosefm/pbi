# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cam_calib_base.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

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

class Ui_CameraCalibration(object):
    def setupUi(self, CameraCalibration):
        CameraCalibration.setObjectName(_fromUtf8("CameraCalibration"))
        CameraCalibration.resize(794, 788)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(CameraCalibration.sizePolicy().hasHeightForWidth())
        CameraCalibration.setSizePolicy(sizePolicy)
        self.horizontalLayout_2 = QtGui.QHBoxLayout(CameraCalibration)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.calpars = CalibParameters(CameraCalibration)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calpars.sizePolicy().hasHeightForWidth())
        self.calpars.setSizePolicy(sizePolicy)
        self.calpars.setMaximumSize(QtCore.QSize(500, 16777215))
        self.calpars.setObjectName(_fromUtf8("calpars"))
        self.verticalLayout_2.addWidget(self.calpars)
        self.groupBox_2 = QtGui.QGroupBox(CameraCalibration)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.btn_save_cal = QtGui.QPushButton(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_save_cal.sizePolicy().hasHeightForWidth())
        self.btn_save_cal.setSizePolicy(sizePolicy)
        self.btn_save_cal.setObjectName(_fromUtf8("btn_save_cal"))
        self.gridLayout_2.addWidget(self.btn_save_cal, 0, 2, 2, 1)
        self.label = QtGui.QLabel(self.groupBox_2)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.txt_addpar = QtGui.QLineEdit(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_addpar.sizePolicy().hasHeightForWidth())
        self.txt_addpar.setSizePolicy(sizePolicy)
        self.txt_addpar.setObjectName(_fromUtf8("txt_addpar"))
        self.gridLayout_2.addWidget(self.txt_addpar, 1, 1, 1, 1)
        self.txt_ori = QtGui.QLineEdit(self.groupBox_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_ori.sizePolicy().hasHeightForWidth())
        self.txt_ori.setSizePolicy(sizePolicy)
        self.txt_ori.setObjectName(_fromUtf8("txt_ori"))
        self.gridLayout_2.addWidget(self.txt_ori, 0, 1, 1, 1)
        self.label_2 = QtGui.QLabel(self.groupBox_2)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        self.groupBox_3 = QtGui.QGroupBox(CameraCalibration)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.gridLayout_3 = QtGui.QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label_3 = QtGui.QLabel(self.groupBox_3)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_3.addWidget(self.label_3, 1, 0, 1, 1)
        self.txt_detected = QtGui.QLineEdit(self.groupBox_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_detected.sizePolicy().hasHeightForWidth())
        self.txt_detected.setSizePolicy(sizePolicy)
        self.txt_detected.setObjectName(_fromUtf8("txt_detected"))
        self.gridLayout_3.addWidget(self.txt_detected, 1, 1, 1, 1)
        self.txt_matched = QtGui.QLineEdit(self.groupBox_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_matched.sizePolicy().hasHeightForWidth())
        self.txt_matched.setSizePolicy(sizePolicy)
        self.txt_matched.setObjectName(_fromUtf8("txt_matched"))
        self.gridLayout_3.addWidget(self.txt_matched, 2, 1, 1, 1)
        self.label_4 = QtGui.QLabel(self.groupBox_3)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout_3.addWidget(self.label_4, 2, 0, 1, 1)
        self.btn_dump_multi = QtGui.QPushButton(self.groupBox_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_dump_multi.sizePolicy().hasHeightForWidth())
        self.btn_dump_multi.setSizePolicy(sizePolicy)
        self.btn_dump_multi.setObjectName(_fromUtf8("btn_dump_multi"))
        self.gridLayout_3.addWidget(self.btn_dump_multi, 1, 2, 2, 1)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        spacerItem = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.cam = CalibPanel(CameraCalibration)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.cam.sizePolicy().hasHeightForWidth())
        self.cam.setSizePolicy(sizePolicy)
        self.cam.setMinimumSize(QtCore.QSize(400, 0))
        self.cam.setObjectName(_fromUtf8("cam"))
        self.verticalLayout.addWidget(self.cam)
        self.groupBox = QtGui.QGroupBox(CameraCalibration)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.gridLayout = QtGui.QGridLayout(self.groupBox)
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.show_hp = QtGui.QCheckBox(self.groupBox)
        self.show_hp.setObjectName(_fromUtf8("show_hp"))
        self.gridLayout.addWidget(self.show_hp, 0, 0, 1, 1)
        self.show_detect = QtGui.QCheckBox(self.groupBox)
        self.show_detect.setObjectName(_fromUtf8("show_detect"))
        self.gridLayout.addWidget(self.show_detect, 0, 1, 1, 1)
        self.show_project = QtGui.QCheckBox(self.groupBox)
        self.show_project.setObjectName(_fromUtf8("show_project"))
        self.gridLayout.addWidget(self.show_project, 1, 0, 1, 1)
        self.show_resids = QtGui.QCheckBox(self.groupBox)
        self.show_resids.setObjectName(_fromUtf8("show_resids"))
        self.gridLayout.addWidget(self.show_resids, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.groupBox)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.btn_load_man = QtGui.QPushButton(CameraCalibration)
        self.btn_load_man.setObjectName(_fromUtf8("btn_load_man"))
        self.horizontalLayout.addWidget(self.btn_load_man)
        self.btn_save_man = QtGui.QPushButton(CameraCalibration)
        self.btn_save_man.setObjectName(_fromUtf8("btn_save_man"))
        self.horizontalLayout.addWidget(self.btn_save_man)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.btn_detect = QtGui.QPushButton(CameraCalibration)
        self.btn_detect.setObjectName(_fromUtf8("btn_detect"))
        self.verticalLayout.addWidget(self.btn_detect)
        self.btn_raw = QtGui.QPushButton(CameraCalibration)
        self.btn_raw.setObjectName(_fromUtf8("btn_raw"))
        self.verticalLayout.addWidget(self.btn_raw)
        self.btn_number = QtGui.QPushButton(CameraCalibration)
        self.btn_number.setObjectName(_fromUtf8("btn_number"))
        self.verticalLayout.addWidget(self.btn_number)
        self.btn_full_calib = QtGui.QPushButton(CameraCalibration)
        self.btn_full_calib.setObjectName(_fromUtf8("btn_full_calib"))
        self.verticalLayout.addWidget(self.btn_full_calib)
        self.horizontalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(CameraCalibration)
        QtCore.QMetaObject.connectSlotsByName(CameraCalibration)

    def retranslateUi(self, CameraCalibration):
        CameraCalibration.setWindowTitle(_translate("CameraCalibration", "Form", None))
        self.groupBox_2.setTitle(_translate("CameraCalibration", "Output", None))
        self.btn_save_cal.setText(_translate("CameraCalibration", "Save", None))
        self.label.setText(_translate("CameraCalibration", "Ori file", None))
        self.label_2.setText(_translate("CameraCalibration", "Distortion parameters", None))
        self.groupBox_3.setTitle(_translate("CameraCalibration", "Save point sets", None))
        self.label_3.setText(_translate("CameraCalibration", "Detected", None))
        self.label_4.setText(_translate("CameraCalibration", "Matched", None))
        self.btn_dump_multi.setText(_translate("CameraCalibration", "Save", None))
        self.groupBox.setTitle(_translate("CameraCalibration", "Visibility", None))
        self.show_hp.setText(_translate("CameraCalibration", "Highpass results", None))
        self.show_detect.setText(_translate("CameraCalibration", "Detected targets", None))
        self.show_project.setText(_translate("CameraCalibration", "Projected reference", None))
        self.show_resids.setText(_translate("CameraCalibration", "Residuals", None))
        self.btn_load_man.setText(_translate("CameraCalibration", "Load manual detection", None))
        self.btn_save_man.setText(_translate("CameraCalibration", "Save manual detection", None))
        self.btn_detect.setText(_translate("CameraCalibration", "Detect calibration points", None))
        self.btn_raw.setText(_translate("CameraCalibration", "Coarse tuning of calibration", None))
        self.btn_number.setText(_translate("CameraCalibration", "Match detrections to reference", None))
        self.btn_full_calib.setText(_translate("CameraCalibration", "Fine tuning", None))

from .calib_panel import CalibPanel
from .calib_pars import CalibParameters

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    CameraCalibration = QtGui.QWidget()
    ui = Ui_CameraCalibration()
    ui.setupUi(CameraCalibration)
    CameraCalibration.show()
    sys.exit(app.exec_())

