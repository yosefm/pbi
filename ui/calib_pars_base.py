# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calib_pars_base.ui'
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

class Ui_calibPars(object):
    def setupUi(self, calibPars):
        calibPars.setObjectName(_fromUtf8("calibPars"))
        calibPars.resize(622, 370)
        self.verticalLayout = QtGui.QVBoxLayout(calibPars)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label_7 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.verticalLayout.addWidget(self.label_7)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.posz = QtGui.QDoubleSpinBox(calibPars)
        self.posz.setDecimals(10)
        self.posz.setMinimum(-500.0)
        self.posz.setMaximum(500.0)
        self.posz.setObjectName(_fromUtf8("posz"))
        self.gridLayout.addWidget(self.posz, 0, 5, 1, 1)
        self.label_3 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 0, 4, 1, 1)
        self.label_2 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.posx = QtGui.QDoubleSpinBox(calibPars)
        self.posx.setDecimals(10)
        self.posx.setMinimum(-500.0)
        self.posx.setMaximum(500.0)
        self.posx.setObjectName(_fromUtf8("posx"))
        self.gridLayout.addWidget(self.posx, 0, 1, 1, 1)
        self.posy = QtGui.QDoubleSpinBox(calibPars)
        self.posy.setDecimals(10)
        self.posy.setMinimum(-500.0)
        self.posy.setMaximum(500.0)
        self.posy.setObjectName(_fromUtf8("posy"))
        self.gridLayout.addWidget(self.posy, 0, 3, 1, 1)
        self.label = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.ang_omega = QtGui.QDoubleSpinBox(calibPars)
        self.ang_omega.setDecimals(10)
        self.ang_omega.setMinimum(-3.15)
        self.ang_omega.setMaximum(3.15)
        self.ang_omega.setSingleStep(0.01)
        self.ang_omega.setObjectName(_fromUtf8("ang_omega"))
        self.gridLayout.addWidget(self.ang_omega, 1, 1, 1, 1)
        self.label_6 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.gridLayout.addWidget(self.label_6, 1, 2, 1, 1)
        self.ang_phi = QtGui.QDoubleSpinBox(calibPars)
        self.ang_phi.setDecimals(10)
        self.ang_phi.setMinimum(-6.3)
        self.ang_phi.setMaximum(6.3)
        self.ang_phi.setSingleStep(0.01)
        self.ang_phi.setObjectName(_fromUtf8("ang_phi"))
        self.gridLayout.addWidget(self.ang_phi, 1, 3, 1, 1)
        self.label_4 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 1, 4, 1, 1)
        self.ang_kappa = QtGui.QDoubleSpinBox(calibPars)
        self.ang_kappa.setDecimals(10)
        self.ang_kappa.setMinimum(-3.15)
        self.ang_kappa.setMaximum(3.15)
        self.ang_kappa.setSingleStep(0.01)
        self.ang_kappa.setObjectName(_fromUtf8("ang_kappa"))
        self.gridLayout.addWidget(self.ang_kappa, 1, 5, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.label_13 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.verticalLayout.addWidget(self.label_13)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.label_14 = QtGui.QLabel(calibPars)
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.horizontalLayout_4.addWidget(self.label_14)
        self.intpar_cc = QtGui.QDoubleSpinBox(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.intpar_cc.sizePolicy().hasHeightForWidth())
        self.intpar_cc.setSizePolicy(sizePolicy)
        self.intpar_cc.setDecimals(5)
        self.intpar_cc.setMaximum(10000.0)
        self.intpar_cc.setProperty("value", 100.0)
        self.intpar_cc.setObjectName(_fromUtf8("intpar_cc"))
        self.horizontalLayout_4.addWidget(self.intpar_cc)
        self.use_cc = QtGui.QCheckBox(calibPars)
        self.use_cc.setText(_fromUtf8(""))
        self.use_cc.setObjectName(_fromUtf8("use_cc"))
        self.horizontalLayout_4.addWidget(self.use_cc)
        self.label_15 = QtGui.QLabel(calibPars)
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.horizontalLayout_4.addWidget(self.label_15)
        self.intpar_xh = QtGui.QDoubleSpinBox(calibPars)
        self.intpar_xh.setDecimals(5)
        self.intpar_xh.setMinimum(-50.0)
        self.intpar_xh.setMaximum(50.0)
        self.intpar_xh.setSingleStep(0.05)
        self.intpar_xh.setObjectName(_fromUtf8("intpar_xh"))
        self.horizontalLayout_4.addWidget(self.intpar_xh)
        self.use_xh = QtGui.QCheckBox(calibPars)
        self.use_xh.setText(_fromUtf8(""))
        self.use_xh.setObjectName(_fromUtf8("use_xh"))
        self.horizontalLayout_4.addWidget(self.use_xh)
        self.label_16 = QtGui.QLabel(calibPars)
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.horizontalLayout_4.addWidget(self.label_16)
        self.intpar_yh = QtGui.QDoubleSpinBox(calibPars)
        self.intpar_yh.setDecimals(5)
        self.intpar_yh.setMinimum(-50.0)
        self.intpar_yh.setMaximum(50.0)
        self.intpar_yh.setSingleStep(0.05)
        self.intpar_yh.setObjectName(_fromUtf8("intpar_yh"))
        self.horizontalLayout_4.addWidget(self.intpar_yh)
        self.use_yh = QtGui.QCheckBox(calibPars)
        self.use_yh.setText(_fromUtf8(""))
        self.use_yh.setObjectName(_fromUtf8("use_yh"))
        self.horizontalLayout_4.addWidget(self.use_yh)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.label_8 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.verticalLayout.addWidget(self.label_8)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_9 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout.addWidget(self.label_9)
        self.radial_k1 = QtGui.QDoubleSpinBox(calibPars)
        self.radial_k1.setDecimals(10)
        self.radial_k1.setMinimum(-500.0)
        self.radial_k1.setMaximum(500.0)
        self.radial_k1.setObjectName(_fromUtf8("radial_k1"))
        self.horizontalLayout.addWidget(self.radial_k1)
        self.use_k1 = QtGui.QCheckBox(calibPars)
        self.use_k1.setText(_fromUtf8(""))
        self.use_k1.setObjectName(_fromUtf8("use_k1"))
        self.horizontalLayout.addWidget(self.use_k1)
        self.radial_k2 = QtGui.QDoubleSpinBox(calibPars)
        self.radial_k2.setDecimals(10)
        self.radial_k2.setMinimum(-500.0)
        self.radial_k2.setMaximum(500.0)
        self.radial_k2.setObjectName(_fromUtf8("radial_k2"))
        self.horizontalLayout.addWidget(self.radial_k2)
        self.use_k2 = QtGui.QCheckBox(calibPars)
        self.use_k2.setText(_fromUtf8(""))
        self.use_k2.setObjectName(_fromUtf8("use_k2"))
        self.horizontalLayout.addWidget(self.use_k2)
        self.radial_k3 = QtGui.QDoubleSpinBox(calibPars)
        self.radial_k3.setDecimals(10)
        self.radial_k3.setMinimum(-500.0)
        self.radial_k3.setMaximum(500.0)
        self.radial_k3.setObjectName(_fromUtf8("radial_k3"))
        self.horizontalLayout.addWidget(self.radial_k3)
        self.use_k3 = QtGui.QCheckBox(calibPars)
        self.use_k3.setText(_fromUtf8(""))
        self.use_k3.setObjectName(_fromUtf8("use_k3"))
        self.horizontalLayout.addWidget(self.use_k3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_10 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.horizontalLayout_2.addWidget(self.label_10)
        self.decent_p1 = QtGui.QDoubleSpinBox(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.decent_p1.sizePolicy().hasHeightForWidth())
        self.decent_p1.setSizePolicy(sizePolicy)
        self.decent_p1.setDecimals(10)
        self.decent_p1.setMinimum(-10.0)
        self.decent_p1.setMaximum(10.0)
        self.decent_p1.setObjectName(_fromUtf8("decent_p1"))
        self.horizontalLayout_2.addWidget(self.decent_p1)
        self.use_p1 = QtGui.QCheckBox(calibPars)
        self.use_p1.setText(_fromUtf8(""))
        self.use_p1.setObjectName(_fromUtf8("use_p1"))
        self.horizontalLayout_2.addWidget(self.use_p1)
        self.decent_p2 = QtGui.QDoubleSpinBox(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.decent_p2.sizePolicy().hasHeightForWidth())
        self.decent_p2.setSizePolicy(sizePolicy)
        self.decent_p2.setDecimals(10)
        self.decent_p2.setMinimum(-10.0)
        self.decent_p2.setMaximum(10.0)
        self.decent_p2.setObjectName(_fromUtf8("decent_p2"))
        self.horizontalLayout_2.addWidget(self.decent_p2)
        self.use_p2 = QtGui.QCheckBox(calibPars)
        self.use_p2.setText(_fromUtf8(""))
        self.use_p2.setObjectName(_fromUtf8("use_p2"))
        self.horizontalLayout_2.addWidget(self.use_p2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_11 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.horizontalLayout_3.addWidget(self.label_11)
        self.scale = QtGui.QDoubleSpinBox(calibPars)
        self.scale.setDecimals(5)
        self.scale.setMinimum(-500.0)
        self.scale.setMaximum(500.0)
        self.scale.setObjectName(_fromUtf8("scale"))
        self.horizontalLayout_3.addWidget(self.scale)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.label_12 = QtGui.QLabel(calibPars)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.horizontalLayout_3.addWidget(self.label_12)
        self.shear = QtGui.QDoubleSpinBox(calibPars)
        self.shear.setDecimals(5)
        self.shear.setMinimum(-500.0)
        self.shear.setMaximum(500.0)
        self.shear.setObjectName(_fromUtf8("shear"))
        self.horizontalLayout_3.addWidget(self.shear)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.label_17 = QtGui.QLabel(calibPars)
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.verticalLayout.addWidget(self.label_17)

        self.retranslateUi(calibPars)
        QtCore.QMetaObject.connectSlotsByName(calibPars)

    def retranslateUi(self, calibPars):
        calibPars.setWindowTitle(_translate("calibPars", "Form", None))
        self.label_7.setText(_translate("calibPars", "Exterior parameters [mm, rad]", None))
        self.label_3.setText(_translate("calibPars", "z", None))
        self.label_2.setText(_translate("calibPars", "y", None))
        self.label.setText(_translate("calibPars", "x", None))
        self.label_5.setText(_translate("calibPars", "omega", None))
        self.label_6.setText(_translate("calibPars", "phi", None))
        self.label_4.setText(_translate("calibPars", "kappa", None))
        self.label_13.setText(_translate("calibPars", "Primary point positioning", None))
        self.label_14.setText(_translate("calibPars", "Distance from sensor", None))
        self.label_15.setText(_translate("calibPars", "X shift", None))
        self.label_16.setText(_translate("calibPars", "Y shift", None))
        self.label_8.setText(_translate("calibPars", "Camera distortion parameters", None))
        self.label_9.setText(_translate("calibPars", "<html><head/><body><p>Radial distortion (k<span style=\" vertical-align:sub;\">i</span>)</p></body></html>", None))
        self.label_10.setText(_translate("calibPars", "<html><head/><body><p>Decentering distortion (p<span style=\" vertical-align:sub;\">i</span>)</p></body></html>", None))
        self.label_11.setText(_translate("calibPars", "Scale ", None))
        self.label_12.setText(_translate("calibPars", "Shear", None))
        self.label_17.setText(_translate("calibPars", "Free variable checkboxes synced with parameters/orient.par in current directory. Ugly but temp.", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    calibPars = QtGui.QWidget()
    ui = Ui_calibPars()
    ui.setupUi(calibPars)
    calibPars.show()
    sys.exit(app.exec_())

