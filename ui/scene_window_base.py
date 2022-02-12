# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'single_cam/scene_window_base.ui'
#
# Created: Mon Oct 19 14:31:31 2015
#      by: PyQt5 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)

class Ui_Scene(object):
    def setupUi(self, Scene):
        Scene.setObjectName(_fromUtf8("Scene"))
        Scene.resize(718, 652)
        self.gridLayout = QtWidgets.QGridLayout(Scene)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.cam1 = CamPanelEpi(Scene)
        self.cam1.setObjectName(_fromUtf8("cam1"))
        self.gridLayout.addWidget(self.cam1, 0, 0, 1, 1)
        self.cam2 = CamPanelEpi(Scene)
        self.cam2.setObjectName(_fromUtf8("cam2"))
        self.gridLayout.addWidget(self.cam2, 0, 1, 1, 1)
        self.cam3 = CamPanelEpi(Scene)
        self.cam3.setObjectName(_fromUtf8("cam3"))
        self.gridLayout.addWidget(self.cam3, 1, 0, 1, 1)
        self.cam4 = CamPanelEpi(Scene)
        self.cam4.setObjectName(_fromUtf8("cam4"))
        self.gridLayout.addWidget(self.cam4, 1, 1, 1, 1)

        self.retranslateUi(Scene)
        QtCore.QMetaObject.connectSlotsByName(Scene)

    def retranslateUi(self, Scene):
        Scene.setWindowTitle(_translate("Scene", "Form", None))

from .epi_panel import CamPanelEpi

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Scene = QtWidgets.Qwidget()
    ui = Ui_Scene()
    ui.setupUi(Scene)
    Scene.show()
    sys.exit(app.exec_())

