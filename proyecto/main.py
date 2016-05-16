#!/usr/bin/env python

import sys
from PyQt4 import QtCore, QtGui
from vtk.qt4.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from vtk.util import numpy_support
from image_processing import ImageProcessing


class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.frame = QtGui.QFrame()
        self.vl = QtGui.QGridLayout()

        self.vtkWidget_right_femur = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget_right_hip = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget_left_femur = QVTKRenderWindowInteractor(self.frame)
        self.vtkWidget_left_hip = QVTKRenderWindowInteractor(self.frame)

        self.vl.addWidget(self.vtkWidget_right_femur, 0, 0)
        self.vl.addWidget(self.vtkWidget_right_hip, 0, 1)
        self.vl.addWidget(self.vtkWidget_left_femur, 1, 0)
        self.vl.addWidget(self.vtkWidget_left_hip, 1, 1)

        self.ren_right_femur = vtk.vtkRenderer()
        self.ren_right_hip = vtk.vtkRenderer()
        self.ren_left_femur = vtk.vtkRenderer()
        self.ren_left_hip = vtk.vtkRenderer()

        self.vtkWidget_right_femur.GetRenderWindow().AddRenderer(self.ren_right_femur)
        self.vtkWidget_right_hip.GetRenderWindow().AddRenderer(self.ren_right_hip)
        self.vtkWidget_left_femur.GetRenderWindow().AddRenderer(self.ren_left_femur)
        self.vtkWidget_left_hip.GetRenderWindow().AddRenderer(self.ren_left_hip)

        self.iren_right_femur = self.vtkWidget_right_femur.GetRenderWindow().GetInteractor()
        self.iren_right_hip = self.vtkWidget_right_hip.GetRenderWindow().GetInteractor()
        self.iren_left_femur = self.vtkWidget_left_femur.GetRenderWindow().GetInteractor()
        self.iren_left_hip = self.vtkWidget_left_hip.GetRenderWindow().GetInteractor()

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.show_dialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)

        self.image_processing = ImageProcessing()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.setGeometry(50, 50, 1200, 800)
        self.show()

    def show_dialog(self):
        fname = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', '', QtGui.QFileDialog.ShowDirsOnly)
        if fname:
            self.image_processing.execute(str(fname))
            self.update_gui()

    def update_gui(self):
        self.clean_gui()
        self.process_femurs()
        self.process_hips()
        self.show()

    def clean_gui(self):
        self.ren_right_femur = vtk.vtkRenderer()
        self.ren_right_hip = vtk.vtkRenderer()
        self.ren_left_femur = vtk.vtkRenderer()
        self.ren_left_hip = vtk.vtkRenderer()

        self.vtkWidget_right_femur.GetRenderWindow().AddRenderer(self.ren_right_femur)
        self.vtkWidget_right_hip.GetRenderWindow().AddRenderer(self.ren_right_hip)
        self.vtkWidget_left_femur.GetRenderWindow().AddRenderer(self.ren_left_femur)
        self.vtkWidget_left_hip.GetRenderWindow().AddRenderer(self.ren_left_hip)

        self.iren_right_femur = self.vtkWidget_right_femur.GetRenderWindow().GetInteractor()
        self.iren_right_hip = self.vtkWidget_right_hip.GetRenderWindow().GetInteractor()
        self.iren_left_femur = self.vtkWidget_left_femur.GetRenderWindow().GetInteractor()
        self.iren_left_hip = self.vtkWidget_left_hip.GetRenderWindow().GetInteractor()

    def process_hips(self):
        hips = self.image_processing.segmented_hips.copy()
        for key in hips.keys():
            actor = self.process_image(hips[key])
            if key == self.image_processing.RIGHT_LEG:
                self.ren_right_hip.AddActor(actor)
                self.ren_right_hip.ResetCamera()
                self.iren_right_hip.Initialize()
            else:
                self.ren_left_hip.AddActor(actor)
                self.ren_left_hip.ResetCamera()
                self.iren_left_hip.Initialize()

    def process_femurs(self):
        femurs = self.image_processing.segmented_legs.copy()
        for key in femurs.keys():
            actor = self.process_image(femurs[key])
            if key == self.image_processing.RIGHT_LEG:
                self.ren_right_femur.AddActor(actor)
                self.ren_right_femur.ResetCamera()
                self.iren_right_femur.Initialize()
            else:
                self.ren_left_femur.AddActor(actor)
                self.ren_left_femur.ResetCamera()
                self.iren_left_femur.Initialize()

    def process_image(self, image):
        dims = image.shape
        width = dims[1]
        height = dims[2]
        depth = dims[0]
        vtk_data = numpy_support.numpy_to_vtk(num_array=image.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

        imgdat = vtk.vtkImageData()
        imgdat.GetPointData().SetScalars(vtk_data)
        imgdat.SetDimensions(height, width, depth)
        imgdat.SetOrigin(0, 0, 0)
        spacing = self.image_processing.spacing
        imgdat.SetSpacing(spacing[0], spacing[1], spacing[2])

        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputData(imgdat)
        dmc.GenerateValues(1, 1, 1)
        dmc.Update()

        smoothingIterations = 15
        passBand = 0.001
        featureAngle = 120.0

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(dmc.GetOutputPort())
        smoother.SetNumberOfIterations(smoothingIterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(featureAngle)
        smoother.SetPassBand(passBand)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return actor


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
