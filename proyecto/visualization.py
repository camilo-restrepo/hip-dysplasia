import vtk
import pickle
import numpy as np
from vtk.util import numpy_support

file = open('femur.txt', 'r')
data = pickle.load(file)
file.close()

dims = data.shape
width = dims[1]
height = dims[2]
depth = dims[0]

NumPy_data_shape = data.shape
VTK_data = numpy_support.numpy_to_vtk(num_array=data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

imgdat = vtk.vtkImageData()
imgdat.GetPointData().SetScalars(VTK_data)
imgdat.SetDimensions(height, width, depth)
imgdat.SetOrigin(0, 0, 0)
imgdat.SetSpacing(0.653, 0.653, 3.0)

imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
imageDataGeometryFilter.SetInputData(imgdat)
imageDataGeometryFilter.Update()

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

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.ResetCamera()

renderWin = vtk.vtkRenderWindow()
renderWin.SetSize(600, 600)
renderWin.AddRenderer(renderer)

renderInteractor = vtk.vtkRenderWindowInteractor()
renderInteractor.SetRenderWindow(renderWin)


# A simple function to be called when the user decides to quit the application.
def exitCheck(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)

# Tell the application to use the function as an exit check.
renderWin.AddObserver("AbortCheckEvent", exitCheck)
renderInteractor.Initialize()
renderWin.Render()
renderInteractor.Start()
