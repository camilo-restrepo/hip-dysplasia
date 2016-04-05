import numpy as np
import SimpleITK
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk, reconstruction, remove_small_objects
import utils
import vtk
from vtk.util import numpy_support


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# outPath = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/out/"

# PathDicom = "D:\imagenes\ALMANZA_RUIZ_JUAN_CARLOS\TAC_DE_PELVIS - 84441\_Bone_30_2"
# outPath = "D:\imagenes\out"

reader = SimpleITK.ImageSeriesReader()
filenames_dicom = reader.GetGDCMSeriesFileNames(PathDicom)
reader.SetFileNames(filenames_dicom)
img_original = reader.Execute()

smooth_filter = SimpleITK.CurvatureFlowImageFilter()
smooth_filter.SetTimeStep(0.125)
smooth_filter.SetNumberOfIterations(5)

closing_radius = disk(1)
remove_small_objects_size = 80


def get_bone_mask(image):
    # Remove noise
    img_smooth = smooth_filter.Execute(image)
    img_smooth_array = SimpleITK.GetArrayFromImage(img_smooth)

    bone = np.zeros_like(img_smooth_array)
    bone[img_smooth_array < 150] = 0
    bone[img_smooth_array > 150] = 1
    bone = closing(bone, closing_radius)
    bone = remove_small_objects(bone.astype(bool), remove_small_objects_size)
    seed = np.copy(bone)
    seed[1:-1, 1:-1] = bone.max()
    mask = bone
    bone = reconstruction(seed, mask, method='erosion')

    return bone


def get_segmented_image(image):
    img_array = SimpleITK.GetArrayFromImage(image)
    mask = get_bone_mask(image)
    img_array = np.multiply(img_array, mask)
    img_array[img_array < 0] = 0
    return img_array


mask_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))
# thresholded_array = np.zeros((img_original.GetWidth(), img_original.GetHeight(), img_original.GetDepth()))

for i in range(0, img_original.GetDepth()):
    mask_array[:, :, i] = get_bone_mask(img_original[:, :, i])
    # utils.np_show(mask_array[:, :, i])
    # thresholded_array[:, :, i] = get_segmented_image(img_original[:, :, i])
    # utils.np_show(thresholded_array[:, :, i])

# vtk_data = numpy_support.numpy_to_vtk(mask_array.ravel(), deep=False)

dims = mask_array.shape
spacing = img_original.GetSpacing()

dataImporter = vtk.vtkImageImport()
dataImporter.SetDataScalarTypeToFloat()
dataImporter.SetNumberOfScalarComponents(1)
dataImporter.SetDataExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
dataImporter.SetWholeExtent(0, dims[2]-1, 0, dims[1]-1, 0, dims[0]-1)
dataImporter.SetDataSpacing(spacing[0], spacing[1], spacing[2])
dataImporter.CopyImportVoidPointer(mask_array, mask_array.nbytes)

ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(dataImporter.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)

ren.AddActor(actor)
iren.Initialize()
renWin.Render()
iren.Start()
