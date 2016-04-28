import utils
import valley
import segmentation
import time
import matplotlib.pyplot as plt
import numpy as np


# PathDicom = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# PathDicom = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"
# PathDicom = "/home/camilo/Documents/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
PathDicom = "/home/camilo/Documents/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"

t0 = time.time()
img_original_array, img_smooth_array = utils.load_dicom(PathDicom)
emphasized_img = valley.get_valley_emphasized_image(img_smooth_array)
bone_mask = segmentation.initial_segmentation(emphasized_img)
t1 = time.time()
# print t1-t0 ----- 47.1907639503

result = np.zeros_like(img_smooth_array)
np.multiply(emphasized_img, bone_mask, result)

ini = 20
end = 50
i = 0
for k in range(ini, end):
    utils.np_show(result[k, :, :])
    i += 1
    if i == 20:
        plt.show()
        i = 0

plt.show()
