from image_processing import ImageProcessing
import matplotlib.pyplot as plt


path = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_10_4/"
# path = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_10_2/"
# path = "/Volumes/Files/imagenes/ALVAREZ_PATINO_SOFIA/PELVIS - 73864/_Bone_10_5/"
# path = "/Volumes/Files/imagenes/CHACON_BARBA_SERGIO_ANDRES/CADERA_SIMPLE - 103000/_Bone_10_4/"
# path = "/Volumes/Files/imagenes/LOAIZA_ORTIZ_JONATHAN_ESTEVEN/CADERA_IZQUIERDA - 94647/_Bone_10_5/"
# path = "/Volumes/Files/imagenes/VARGAS_DIAZ_CLAUDIA_PATRICIA/PELVIS_3D - 84024/_Bone_10_5/"

c = ImageProcessing()
c.execute(path)
legs = c.legs
segmented_legs = c.segmented_legs
segmented_hips = c.segmented_hips

for leg_key in legs.keys():

    for k in range(100, legs[leg_key].shape[0]):
        fig = plt.figure(k)
        a = fig.add_subplot(1, 2, 1)
        img = plt.imshow(segmented_legs[leg_key][k, :, :], cmap='Greys_r', interpolation="none")
        a = fig.add_subplot(1, 2, 2)
        img = plt.imshow(segmented_hips[leg_key][k, :, :], cmap='Greys_r', interpolation="none")
        if k % 20 == 0:
            plt.show()
    plt.show()
