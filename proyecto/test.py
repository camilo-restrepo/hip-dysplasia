from image_processing import ImageProcessing
import matplotlib.pyplot as plt

class Formatter(object):
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        z = self.im[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


# path = "/Volumes/Files/imagenes/ALMANZA_RUIZ_JUAN_CARLOS/TAC_DE_PELVIS - 84441/_Bone_30_2/"
# path = "/Volumes/Files/imagenes/AVILA_MALAGON_ZULMA_IVONNE/TAC_DE_PELVIS_SIMPLE - 89589/_Bone_30_2/"
path = "/Volumes/Files/imagenes/ALVAREZ_PATINO_SOFIA/PELVIS - 73864/_Bone_30_3/"
# path = "/Volumes/Files/imagenes/CHACON_BARBA_SERGIO_ANDRES/CADERA_SIMPLE - 103000/_Bone_30_2/"
# path = "/Volumes/Files/imagenes/LOAIZA_ORTIZ_JONATHAN_ESTEVEN/CADERA_IZQUIERDA - 94647/_Bone_30_2/"

c = ImageProcessing()
c.execute(path)
legs = c.legs
no_noise = c.no_noise
emphasized = c.emphasized
initial_segmented = c.initial_segmented
segmented_legs = c.segmented_legs
segmented_hips = c.segmented_hips

for leg_key in legs.keys():
    for k in range(30, 42):
        fig = plt.figure(k)
        a = fig.add_subplot(1, 4, 1)
        img = plt.imshow(emphasized[leg_key][k, :, :], cmap='Greys_r', interpolation="none")
        a.format_coord = Formatter(emphasized[leg_key][k, :, :])
        a = fig.add_subplot(1, 4, 2)
        img = plt.imshow(initial_segmented[leg_key][k, :, :], cmap='Greys_r', interpolation="none")
        a = fig.add_subplot(1, 4, 3)
        img = plt.imshow(segmented_legs[leg_key][k, :, :], cmap='Greys_r', interpolation="none")
        a = fig.add_subplot(1, 4, 4)
        img = plt.imshow(segmented_hips[leg_key][k, :, :], cmap='Greys_r', interpolation="none")
        if k % 20 == 0:
            plt.show()
    plt.show()
