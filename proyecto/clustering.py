import numpy as np
from scipy.spatial.distance import cdist
import pickle


idx = 'idx'
yx = 'yx'
d1 = 'd1'
d2 = 'd2'
m1 = 'm1'
m2 = 'm2'
alpha = 0.85
p = 2


def generate_centroids(ncentroids, mean, std):
    centroids = np.empty(2)
    for i in range(0, ncentroids):
        centroids[i] = np.random.normal(mean, std)
    return centroids


def get_neighbors(point, image):
    z = point[0]
    x = point[1]
    y = point[2]

    x_ini = x - 5
    x_end = x + 6
    y_ini = y - 5
    y_end = y + 6
    z_ini = z - 2
    z_end = z + 3

    vecinos_x = image[z_ini:z_end, x_ini:x_end, y_ini:y_end]
    size = vecinos_x.shape[0] * vecinos_x.shape[1] * vecinos_x.shape[2]
    data = np.zeros(size, dtype={'names': [idx, yx, d1, d2, m1, m2], 'formats': ['3int8', 'f4', 'f4', 'f4', 'f4', 'f4']})

    i = 0
    for z in range(0, vecinos_x.shape[0]):
        for x in range(0, vecinos_x.shape[1]):
            for y in range(0, vecinos_x.shape[2]):
                data[i] = ([z, x, y], vecinos_x[z, x, y], 0, 0, 0, 0)
                i += 1

    return data


def precalc_distances(data, centroids):
    centroids = np.reshape(centroids, (2, 1))
    for i in range(0, data.shape[0]):
        px_value = np.reshape(data[i][yx], (1, 1))
        distances = cdist(px_value, centroids, 'euclidean')
        data[i][d1] = distances[0, 0]
        data[i][d2] = distances[0, 1]
    return data


def get_divisor(data, centroids, i):
    coef = alpha / data.shape[0]
    divisor = 0
    sum_d1 = sum(data[d1])
    sum_d2 = sum(data[d2])

    for cdx, c in enumerate(centroids):
        cdx += 4
        euclidean_acc = sum_d1
        get = d1
        if cdx == 5:
            euclidean_acc = sum_d2
            get = d2

        euclidean_distance = data[i][get]
        divisor += (euclidean_distance + (coef * euclidean_acc))**(-1/(p - 1))
    return divisor


def get_dividend(data, cdx, i):
    coef = alpha / data.shape[0]
    get = d1
    if cdx == 5:
        get = d2

    euclidean_distance = data[i][get]
    euclidean_acc = sum(data[get])
    dividend = (euclidean_distance + (coef * euclidean_acc))**(-1/(p - 1))
    return dividend


def compute_membership(data, centroids):
    data = precalc_distances(data, centroids)
    for i in range(0, data.shape[0]):
        for cdx, c in enumerate(centroids):
            cdx += 4
            dividend = get_dividend(data, cdx, i)
            divisor = get_divisor(data, centroids, i)
            u = dividend / divisor
            if cdx == 4:
                data[i][m1] = u
            else:
                data[i][m2] = u
    return data


def recalc_centroids(centroids, data):
    coef = alpha / data.shape[0]
    sum_yr = sum(data[yx])*coef

    for cdx, c in enumerate(centroids):
        cdx += 4
        get = m1
        if cdx == 5:
            get = m2

        u = data[get]**p
        ys = data[yx] + sum_yr
        dividend = sum(np.multiply(u, ys))
        sum_u = (1 + alpha) * sum(u)
        centroids[cdx-4] = dividend / sum_u

    return centroids


def evaluate_condition(data, data_old):
    if sum(data[m1]) == 0:
        return True

    sum_m1 = sum(data[m1] - data_old[m1])
    sum_m2 = sum(data[m2] - data_old[m2])
    sum_tot = sum_m1 + sum_m2
    return sum_tot > 0.0001


def modified_fuzzy_cmeans(boundary_points, image, ncentroids=2):
    depth = image.shape[0]
    for px in boundary_points:
        if 3 < px[0] < depth-4:
            data = get_neighbors(px, image)
            data_old = data.copy()
            centroids = generate_centroids(ncentroids, np.mean(data[yx]), np.std(data[yx]))
            print centroids
            while evaluate_condition(data, data_old):
                data_old = data.copy()
                data = compute_membership(data, centroids)
                centroids = recalc_centroids(centroids, data)
                print centroids
            # print centroids
            # print data

            means = np.empty(2)
            stds = np.empty(2)
            for cdx, c in enumerate(centroids):
                cdx += 4
                get = m1
                if cdx == 5:
                    get = m2
                u = data[get]
                coef = 1 / sum(u)
                ys = data[yx]
                means[cdx-4] = coef * sum(np.multiply(u, ys))
                stds[cdx-4] = coef * sum(np.multiply(u, (ys-means[cdx - 4])**2))
            print means, stds
            break

boundaries = pickle.load(open('boundaries.txt'))
img = pickle.load(open('emphasized_imgs.txt'))
# modified_fuzzy_cmeans(boundaries, img)

