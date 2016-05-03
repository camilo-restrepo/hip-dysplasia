import numpy as np
from scipy.spatial.distance import euclidean, cdist
import pickle
import utils
import matplotlib.pyplot as plt


idx = 'idx'
yx = 'yx'
d1 = 'd1'
d2 = 'd2'
m1 = 'm1'
m2 = 'm2'


def generate_centroids(ncentroids, mean, std):
    centroids = np.empty(2)
    for i in range(0, ncentroids):
        centroids[i] = np.random.normal(mean, std)
    return centroids


def belonging_to_class(voxel, centroid, centroids, p=2):
    sum = 0
    for c in centroids:
        sum += (abs(voxel - centroid)/abs(voxel - c))**(2/(p - 1))
    result = 1 / sum
    return result


def calc_distance(voxel, centroid, centroids):
    u = belonging_to_class(voxel, centroid, centroids)
    return u*euclidean(voxel, centroid)


def compute_membership(centroids, voxel):
    min_distance = 100000
    cluster = 0
    for idx, val in enumerate(centroids):
        distance = calc_distance(voxel, val, centroids)
        if distance < min_distance:
            min_distance = distance
            cluster = idx
    return cluster


def fuzzy_cmeans(boundary_points, image, ncentroids=2):
    point = ()
    for p in boundary_points:
        if p[0] > 11:
            point = p
            break

    z = point[0]
    x = point[1]
    y = point[2]

    x_ini = x - 11
    x_end = x + 12
    y_ini = y - 11
    y_end = y + 12
    z_ini = z - 11
    z_end = z + 12

    cube = image[z_ini:z_end, x_ini:x_end, y_ini:y_end]
    data = cube.flatten()

    centroids = generate_centroids(ncentroids, np.mean(data), np.std(data))
    clusters = np.zeros_like(data)
    centroids_old = centroids.copy()
    error = 1
    while abs(error) > 0.001:
        # Calc cluster de cada pixel
        for idx, val in enumerate(data):
            clusters[idx] = compute_membership(centroids, val)

        # Recalc centroides
        for idx, val in enumerate(centroids):
            pxs = []
            for jdx, cval in enumerate(clusters):
                if cval == idx:
                    pxs.append(data[jdx])

            sum = np.sum(pxs)
            if len(pxs) != 0:
                centroids_old[idx] = centroids[idx]
                centroids[idx] = (sum/len(pxs))

        error = np.sum(centroids_old-centroids)
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


def calc_neighbors_distance(vecinos_x, c):
    sum = 0
    for v in vecinos_x:
        sum += euclidean(v, c)
    return sum


def precalc_distances(data, centroids):
    centroids = np.reshape(centroids, (2, 1))
    for i in range(0, data.shape[0]):
        px_value = np.reshape(data[i][yx], (1, 1))
        distances = cdist(px_value, centroids)
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


def compute_membership2(data, centroids):
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
        sum_u = (1 + alpha) * sum(data[get]**p)
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

            while evaluate_condition(data, data_old):
                # Calc membership
                data_old = data.copy()
                data = compute_membership2(data, centroids)

                # Recalc centroides
                centroids = recalc_centroids(centroids, data)

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


alpha = 0.85
p = 2

boundaries = pickle.load(open('boundaries.txt'))
img = pickle.load(open('emphasized_imgs.txt'))

modified_fuzzy_cmeans(boundaries, img)

