import numpy as np
from scipy.spatial.distance import euclidean
import pickle
import utils
import matplotlib.pyplot as plt


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


def calc_neighbors_distance(vecinos_x, c):
    sum = 0
    for v in vecinos_x:
        sum += euclidean(v, c)
    return sum


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

    vecinos_x = image[z_ini:z_end, x_ini:x_end, y_ini:y_end].flatten()
    return vecinos_x

def compute_membership2(y_x, vecinos_x, centroids, alpha=0.85, p=2):
    coef = alpha/len(vecinos_x)
    results = []
    for c in centroids:
        euclidean_distance = euclidean(y_x, c)
        neighbors_distance = calc_neighbors_distance(vecinos_x, c)
        dividendo = ((euclidean_distance + (coef * neighbors_distance))**(-1/(p-1)))
        results.append(dividendo)

    divisor = sum(results)
    for r in range(0, len(results)):
        results[r] = results[r] / divisor
    return results


def modified_fuzzy_cmeans(boundary_points, image, ncentroids=2):
    depth = image.shape[0]
    for p in boundary_points:
        if 3 < p[0] < depth-4:
            vecinos_x = get_neighbors(p, image)
            centroids = generate_centroids(ncentroids, np.mean(vecinos_x), np.std(vecinos_x))

            # Calc membership
            y_x = image[p[0], p[1], p[2]]
            membership = []
            if len(vecinos_x) == 605:
                membership = compute_membership2(y_x, vecinos_x, centroids)
                print y_x, membership, centroids
            else:
                print p

            print p

            # utils.np_show(image[z, :, :])
            # plt.show()
            # break

            # Recalc centroides








boundaries = pickle.load(open('boundaries.txt'))
img = pickle.load(open('emphasized_imgs.txt'))

modified_fuzzy_cmeans(boundaries, img)

