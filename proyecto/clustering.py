import numpy as np
from scipy.spatial.distance import euclidean


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


def nearest_centroid(centroids, voxel):
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
            clusters[idx] = nearest_centroid(centroids, val)

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

