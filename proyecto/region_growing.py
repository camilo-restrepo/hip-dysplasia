from skimage.measure import label, regionprops


def simple_2d_binary_region_growing(image, seed_points, mean_centroid):
    visited = set()
    queue = set()
    queue |= seed_points
    height = image.shape[0]
    width = image.shape[1]

    label_image = label(image)
    for region in regionprops(label_image):
        centroid = region.centroid[1]
        if centroid > mean_centroid:
            for coord in region.coords:
                label_image[coord[0], coord[1]] = 0

    while len(queue) != 0:
        px = queue.pop()
        z = px[0]
        x = px[1]
        y = px[2]

        if image[x, y] == 1:
            visited.add(px)

            # if 0 < x - 1 < height and 0 < y - 1 < width:
            #     if image[x - 1, y - 1] == 1 and (z, x - 1, y - 1) not in visited:
            #         queue.add((z, x - 1, y - 1))

            if 0 < x - 1 < height and 0 < y < width:
                if image[x - 1, y] == 1 and (z, x - 1, y) not in visited:
                    queue.add((z, x - 1, y))

            # if 0 < x - 1 < height and 0 < y + 1 < width:
            #     if image[x - 1, y + 1] == 1 and (z, x - 1, y + 1) not in visited:
            #         queue.add((z, x - 1, y + 1))

            if 0 < x < height and 0 < y - 1 < width:
                if image[x, y - 1] == 1 and (z, x, y - 1) not in visited:
                    queue.add((z, x, y - 1))

            # ----------------------------------------

            for j in range(2, 13):
                if 0 < x < height and 0 < y - j < width:
                    if image[x, y - j] == 1 and (z, x, y - j) not in visited and label_image[x, y - j] != 0:
                        queue.add((z, x, y - j))

            # ----------------------------------------

            if 0 < x < height and 0 < y + 1 < width:
                if image[x, y + 1] == 1 and (z, x, y + 1) not in visited:
                    queue.add((z, x, y + 1))

            # if 0 < x + 1 < height and 0 < y - 1 < width:
            #     if image[x + 1, y - 1] == 1 and (z, x + 1, y - 1) not in visited:
            #         queue.add((z, x + 1, y - 1))

            if 0 < x + 1 < height and 0 < y < width:
                if image[x + 1, y] == 1 and (z, x + 1, y) not in visited:
                    queue.add((z, x + 1, y))

            # if 0 < x + 1 < height and 0 < y + 1 < width:
            #     if image[x + 1, y + 1] == 1 and (z, x + 1, y + 1) not in visited:
            #         queue.add((z, x + 1, y + 1))

    return visited
