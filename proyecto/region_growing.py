

def simple_2d_binary_region_growing(image, seed_points):
    visited = set()
    queue = set()

    # for point in seed_points:
        # queue.add(point)

    queue |= seed_points
    height = image.shape[0]
    width = image.shape[1]

    while len(queue) != 0:
        px = queue.pop()
        z = px[0]
        x = px[1]
        y = px[2]

        if image[x, y] == 1:
            visited.add(px)

            if 0 < x - 1 < height and 0 < y - 1 < width:
                if image[x - 1, y - 1] == 1 and (z, x - 1, y - 1) not in visited:
                    queue.add((z, x - 1, y - 1))

            if 0 < x - 1 < height and 0 < y < width:
                if image[x - 1, y] == 1 and (z, x - 1, y) not in visited:
                    queue.add((z, x - 1, y))

            if 0 < x - 1 < height and 0 < y + 1 < width:
                if image[x - 1, y + 1] == 1 and (z, x - 1, y + 1) not in visited:
                    queue.add((z, x - 1, y + 1))

            if 0 < x < height and 0 < y - 1 < width:
                if image[x, y - 1] == 1 and (z, x, y - 1) not in visited:
                    queue.add((z, x, y - 1))

            if 0 < x < height and 0 < y + 1 < width:
                if image[x, y + 1] == 1 and (z, x, y + 1) not in visited:
                    queue.add((z, x, y + 1))

            if 0 < x + 1 < height and 0 < y - 1 < width:
                if image[x + 1, y - 1] == 1 and (z, x + 1, y - 1) not in visited:
                    queue.add((z, x + 1, y - 1))

            if 0 < x + 1 < height and 0 < y < width:
                if image[x + 1, y] == 1 and (z, x + 1, y) not in visited:
                    queue.add((z, x + 1, y))

            if 0 < x + 1 < height and 0 < y + 1 < width:
                if image[x + 1, y + 1] == 1 and (z, x + 1, y + 1) not in visited:
                    queue.add((z, x + 1, y + 1))

    return visited
