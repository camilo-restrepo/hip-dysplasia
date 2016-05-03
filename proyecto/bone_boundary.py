

def pixel_belongs_to_boundary(img_array, x, y, z):
    pixel = img_array[z, x, y]
    if pixel != 0:
        neighbors = [
            # img_array[z, x - 1, y - 1],
            img_array[z, x - 1, y],
            # img_array[z, x - 1, y + 1],
            img_array[z, x, y - 1],
            img_array[z, x, y + 1],
            # img_array[z, x + 1, y - 1],
            img_array[z, x + 1, y],
            # img_array[z, x + 1, y + 1],
            img_array[z - 1, x, y],
            img_array[z + 1, x, y]
        ]

        for n in neighbors:
            if n == 0:
                return True
    return False


def compute_boundary(image):
    width = image.shape[1]
    height = image.shape[2]
    depth = image.shape[0]

    e_b = set()
    for z in range(0, depth):
        for x in range(0, height):
            for y in range(0, width):
                if 0 < x < height-1 and 0 < y < width-1 and 0 < z < depth-1:
                    if pixel_belongs_to_boundary(image, x, y, z):
                        e_b.add((z, x, y))
    return e_b
