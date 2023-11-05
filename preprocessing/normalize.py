import numpy as np
import skimage.io as io
import os


def daugman_normalizaiton(image, height, width, r_in, r_out):  # Daugman归一化，输入为640*480,输出为width*height
    """
    Source: https://github.com/YifengChen94/IrisReco
    """

    if 2*r_out >= image.shape[0] + 4:
        r_out = image.shape[0] // 2
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    # r_out = r_in + r_out

    # Create empty flatten image
    flat = np.zeros((height, width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            try:
                color = image[int(Xc)][int(Yc)]  # color of the pixel
            except IndexError:
                try:
                    color = image[int(Xc)][int(Yc) - 1]  # color of the pixel
                except IndexError:
                    try:
                        color = image[int(Xc) - 1][int(Yc)]  # color of the pixel
                    except IndexError:
                        color = image[int(Xc) - 1][int(Yc) - 1]  # color of the pixel

            flat[j][i] = color
    return flat  # liang


def normalize(file_name, folder_path):
    inner_path = os.path.join(folder_path, f"{file_name}_inner_boundary.png")
    outher_path = os.path.join(folder_path, f"{file_name}_outer_boundary.png")
    iris_path = os.path.join(folder_path, f"{file_name}_iris.png")

    inner_img = io.imread(inner_path, as_gray=True).astype(float)
    outher_img = io.imread(outher_path, as_gray=True).astype(float)

    inner_img = inner_img[:, ~np.all(inner_img == 0., axis=0)]
    inner_img = inner_img[~np.all(inner_img == 0., axis=1)]

    inner_radius = max(inner_img.shape) // 2

    outher_img = outher_img[:, ~np.all(outher_img == 0., axis=0)]
    outher_img = outher_img[~np.all(outher_img == 0., axis=1)]

    outher_radius = max(outher_img.shape) // 2

    iris = io.imread(iris_path, as_gray=True).astype(float)

    result = daugman_normalizaiton(iris, 50, 100, inner_radius, outher_radius)

    io.imsave(os.path.join(folder_path, f'{file_name}_normalized.png'), result)

    del iris, inner_img, outher_img
