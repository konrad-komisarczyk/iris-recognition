import skimage.io as io
import cv2
import numpy as np
import os


def get_iris(file_name, folder_path):
    img_path = os.path.join(folder_path, f"{file_name}.bmp")
    result_path = os.path.join(folder_path, f"{file_name}_seg.png")

    img = io.imread(img_path, as_gray=True).astype(float)

    result = io.imread(result_path, as_gray=True).astype(float)
    result_shape = result.shape
    img = cv2.resize(img, (result_shape[1], result_shape[0]), interpolation=cv2.INTER_NEAREST)

    result = result != 0

    # Create cropped iris image
    iris = np.zeros(img.shape)
    iris[result] = img[result]
    iris = iris[:, ~np.all(iris == 0., axis=0)]
    iris = iris[~np.all(iris == 0., axis=1)]

    # Extend iris to a square
    vertical_dim = np.max(iris.shape)
    horizontal_dim = np.min(iris.shape)
    iris = np.vstack([np.zeros([vertical_dim-horizontal_dim, vertical_dim]), iris])

    iris = ((iris) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(folder_path, f"{file_name}_iris.png"), iris)



