from PIL import Image
import os
import hashlib as hash
from datetime import datetime
from irisverify.settings import TMP_FILES_DIR, PREPROCESSOR


def verify(iris_image, feature_vector):

    new_feature_vector = extract_feature_vector(iris_image)

    # TODO: matching logic
    # if similarity(new_feature_vector, feature_vector) > threshold: return True
    return True


def extract_feature_vector(iris_image):
    img = Image.open(iris_image)
    filename = hash.md5(str(datetime.now()).encode()).hexdigest() + '.png'
    img.save(os.path.join(TMP_FILES_DIR, filename))

    normalized_img_tensor = PREPROCESSOR.preprocess(filename)

    # TODO: feature extraction logic
    # model(normalized_img_tensor)
    # extract_features(filename)

    for file in os.listdir(TMP_FILES_DIR):
        os.remove(os.path.join(TMP_FILES_DIR, file))

    return [1,2,3,4,5]