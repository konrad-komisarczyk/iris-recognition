from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime

from torch import Tensor
from torchvision import transforms
import os
from PIL import Image

from iris_recognition.tools.logger import get_logger
from iris_recognition.tools.path_organizer import PathOrganizer
from preprocessing.get_iris import get_iris
from preprocessing.normalize import normalize


class Preprocessor:
    """
    Class representing Preprocessor object. Used to preprocess input images
    """

    def __init__(self, prefix: str | None = None) -> None:
        self.path_organizer = PathOrganizer(prefix)
        self.logger = get_logger("Preprocessor")
        self.destination_dir = os.path.join(self.path_organizer.get_root(), "irisverify", "TMP_FILES")

    def preprocess_file(self, input_filename: str) -> Image:
        input_file_path = os.path.join(self.destination_dir, input_filename)

        predictor_path = os.path.join(self.path_organizer.get_root(), "preprocessing", "predict_one_img.py")
        subprocess.run(
            ["python", predictor_path, "--gpu", "0", "--img_path", input_file_path, "--save_dir", self.destination_dir,
             "--model", "M1"])

        get_iris(input_filename, self.destination_dir)
        normalize(input_filename, self.destination_dir, 56, 112)

        normalized_img = Image.open(os.path.join(self.destination_dir, f'{input_filename[:-4]}_normalized.png'))
        return normalized_img

    def preprocess_image(self, image: Image) -> Image:
        filename = hashlib.md5(str(datetime.now()).encode()).hexdigest() + '.png'
        image.save(os.path.join(self.destination_dir, filename))
        return self.preprocess_file(filename)
