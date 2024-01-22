from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime

import torch.cuda
from torch import Tensor
from torchvision import transforms
import os
from PIL import Image

from iris_recognition.tools.fs_tools import FsTools
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
        try:
            input_file_path = os.path.join(self.destination_dir, input_filename)
            self.logger.info(f"Started preprocessing image {input_file_path}")
            predictor_path = os.path.join(self.path_organizer.get_root(), "preprocessing", "predict_one_img.py")
            process_call = [
                "python", predictor_path,
                "--gpu", "0" if torch.cuda.is_available() else "None",
                "--img_path", input_file_path,
                "--save_dir", self.destination_dir,
                "--model", "M1"
            ]
            self.logger.info(f"Calling process: {process_call}")
            subprocess.run(process_call)
            self.logger.info(f"Done predicting boundaries")
            self.logger.info(f"Starting segmentation...")
            get_iris(input_filename, self.destination_dir)
            self.logger.info(f"Done segmentation.")
            self.logger.info(f"Starting normalize...")
            normalize(input_filename, self.destination_dir, 56, 112)
            self.logger.info(f"Done normalize.")

            normalized_image_path = os.path.join(self.destination_dir, f'{input_filename[:-4]}_normalized.png')
            self.logger.info(f"Done normalizing. Saved to {normalized_image_path}")
            normalized_img = Image.open(normalized_image_path)
        except Exception as e:
            self.logger.error(f"Preprocessing failed. Error that occurred: {e}")
            raise e
        finally:
            self.logger.info(f"Removing temporary image files.")
            FsTools.rm_file(os.path.join(self.destination_dir, input_filename))
            FsTools.rm_file(os.path.join(self.destination_dir, f'{input_filename[:-4]}_inner_boundary.png'))
            FsTools.rm_file(os.path.join(self.destination_dir, f'{input_filename[:-4]}_outer_boundary.png'))
            FsTools.rm_file(os.path.join(self.destination_dir, f'{input_filename[:-4]}_seg.png'))
            FsTools.rm_file(os.path.join(self.destination_dir, f'{input_filename[:-4]}_normalized.png'))
        return normalized_img

    def preprocess_image(self, image: Image) -> Image:
        filename = hashlib.md5(str(datetime.now()).encode()).hexdigest() + '.png'
        FsTools.mkdir(self.destination_dir)
        image.save(os.path.join(self.destination_dir, filename))
        return self.preprocess_file(filename)
