import os.path
import subprocess
import sys
import zipfile

import gdown
import torch

from iris_recognition.final_solution_config import FINAL_SOLUTION_MODEL_NAME, FINAL_SOLUTION_MODEL_TAG, \
    FINAL_SOLUTION_MODEL_EPOCH
from iris_recognition.tools.fs_tools import FsTools
from iris_recognition.tools.path_organizer import PathOrganizer

path_organizer = PathOrganizer()
SEGMENTATION_MODEL_FILE_ID = '1aKSvel_YCRTeY59DNoFjbaxnghlrpqYM'
EXTRACTION_MODEL_FILE_ID = '1o0yE0yYyN9SwrCqTJZPZ-tvEI9JQ2V8P'


def unzip_file(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def get_yn_input(question: str) -> bool:
    while True:
        user_input = input(f"{question} (T/N): ").upper()
        if user_input == 'T':
            return True
        elif user_input == 'N':
            return False
        else:
            print("Nieprawidłowe wejście. Wprowadź jedną literę T lub N.")


def download_feature_extraction_model() -> None:
    try:
        print("Rozpoczęto pobieranie modelu do ekstrakcji cech.")
        url = f'https://drive.google.com/uc?id={EXTRACTION_MODEL_FILE_ID}'
        model_path = path_organizer.get_finetuned_model_path(FINAL_SOLUTION_MODEL_NAME, FINAL_SOLUTION_MODEL_TAG,
                                                             FINAL_SOLUTION_MODEL_EPOCH)
        print("Rozpakowywanie modelu...")
        FsTools.ensure_dir(model_path)
        gdown.download(url, model_path, quiet=False)
        print("Z powodzeniem pobrano i rozpakowano model do ekstrakcji cech.")
    except Exception:
        print("Pobieranie modelu zakończone niepowodzeniem. \n"
              "W tej sytuacji konieczne jest użycie instrukcji instalacji dołączonej do pracy.")
        sys.exit(1)


def download_extract_segmentation_model() -> None:
    try:
        print("Rozpoczęto pobieranie modelu do segmentacji.")
        url = f'https://drive.google.com/uc?id={SEGMENTATION_MODEL_FILE_ID}'
        zip_file = os.path.join(path_organizer.get_root(), "downloaded_segmentation_model.zip")
        gdown.download(url, zip_file, quiet=False)
        model_path = path_organizer.get_segmentation_model_path()
        FsTools.mkdir(model_path)
        print("Rozpakowywanie modelu...")
        unzip_file(zip_file, model_path)
        FsTools.rm_file(zip_file)
        print("Z powodzeniem pobrano model do segmentacji.")
    except Exception:
        print("Pobieranie modelu zakończone niepowodzeniem. \n"
              "W tej sytuacji konieczne jest użycie instrukcji instalacji dołączonej do pracy.")
        sys.exit(2)


def init_webapp() -> None:
    print("Przygotowywanie serwera aplikacji webowej")
    try:
        process_call = [sys.executable, "irisverify/manage.py", "makemigrations"]
        print(f"Calling process: {process_call}")
        subprocess.run(process_call)
        process_call = [sys.executable, "irisverify/manage.py", "migrate"]
        print(f"Calling process: {process_call}")
        subprocess.run(process_call)
        process_call = [sys.executable, "irisverify/manage.py", "collectstatic"]
        print(f"Calling process: {process_call}")
        subprocess.run(process_call)
        print("Serwer aplikacji webowej został przygotowany. Instalacja przebiegła pomyślnie.")
    except Exception:
        print("Przygotowywanie serwera aplikacji webowej zakończone niepowodzeniem.\n"
              "W tej sytuacji konieczne jest użycie instrukcji instalacji dołączonej do pracy.")
        sys.exit(3)


def run_webapp() -> None:
    print("Uruchamianie serwera aplikacji webowej...")
    try:
        process_call = [sys.executable, "irisverify/manage.py", "runserver"]
        print(f"Calling process: {process_call}")
        subprocess.run(process_call)
    except Exception:
        print("Uruchomienie serwera aplikacji webowej zakończone niepowodzeniem.\n"
              "W tej sytuacji konieczne jest użycie instrukcji instalacji dołączonej do pracy.")
        sys.exit(4)
    print("Zamknięto serwer aplikacji webowej")


def main() -> None:
    if not torch.cuda.is_available():
        print("UWAGA: Oprogramowanie CUDA nie jest dostępne.\n"
              "Czy zainstalowałeś wersję torch zgodnie z sekcją 'Pytorch CUDA' w instrukcji?\n"
              "Brak akceleracji sprzętowej powoduje, że ekstrakcja cech może zająć ~10min.")

    if not os.path.isfile(path_organizer.get_finetuned_model_path(FINAL_SOLUTION_MODEL_NAME, FINAL_SOLUTION_MODEL_TAG,
                                                                  FINAL_SOLUTION_MODEL_EPOCH)):
        download_feature_extraction_model()
    if not os.path.isdir(os.path.join(path_organizer.get_segmentation_model_path(), "M1")):
        download_extract_segmentation_model()
    if not os.path.isfile(os.path.join("irisverify", "db.sqlite3")):
        init_webapp()
    if get_yn_input("Czy chcesz uruchomić serwer aplikacji webowej?"):
        run_webapp()
    else:
        print("Możesz teraz w dowolnej chwili uruchomić serwer aplikacji webowej następującym poleceniem:\n"
              "python irisverify/manage.py runserver")


main()
