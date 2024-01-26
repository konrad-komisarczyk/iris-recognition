import json
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from iris_recognition.tools.path_organizer import PathOrganizer

INPUT_METRICS_PATH = PathOrganizer().get_finetuned_model_metrics_path("AlexNet", "t1")
PLOT_LOSSES = True
OUTPUT_PLOT_PATH = ""

with open(INPUT_METRICS_PATH, mode="r", encoding="utf-8") as f:
    metrics: list[dict[str, Any]] = [json.loads(line.replace("\'", "\"")) for line in f]
    df_dict: dict[int, list[float]] = {i: [float(metric["train loss"]), float(metric["train acc"]),
                                           float(metric["val loss"]), float(metric["val acc"])
                                           ] for i, metric in enumerate(metrics)}
    df = pd.DataFrame.from_dict(df_dict, orient="index", columns=["train_loss", "train_acc", "val_loss", "val_acc"])
    if PLOT_LOSSES:
        df = df[["train_loss", "val_loss"]]
        df.columns = ["Wartość straty na podzbiorze treningowym", "Wartość straty na podzbiorze walidacyjnym"]
    else:
        df = df[["train_acc", "val_acc"]]
        df.columns = ["Dokładność na podzbiorze treningowym", "Dokładność na podzbiorze walidacyjnym"]
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 14
    df.plot()
    plt.title("Tytuł do dopisania")
    plt.grid(True)
    plt.xlabel("Numer epoki")
    plt.show()





