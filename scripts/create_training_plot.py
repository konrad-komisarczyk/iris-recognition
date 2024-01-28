import json
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from iris_recognition.tools.path_organizer import PathOrganizer

TAG = "mmu_all_best"
MODEL = "AlexNetFromZero"
PLOT_LOSSES: bool = False

INPUT_METRICS_PATH = PathOrganizer().get_finetuned_model_metrics_path(MODEL, TAG)
with open(INPUT_METRICS_PATH, mode="r", encoding="utf-8") as f:
    metrics: list[dict[str, Any]] = [json.loads(line.replace("\'", "\"")) for line in f]
    df_dict: dict[int, list[float]] = {i: [float(metric["train loss"]), float(metric["train acc"]),
                                           float(metric["val loss"]), float(metric["val acc"])
                                           ] for i, metric in enumerate(metrics)}
    df = pd.DataFrame.from_dict(df_dict, orient="index", columns=["train_loss", "train_acc", "val_loss", "val_acc"])
    if PLOT_LOSSES:
        df = df[["train_loss", "val_loss"]]
    else:
        df = df[["train_acc", "val_acc"]]
    df.columns = ["na podzbiorze treningowym", "na podzbiorze walidacyjnym"]
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['font.size'] = 14
    df.plot()
    if PLOT_LOSSES:
        plt.title("Zmiana wartości funkcji straty w czasie")
        plt.xlabel("Wartość funkcji straty")
    else:
        plt.title("Zmiana dokładności modelu w czasie")
        plt.xlabel("Dokładność")
    plt.grid(True)
    plt.xlabel("Numer epoki")
    plot_path = f"{PathOrganizer.get_root()}/learning_curves/{TAG}-{'loss' if PLOT_LOSSES else 'acc'}.png"
    plt.savefig(plot_path)
    plt.show()
    print("Done")




