import logging
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def read_config(file_path="config.yaml"):

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_data(split=True):
    """
    if split is True
    retrun X_train, X_test, y_train, y_test
    if split is False
    retrun  df
    """
    iris = load_iris()

    if split:
        X = iris.data
        y = iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

    else:
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

        return df


# Logger setup
class Logger:
    def __init__(
        self, name="app_logger", log_file="app/logs/app.log", level=logging.INFO
    ):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(self._formatter())
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self._formatter())
            self.logger.addHandler(console_handler)

    def _formatter(self):
        return logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def get_logger(self):
        return self.logger
