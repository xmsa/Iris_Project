import warnings

import mlflow
from model import model_classes, save_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from utils import Logger, load_data, read_config


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="macro"),
        "recall": recall_score(y_test, preds, average="macro"),
        "f1_score": f1_score(y_test, preds, average="macro"),
        "mcc": matthews_corrcoef(y_test, preds),
    }


def run_models(mode="train"):

    config_path = (
        "app/config/model_config.yaml"
        if mode == "experiment"
        else "app/config/model_best.yaml"
    )

    logger.info(f"{mode} is running...")
    config = read_config(config_path)
    logger.info(f"Config ({config_path}) loaded.")

    logger.info("Loading Dataset...")
    X_train, X_test, y_train, y_test = load_data(split=True)
    logger.info("Dataset loaded.")

    for model_name, param_list in config.items():
        ModelClass = model_classes.get(model_name)
        if not ModelClass:
            logger.warning(f"Model '{model_name}' not found.")
            continue

        for i, params in enumerate(param_list):
            param_str = "_".join([f"{k}={v}" for k, v in params.items()])
            run_name = f"{model_name}_config_{i+1}_{param_str}"

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                try:
                    model = ModelClass(**params)
                    logger.info(f"Creating model: {run_name}")
                    model.fit(X_train, y_train)
                    logger.info("Model trained.")

                    metrics = evaluate_model(model, X_test, y_test)
                    logger.info("Model evaluated.")

                    if mode == "experiment":
                        with mlflow.start_run(run_name=run_name):
                            mlflow.log_param("model", model_name)
                            mlflow.log_params(params)
                            mlflow.log_metrics(metrics)
                            logger.info("Metrics logged to MLflow.")
                    else:
                        save_model(model, model_name)
                        logger.info(f"Model saved: {model_name}")

                    summary = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    logger.info(f"{run_name} results -> {summary}")

                except Exception as e:
                    logger.error(f"[ERROR] {run_name} failed: {e}")
                    if mode == "experiment":
                        mlflow.set_tag("run_status", "failed")
                        mlflow.set_tag("error", str(e))


if __name__ == "__main__":
    logger = Logger().get_logger()

    run_models(mode="experiment")
    run_models(mode="train")
