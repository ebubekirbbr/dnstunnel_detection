from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import mlflow
import json
from anomaly_data_generator import AnomalyDataProcessor
import joblib


def get_or_create_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created experiment: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def evaluate_model(model, test_gen):
    y_true, y_pred = [], []
    for X_batch, y_batch in tqdm(test_gen, desc="validation"):
        y_true += y_batch.values.tolist()
        y_pred += model.predict(X_batch.values).tolist()
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    return {
        "precision": {"safe": round(precision[0], 2), "dnstunnel": round(precision[1], 2)},
        "recall": {"safe": round(recall[0], 2), "dnstunnel": round(recall[1], 2)},
        "cm": cm.tolist()
    }, y_true, y_pred


def save_results(model, test_gen, feature_names):
    model.save_model("../model_results/tabnet_model")

    results, y_true, y_pred = evaluate_model(model, test_gen)

    mlflow.log_metric("dnstunnel_precison", results["precision"]["dnstunnel"])
    mlflow.log_metric("dnstunnel_recall", results["recall"]["dnstunnel"])
    mlflow.log_metric("safe_recall", results["recall"]["safe"])
    mlflow.log_metric("safe_precision", results["precision"]["safe"])

    with open(f"../model_results/tabnet_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"../model_results/y_true.json", "w") as f:
        json.dump(y_true, f)

    with open(f"../model_results/y_pred.json", "w") as f:
        json.dump(y_pred, f)

    mlflow.log_artifacts("../model_results")

def main_train(params):
    experiment_id = get_or_create_experiment(params["experiment"])
    mlflow.start_run(experiment_id=experiment_id, nested=True)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])

    for k, v in params.items():
        mlflow.log_param(k, str(v) if isinstance(v, (dict, list)) else v)

    features = [f.strip() for f in open("../input/features.txt").readlines() if f.strip() != "label"]
    if params["features"]:
        features = [features[i] for i in params["features"]]

    gen_train = AnomalyDataProcessor(
        params["file_train"],
        read_line=params["train_read_line"],
        features=features,
        batch_size=params["train_batch"]
    )

    gen_test = AnomalyDataProcessor(
        params["file_test"],
        read_line=params["test_read_line"],
        features=features,
        batch_size=params["test_batch"]

    )

    model = NGBClassifier(Dist=Bernoulli, verbose=False)
    for X_batch, y_batch in tqdm(gen_train, desc="training"):
        if len(np.unique(y_batch)) < 2:
            continue
        model.fit(X_batch.values, y_batch.values.ravel())

    save_results(model, gen_test)
    mlflow.end_run()


def main():
    params_multi = [{
        "features": None,
        "experiment": "tunnel",
        "experiment_name": "ngboost_run_1",
        "train_read_line": 50000,
        "test_read_line": 50000,
        "train_batch": 1000000,
        "test_batch": 1000000,
        "dir_dataset": "../dataset/",
        "file_train": "../dataset/train.csv",
        "file_test": "../dataset/test.csv"
    }]
    for params in params_multi:
        print(params["experiment_name"])
        main_train(params)


if __name__ == "__main__":
    main()