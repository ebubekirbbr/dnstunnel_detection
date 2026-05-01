
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import mlflow
from anomaly_data_generator import AnomalyDataProcessor
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import os


def plot_feature_importance(feature_names, importance):
    """
    Özellik önemlilik grafiğini çizer.
    data: model.feature_importances_
    gen: [f_importance.get(f, 0) for f in feature_names]
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    importance = importance.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(importance['feature'], importance['importance'])
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(f"../model_results/tabnet_fi.png")


def get_or_create_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def evaluate_model(model, test_gen):
    y_true = []
    y_pred = []

    for X_batch, y_batch in tqdm(test_gen, desc="validation", ascii=True, dynamic_ncols=True):
        preds = model.predict(X_batch.values)
        y_true += y_batch.values.flatten().tolist()
        y_pred += preds.tolist()

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    results = {
        "precision": {"safe": round(precision[0], 2), "dnstunnel": round(precision[1], 2)},
        "recall": {"safe": round(recall[0], 2), "dnstunnel": round(recall[1], 2)},
        "cm": cm.tolist()
    }
    return results, y_true, y_pred


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

    plot_feature_importance(feature_names, model.feature_importances_)
    mlflow.log_artifacts("../model_results")

def main_train(params):
    experiment_id = get_or_create_experiment(params["experiment"])

    mlflow.start_run(experiment_id=experiment_id, nested=True)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])

    for key, value in params.items():
        mlflow.log_param(key, str(value) if isinstance(value, (dict, list)) else value)

    mlflow.log_param("algorithm", "tabnet")

    feature_names = [line.strip() for line in open(f"../input/features.txt", "r").readlines()]
    feature_names.remove("label")
    if params["features"]:
        feature_names = [feature_names[i] for i in params["features"]]

    mlflow.log_param("features_names", feature_names)
    mlflow.log_param("len_features", len(feature_names))

    gen_train = AnomalyDataProcessor(params["file_train"], read_line=params["train_read_line"],
                                     features=feature_names, batch_size=params["train_batch"])
    gen_test = AnomalyDataProcessor(params["file_test"], read_line=params["test_read_line"],
                                    features=feature_names, batch_size=params["test_batch"])

    mlflow.log_param("train_data", gen_train.row_count)
    mlflow.log_param("test_data", gen_test.row_count)

    #model = TabNetClassifier(verbose=0, seed=42, device_name="cuda" if torch.cuda.is_available() else "cpu")
    model = TabNetClassifier(
        n_d=16,  # was 64 → ↓ kapasite
        n_a=16,  # was 64 → ↓ attention boyutu
        n_steps=3,  # was 7 → ↓ karar adımı sayısı
        gamma=1.3,  # was 1.5 → biraz daha hızlı
        n_independent=1,  # was 2 → daha az blok
        n_shared=1,  # was 2 → paylaşımlı parametre
        optimizer_params=dict(lr=2e-2),  # lr yüksek tutularak hızlı öğrenme sağlanır
        verbose=1,
        seed=42,
        device_name='cuda' if torch.cuda.is_available() else 'cpu'
    )
    for X_batch, y_batch in tqdm(gen_train, desc="training", ascii=True, dynamic_ncols=True):
        if len(np.unique(y_batch)) < 2:
            continue  # skip batch with single class
        model.fit(X_batch.values, y_batch.values.ravel(), max_epochs=1, patience=0)

    save_results(model, gen_test, feature_names)
    mlflow.end_run()

def main():
    params_multi = [{
        "features": None,
        "experiment": "tunnel",
        "experiment_name": "tabnet_run1",
        "algorithm": "tabnet",
        "train_read_line": None,
        "test_read_line": None,
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
