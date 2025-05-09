import json
import subprocess

import mlflow
import numpy as np
import xgboost
import xgboost as xgb
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from anomaly_data_generator import AnomalyDataProcessor


def get_or_create_experiment(experiment_name):
    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # If the experiment does not exist, create it
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        # If the experiment exists, return its ID
        experiment_id = experiment.experiment_id

    return experiment_id


def evaluate_model(model, test_gen, feature_names):

    y_true = []
    y_pred = []
    for X_batch, y_batch in tqdm(test_gen, desc="validation", ascii=True, dynamic_ncols=True):
        y_true += y_batch.values.flatten().tolist()

        #y_pred_batch = model.predict(X_batch)
        dtest = xgb.DMatrix(X_batch, feature_names=feature_names)
        y_prob_batch = model.predict(dtest)
        y_pred_batch = (y_prob_batch >= 0.5).astype(int)
        y_pred += y_pred_batch.tolist()

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    results = {
        "precision": {"safe": round(precision[0], 2), "dnstunnel": round(precision[1], 2)},
        "recall": {"safe": round(recall[0], 2), "dnstunnel": round(recall[1], 2)},
        "cm": cm.tolist()
    }

    return results

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
    plt.savefig(f"../model_results/fi.png")


def save_results(model: xgboost.Booster, feature_names, test_gen):
    model_path = f'../model_results/model.bin'

    model.save_model(model_path)

    #results = evaluate_generator(model, test_gen, feature_names)
    results = evaluate_model(model, test_gen, feature_names=feature_names)

    open(f"../model_results/results.json", "w").write(json.dumps(results, indent=2))

    importance = model.get_score(importance_type='weight')
    importance = {feat: importance.get(feat, 0) for feat in feature_names}

    plot_feature_importance(feature_names, list(importance.values()))


def train_with_generator(xgb_params, gen, feature_names):
    xgb_model = None

    xgb_params_inc = {'process_type': 'update', 'updater': 'refresh', 'refresh_leaf': True}

    for X_batch, y_batch in tqdm(gen, desc="train", ascii=True, dynamic_ncols=True):

        dtrain = xgb.DMatrix(X_batch, label=y_batch)
        assert len(dtrain.get_label()) == dtrain.num_row(), "Mismatch in labels and rows"

        if xgb_model:
            xgb_params.update(xgb_params_inc)

        xgb_model = xgb.train(params=xgb_params,
                              dtrain=dtrain,
                              xgb_model=xgb_model,
                              num_boost_round=xgb_params["n_estimators"],
                              verbose_eval=xgb_params["verbose"]
                              )

    return xgb_model


def main_train(params):

    experiment = "xgb_tunnel"

    experiment_id = get_or_create_experiment(experiment)

    mlflow.start_run(experiment_id=experiment_id, nested=True)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])
    for key, value in params.items():
        mlflow.log_param(key, value)

    feature_names = [line.strip() for line in open(f"../input/features.txt", "r").readlines()]
    feature_names = np.array(feature_names)

    if params["features"]:
        feature_names = feature_names[params["features"]]

    feature_names = feature_names.tolist()
    gen_train = AnomalyDataProcessor(
        params["file_train"],
        read_line=params["train_read_line"],
        features=feature_names,
        batch_size=params["train_batch"]
    )

    gen_test = AnomalyDataProcessor(
        params["file_test"],
        read_line=params["test_read_line"],
        features=feature_names,
        batch_size=params["test_batch"]

    )

    mlflow.log_param("train_data", gen_train.row_count)
    mlflow.log_param("test_data", gen_test.row_count)

    model = train_with_generator(
        params["xgb_params"],
        gen_train,
        feature_names
    )

    save_results(model, feature_names, gen_test)
    mlflow.log_artifacts("../model_results")
    mlflow.end_run()


def main():

    params_multi = [

    {

        #"features": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
        "features": [0, 1, 2, 3, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,42,43],
        #"features": [0, 1, 2, 3, 8, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        "experiment_name": "init",
        "train_read_line": None,
        "test_read_line": None,
        "train_batch": 1000000,
        "test_batch": 1000000,
        "dir_dataset": "../dataset/",
        "file_train": "../dataset/train.csv",
        "file_test": "../dataset/test.csv",
        "xgb_params": {
            "verbosity": 0,
            "nthread": -1,
            "tree_method": "hist",
            "max_depth": 12,
            "scale_pos_weight": 600,
            "learning_rate": 0.1,
            "colsample_bytree": 0.8,
            "alpha": 0,
            "lambda": 1,
            "objective": "binary:logistic",
            "n_estimators": 100,
            "eval_metric": "logloss",
            "verbose": False
        }
    }
    ]

    for params in params_multi:
        #subprocess.call("rm -f ../model_results/*", shell=True)
        print(params["experiment_name"])
        main_train(params)


if __name__ == "__main__":
    main()

