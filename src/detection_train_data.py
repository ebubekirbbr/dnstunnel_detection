import json
import subprocess

import mlflow
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt



def evaluate_model(model, X, y, feature_names):
    y_pred = model.predict(X)
    #y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:\n", cm)

    # Precision
    precision = precision_score(y, y_pred, average=None)
    print("Precision:", precision)

    # Recall
    recall = recall_score(y, y_pred, average=None)
    print("Recall:", recall)

    # Metrikler
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


def save_results(model, feature_names, X_test, y_test):
    model_path = f'../model_results/model.bin'

    model.save_model(model_path)

    #results = evaluate_generator(model, test_gen, feature_names)
    results = evaluate_model(model, X_test, y_test, feature_names=feature_names)

    open(f"../model_results/results.json", "w").write(json.dumps(results, indent=2))

    plot_feature_importance(feature_names, model.feature_importances_)


def main_train(params):
    feature_names = [line.strip() for line in open(f"../input/features.txt", "r").readlines()]
    feature_names = np.array(feature_names)

    if params["features"]:
        feature_names = feature_names[params["features"]]

    df_train = pd.read_csv(params["file_train"])
    y_train = df_train['label'].map({
        'safe': 0,
        'dnstunnel': 1
    })
    X_train = df_train[feature_names]  # label haricindeki tüm sütunlar X

    df_test = pd.read_csv(params["file_test"])
    y_test = df_test['label'].map({
        'safe': 0,
        'dnstunnel': 1
    })

    X_test = df_test[feature_names]  # label haricindeki tüm sütunlar X

    mlflow.log_param(f"X_train.shape", X_train.shape)
    mlflow.log_param(f"X_test.shape", X_test.shape)
    mlflow.log_param(f"Y_train.shape", y_train.shape)
    mlflow.log_param(f"Y_test.shape", y_test.shape)

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=params["xgb_params"]["verbose"]
    )

    save_results(model, feature_names, X_test, y_test)


def main():

    params_multi = [

    {

        #"features": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
        "features": [0, 1, 2, 3, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,42,43],
        #"features": [0, 1, 2, 3, 8, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        "experiment_name": "init",
        "train_read_line": None,
        "test_read_line": None,
        "train_batch": 10000,
        "test_batch": 10000,
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

