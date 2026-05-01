import json
import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt


def get_or_create_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id

    return experiment_id

def evaluate_model(model, X, y, feature_names):
    y_prob = model.predict(X)
    y_pred = (y_prob >= 0.5).astype(int)
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

    return results, y.tolist(), y_pred.tolist()

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
    results, y_true, y_pred = evaluate_model(model, X_test, y_test, feature_names=feature_names)

    mlflow.log_metric("dnstunnel_precison", results["precision"]["dnstunnel"], 0)
    mlflow.log_metric("dnstunnel_recall", results["recall"]["dnstunnel"], 0)
    mlflow.log_metric("safe_recall", results["recall"]["safe"], 0)
    mlflow.log_metric("safe_precision", results["precision"]["safe"], 0)

    open(f"../model_results/results.json", "w").write(json.dumps(results, indent=2))

    with open(f"../model_results/rf_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"../model_results/y_true.json", "w") as f:
        json.dump(y_true, f)

    with open(f"../model_results/y_pred.json", "w") as f:
        json.dump(y_true, f)

    feature_importance = model.feature_importance()

    plot_feature_importance(feature_names, feature_importance)

    mlflow.log_artifacts("../model_results")


def main_train(params):
    experiment = params["experiment_name"]
    experiment_id = get_or_create_experiment(experiment)

    mlflow.start_run(experiment_id=experiment_id, nested=True)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])

    feature_names = [line.strip() for line in open(f"../input/features.txt", "r").readlines()]
    feature_names.remove("label")
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

    train_dataset = lgb.Dataset(X_train, label=y_train)
    test_dataset = lgb.Dataset(X_test, label=y_test, reference=train_dataset)

    mlflow.log_param(f"X_train.shape", X_train.shape)
    mlflow.log_param(f"X_test.shape", X_test.shape)
    mlflow.log_param(f"Y_train.shape", y_train.shape)
    mlflow.log_param(f"Y_test.shape", y_test.shape)

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,  # daha düşük değer
        'max_depth': 5,    # daha düşük değer
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'min_data_in_leaf': 50,  # overfitting'i önlemek için
        'min_gain_to_split': 0.01,  # minimum gain değeri
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1,  # L2 regularization
    }

    print("\nLightGBM modeli eğitiliyor...")

    # Modeli eğit
    model = lgb.train(
        lgb_params,
        train_dataset,
        num_boost_round=100,
        valid_sets=[test_dataset],
        #callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    save_results(model, feature_names, X_test, y_test)


def main():

    params_multi = [

    {

        #"features": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
        "features": None,
        #"features": [0, 1, 2, 3, 8, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        "experiment_name": "tunnel",
        "algorithm": "lightgbm",
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

