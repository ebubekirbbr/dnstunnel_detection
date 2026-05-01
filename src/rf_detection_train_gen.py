import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import mlflow
from anomaly_data_generator import AnomalyDataProcessor
import joblib


def get_or_create_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id

    return experiment_id


def evaluate_model(model, test_gen, feature_names):
    y_true = []
    y_pred = []

    for X_batch, y_batch in tqdm(test_gen, desc="validation", ascii=True, dynamic_ncols=True):
        y_true += y_batch.values.flatten().tolist()
        y_pred_batch = model.predict(X_batch)
        y_pred += y_pred_batch.tolist()

    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)

    results = {
        "precision": {"safe": round(precision[0], 2), "dnstunnel": round(precision[1], 2)},
        "recall": {"safe": round(recall[0], 2), "dnstunnel": round(recall[1], 2)},
        "cm": cm.tolist()
    }
    return results, y_true, y_pred


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
    plt.savefig(f"../model_results/rf_fi.png")


def save_results(model, feature_names, test_gen):
    joblib.dump(model, '../model_results/rf_model.joblib')

    results, y_true, y_pred = evaluate_model(model, test_gen, feature_names)

    mlflow.log_metric("dnstunnel_precison", results["precision"]["dnstunnel"], 0)
    mlflow.log_metric("dnstunnel_recall", results["recall"]["dnstunnel"], 0)
    mlflow.log_metric("safe_recall", results["recall"]["safe"], 0)
    mlflow.log_metric("safe_precision", results["precision"]["safe"], 0)

    with open(f"../model_results/rf_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(f"../model_results/y_true.json", "w") as f:
        json.dump(y_true, f)

    with open(f"../model_results/y_pred.json", "w") as f:
        json.dump(y_true, f)

    plot_feature_importance(feature_names, model.feature_importances_)

    mlflow.log_artifacts("../model_results")


class BatchRandomForest:
    def __init__(self, n_estimators, max_depth, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.model = None

    def partial_fit(self, X, y, n_trees_per_batch=10):
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=n_trees_per_batch,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs,
                #warm_start=True
            )
        else:
            self.model.n_estimators += n_trees_per_batch

        """elif self.model.n_estimators < self.n_estimators:
            self.model.n_estimators += n_trees_per_batch"""

        self.model.fit(X, y)
        return self

def train_with_generator(rf_params, gen, feature_names):
    # BatchRandomForest sınıfından bir örnek oluştur
    model = BatchRandomForest(
        n_estimators=rf_params.get('n_estimators', 100),
        max_depth=rf_params.get('max_depth', 12),
        n_jobs=rf_params.get('nthread', -1)
    )

    # Her batch için incremental eğitim yap
    for X_batch, y_batch in tqdm(gen, desc="train", ascii=True, dynamic_ncols=True):
        model.partial_fit(
            X=X_batch,
            y=y_batch.values.ravel(),
            n_trees_per_batch=10
        )

        if model.model:  # model oluşturulduysa
            print(f"Mevcut ağaç sayısı: {model.model.n_estimators}/{model.n_estimators}")

    print(f"Eğitim tamamlandı. Toplam ağaç sayısı: {model.model.n_estimators}")
    return model.model


def main_train(params):
    experiment = params["experiment"]
    experiment_id = get_or_create_experiment(experiment)

    mlflow.start_run(experiment_id=experiment_id, nested=True)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])
    for key, value in params.items():
        if isinstance(value, (dict, list)):
            mlflow.log_param(key, str(value))
        else:
            mlflow.log_param(key, value)

    feature_names = [line.strip() for line in open(f"../input/features.txt", "r").readlines()]
    feature_names.remove("label")
    feature_names = np.array(feature_names)

    if params["features"]:
        feature_names = feature_names[params["features"]]

    feature_names = feature_names.tolist()

    mlflow.log_param("features_names", feature_names)
    mlflow.log_param("len_features", len(feature_names))

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
        params["rf_params"],
        gen_train,
        feature_names
    )

    save_results(model, feature_names, gen_test)
    mlflow.end_run()


def main():
    params_multi = [{
        "features": None,
        "experiment": "tunnel",
        "experiment_name": "tr1",
        "algorithm": "rf",
        "train_read_line": None,
        "test_read_line": None,
        "train_batch": 1000000,
        "test_batch": 1000000,
        "dir_dataset": "../dataset/",
        "file_train": "../dataset/train.csv",
        "file_test": "../dataset/test.csv",
        "rf_params": {
            "n_estimators": 100,
            "max_depth": 12,
            "n_jobs": -1
        }
    }]

    for params in params_multi:
        print(params["experiment_name"])
        main_train(params)


if __name__ == "__main__":
    main()