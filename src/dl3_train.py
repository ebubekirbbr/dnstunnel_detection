import os
import json
import numpy as np
import mlflow
from tensorflow import keras
from anomaly_data_generator import AnomalyDataProcessor
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report

def create_dnn(input_shape):
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Flatten()(inp)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inp, out)


def get_or_create_experiment(experiment_name):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def train_on_batch_dnn(file_train, feature_file, batch_size, epochs, class_weight):
    feature_names = [line.strip() for line in open(feature_file) if line.strip() != "label"]
    input_shape = (len(feature_names), 1)

    gen_train = AnomalyDataProcessor(
        file_train,
        read_line=None,
        features=feature_names,
        batch_size=batch_size
    )

    model = create_dnn(input_shape)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    w0, w1 = float(class_weight[0]), float(class_weight[1])
    steps_per_epoch = int(np.ceil(len(gen_train) / batch_size)) if hasattr(gen_train, "__len__") else None

    for epoch in range(1, epochs + 1):
        gen_train.idx = 0
        pbar = tqdm(gen_train, total=steps_per_epoch, desc=f"Epoch {epoch}/{epochs}")
        epoch_loss, epoch_acc, n = 0.0, 0.0, 0
        for Xb, yb in pbar:
            X = np.expand_dims(Xb.values, -1).astype("float32")
            y = yb.values.astype("float32")
            sw = np.where(y == 1, w1, w0).astype("float32")
            loss, acc = model.train_on_batch(X, y, sample_weight=sw)
            epoch_loss += loss; epoch_acc += acc; n += 1
            pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
        if n > 0:
            mlflow.log_metric("train_loss", epoch_loss / n, step=epoch)
            mlflow.log_metric("train_accuracy", epoch_acc / n, step=epoch)

    return model


def evaluate(model, file_test, feature_file, batch_size):
    feature_names = [line.strip() for line in open(feature_file) if line.strip() != "label"]

    gen_test = AnomalyDataProcessor(
        file_test,
        read_line=None,
        features=feature_names,
        batch_size=batch_size
    )

    y_true_all, y_pred_all = [], []
    gen_test.idx = 0
    for Xb, yb in tqdm(gen_test, desc="test"):
        X = np.expand_dims(Xb.values, -1).astype("float32")
        probs = model.predict(X, verbose=0).ravel()
        preds = (probs > 0.5).astype(int)
        y_true_all.extend(yb.values.tolist())
        y_pred_all.extend(preds.tolist())

    prec = precision_score(y_true_all, y_pred_all, zero_division=0, average=None)
    rec = recall_score(y_true_all, y_pred_all, zero_division=0, average=None)
    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(y_true_all, y_pred_all, target_names=["safe", "dnstunnel"], output_dict=True)
    print(f"precision: {prec}")
    print(f"recall: {rec}")
    return {
        "precision": {"safe": float(prec[0]), "dnstunnel": float(prec[1])},
        "recall": {"safe": float(rec[0]), "dnstunnel": float(rec[1])},
        "cm": cm.tolist(),
        "report": report,
    }


def main():
    params = {
        "experiment": "tunnel",
        "experiment_name": "keras_dnn_run1",
        "algorithm": "dnn",
        "file_train": "../dataset/train.csv",
        "file_test": "../dataset/test.csv",
        "feature_file": "../input/features.txt",
        "train_batch": 1000000,
        "test_batch": 100000,
        "epochs": 10,
        "class_weight": {0: 0.5, 1: 600.0},
    }

    experiment_id = get_or_create_experiment(params["experiment"])
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])
    for k, v in params.items():
        mlflow.log_param(k, str(v) if isinstance(v, (dict, list)) else v)
    mlflow.log_param("class_weight_0", params["class_weight"][0])
    mlflow.log_param("class_weight_1", params["class_weight"][1])

    model = train_on_batch_dnn(
        file_train=params["file_train"],
        feature_file=params["feature_file"],
        batch_size=params["train_batch"],
        epochs=params["epochs"],
        class_weight=params["class_weight"],
    )

    results = evaluate(model, params["file_test"], params["feature_file"], params["test_batch"])

    os.makedirs("../model_results", exist_ok=True)
    model_path = "../model_results/dnn.keras"
    model.save(model_path)
    mlflow.log_artifact(model_path)

    results_path = "../model_results/dnn_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    mlflow.log_artifact(results_path)

    mlflow.log_metric("safe_precision", results["precision"]["safe"])
    mlflow.log_metric("dnstunnel_precison", results["precision"]["dnstunnel"])
    mlflow.log_metric("safe_recall", results["recall"]["safe"])
    mlflow.log_metric("dnstunnel_recall", results["recall"]["dnstunnel"])

    mlflow.end_run()


if __name__ == "__main__":
    main()
