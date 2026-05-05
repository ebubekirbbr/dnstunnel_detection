import time

import numpy as np
import json
import mlflow
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score
from anomaly_data_generator import AnomalyDataProcessor
from tqdm import tqdm

def create_rnn(input_shape):
    # BiLSTM tabanlı RNN
    inputs = keras.Input(shape=input_shape)  # (timesteps, features=1)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# CNN modeli
def create_cnn(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# MLflow experiment oluşturma
def get_or_create_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
    return experiment_id

# Sonuçları kaydetme
def save_results(model, y_true, y_pred, model_name="cnn_model"):
    os.makedirs("../model_results", exist_ok=True)

    # Modeli kaydet
    model_path = f"../model_results/{model_name}.h5"
    model.save(model_path)
    mlflow.log_artifact(model_path)

    # Metrikler
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    results = {
        "precision": {"safe": round(precision[0], 3), "dnstunnel": round(precision[1], 3)},
        "recall": {"safe": round(recall[0], 3), "dnstunnel": round(recall[1], 3)},
        "cm": cm.tolist(),
        "report": classification_report(y_true, y_pred, target_names=["safe", "dnstunnel"], output_dict=True)
    }

    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]

    with open("../model_results/cnn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("../model_results/y_true.json", "w") as f:
        json.dump(y_true, f, indent=2)

    with open("../model_results/y_pred.json", "w") as f:
        json.dump(y_pred, f, indent=2)


    # MLflow metrik logları
    mlflow.log_metric("safe_precision", results["precision"]["safe"])
    mlflow.log_metric("dnstunnel_precison", results["precision"]["dnstunnel"])
    mlflow.log_metric("safe_recall", results["recall"]["safe"])
    mlflow.log_metric("dnstunnel_recall", results["recall"]["dnstunnel"])

    mlflow.log_artifacts("../model_results")


def main_train(params):
    experiment_id = get_or_create_experiment(params["experiment"])
    mlflow.start_run(experiment_id=experiment_id, nested=True)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])

    # Parametreleri logla
    for k, v in params.items():
        mlflow.log_param(k, str(v) if isinstance(v, (dict, list)) else v)

    # Feature listesi
    feature_names = [line.strip() for line in open("../input/features.txt") if line.strip() != "label"]

    # Generatorlar
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

    # CNN oluştur
    input_shape = (len(feature_names), 1)
    #model = create_cnn(input_shape)
    model = create_cnn(input_shape)

    # Eğitim döngüsü
    epochs = params.get("epochs", 5)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        gen_train.idx = 0
        for X_batch, y_batch in tqdm(gen_train, desc=f"training epoch {epoch+1}"):
            X_batch = np.expand_dims(X_batch.values, axis=-1)
            y_batch = y_batch.values
            loss, acc = model.train_on_batch(X_batch, y_batch)
        mlflow.log_metric(f"epoch_{epoch+1}_loss", loss)
        mlflow.log_metric(f"epoch_{epoch+1}_acc", acc)

    # Test
    y_true, y_pred = [], []
    gen_test.idx = 0
    for X_batch, y_batch in tqdm(gen_test, desc="testing"):
        X_batch = np.expand_dims(X_batch.values, axis=-1)
        preds = (model.predict(X_batch) > 0.5).astype(int)
        y_true.extend(y_batch.values)
        y_pred.extend(preds.flatten())

    save_results(model, y_true, y_pred)
    mlflow.end_run()

def main():
    params_multi = [{
        "features": None,
        "experiment": "tunnel",
        "experiment_name": "keras_cnn_run1",
        "algorithm": "cnn",
        "train_read_line": None,
        "test_read_line": None,
        "train_batch": 10000,
        "test_batch": 10000,
        "epochs": 1,
        "file_train": "../dataset/train.csv",
        "file_test": "../dataset/test.csv"
    }]
    for params in params_multi:
        print(params["experiment_name"])
        main_train(params)

if __name__ == "__main__":
    main()