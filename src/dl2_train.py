import numpy as np
import json
import mlflow
import os
from math import ceil
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, precision_recall_curve
from anomaly_data_generator import AnomalyDataProcessor
from tqdm import tqdm

# ----------------------------
# Modeller
# ----------------------------
def create_cnn(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

def create_rnn(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    x = layers.Bidirectional(layers.LSTM(32))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Dense(ff_dim, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    return x + res

def create_transformer(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.2)
    x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=64, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

def build_model(algorithm, input_shape):
    alg = (algorithm or "cnn").lower()
    if alg in ("rnn", "lstm"):
        return create_rnn(input_shape)
    if alg == "transformer":
        return create_transformer(input_shape)
    return create_cnn(input_shape)

# ----------------------------
# MLflow yardımcıları
# ----------------------------
def get_or_create_experiment(experiment_name):
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment_name)
        print(f"Experiment '{experiment_name}' created with ID: {exp_id}")
    else:
        exp_id = exp.experiment_id
    return exp_id

def save_results(model, y_true, y_pred, model_name="model"):
    os.makedirs("../model_results", exist_ok=True)

    # model
    model_path = f"../model_results/{model_name}.h5"
    model.save(model_path)
    mlflow.log_artifact(model_path)

    # metrikler
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["safe", "dnstunnel"], output_dict=True)

    results = {
        "precision": {"safe": round(float(precision[0]), 3), "dnstunnel": round(float(precision[1]), 3)},
        "recall": {"safe": round(float(recall[0]), 3), "dnstunnel": round(float(recall[1]), 3)},
        "cm": cm.tolist(),
        "report": report
    }

    with open("../model_results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # mlflow metrik
    mlflow.log_metric("safe_precision", results["precision"]["safe"])
    mlflow.log_metric("dnstunnel_precison", results["precision"]["dnstunnel"])
    mlflow.log_metric("safe_recall", results["recall"]["safe"])
    mlflow.log_metric("dnstunnel_recall", results["recall"]["dnstunnel"])

    # tüm çıktı klasörünü yükle
    mlflow.log_artifacts("../model_results")

# ----------------------------
# Eğitim (generator ile)
# ----------------------------
def main_train(params):
    # mlflow
    experiment_id = get_or_create_experiment(params["experiment"])
    mlflow.start_run(experiment_id=experiment_id, nested=True)
    mlflow.set_tag("mlflow.note.content", params["experiment_name"])
    for k, v in params.items():
        mlflow.log_param(k, str(v) if isinstance(v, (dict, list)) else v)

    # feature listesi
    feature_names = [line.strip() for line in open("../input/features.txt") if line.strip() != "label"]
    input_shape = (len(feature_names), 1)

    # generator'lar
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

    # model
    model = build_model(params.get("algorithm", "cnn"), input_shape)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(curve='PR', name='auc_pr')]
    )

    # --- sınıf sayımı (RAM'e almadan) + steps hesapla
    train_batch_size = params["train_batch"]
    total_train, count0, count1, num_batches = 0, 0, 0, 0
    """gen_train.idx = 0
    for _, yb in tqdm(gen_train, desc="Counting train labels"):
        yv = yb.values
        total_train += yv.shape[0]
        count0 += int((yv == 0).sum())
        count1 += int((yv == 1).sum())
        num_batches += 1

    # tekrar başa sar
    gen_train.idx = 0

    # class_weight (güvenli)
    c0 = max(count0, 1)
    c1 = max(count1, 1)
    total = c0 + c1
    class_weight = {
        0: total / (2.0 * c0),
        1: total / (2.0 * c1)
    }

    print(f"Label counts -> safe: {count0}, dnstunnel: {count1}")
    print("Class weights:", class_weight)
    mlflow.log_param("class_weight_0", class_weight[0])
    mlflow.log_param("class_weight_1", class_weight[1])
    """
    steps_per_epoch = ceil(total_train / train_batch_size)
    print("steps_per_epoch:", steps_per_epoch)

    # Keras'a uygun sonsuz generator
    def keras_train_gen():
        while True:
            gen_train.idx = 0
            for Xb, yb in gen_train:
                X = np.expand_dims(Xb.values, axis=-1)  # (batch, timesteps, 1)
                y = yb.values
                yield X, y

    # eğitim
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='auc_pr', mode='max', patience=2, restore_best_weights=True)
    ]
    model.fit(
        keras_train_gen(),
        steps_per_epoch=steps_per_epoch,
        epochs=params.get("epochs", 5),
        #class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    # ----------------------------
    # Test: batch batch tahmin
    # ----------------------------
    y_true, y_prob = [], []
    gen_test.idx = 0
    for Xb, yb in tqdm(gen_test, desc="testing"):
        X = np.expand_dims(Xb.values, axis=-1)
        probs = model.predict(X, verbose=0).ravel()
        y_true.extend(yb.values.tolist())
        y_prob.extend(probs.tolist())

    # threshold optimizasyonu (F1 maks.)
    p, r, th = precision_recall_curve(y_true, y_prob)
    f1s = 2 * p * r / (p + r + 1e-12)
    # precision_recall_curve th uzunluğu (len(p)-1); guard:
    best_idx = int(np.argmax(f1s[:-1])) if len(th) > 0 else 0
    best_th = float(th[best_idx]) if len(th) > 0 else 0.5
    mlflow.log_metric("best_threshold", best_th)
    y_pred = (np.array(y_prob) >= best_th).astype(int)

    # sonuçları kaydet
    model_name = f'{params.get("algorithm","cnn").lower()}_gen_model'
    save_results(model, y_true, y_pred, model_name=model_name)

    mlflow.end_run()

# ----------------------------
# main
# ----------------------------
def main():
    params_multi = [{
        "features": None,
        "experiment": "tunnel",
        "experiment_name": "keras_transformer_gen_run1",
        "algorithm": "transformer",   # "cnn" | "rnn" | "transformer"
        "train_read_line": None,
        "test_read_line": None,
        "train_batch": 100000,
        "test_batch": 100000,
        "epochs": 1,
        "file_train": "../dataset/train.csv",
        "file_test": "../dataset/test.csv"
    }]
    for params in params_multi:
        print(params["experiment_name"])
        main_train(params)

if __name__ == "__main__":
    main()