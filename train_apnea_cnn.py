import os
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


# -------------------------------------------------------
# Load one record (ECG + Apnea labels)
# -------------------------------------------------------
def load_record(base_dir, record_name):
    """Loads ECG signal and apnea labels for a given record."""
    record_path = os.path.join(base_dir, record_name)
    rec = wfdb.rdrecord(record_path)
    fs = int(rec.fs)
    ecg = rec.p_signal[:, 0].astype(np.float32)

    # ---------- Read apnea labels from WFDB annotation ----------
    try:
        ann = wfdb.rdann(record_path, 'apn')
        symbols = ann.symbol
        labels = [1 if s == 'A' else 0 for s in symbols]
        print(f"[INFO] {record_name}: A={labels.count(1)}  N={labels.count(0)}")
    except Exception as e:
        print(f"[WARN] Could not load annotations for {record_name}: {e}")
        labels = []

    # ---------- Segment ECG into 60-second windows ----------
    win_len = fs * 60
    num_segments = len(labels)
    X, y = [], []
    for i in range(num_segments):
        start = i * win_len
        end = start + win_len
        if end <= len(ecg):
            seg = ecg[start:end]
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-6)
            X.append(seg)
            y.append(labels[i])

    return np.array(X), np.array(y), fs


# -------------------------------------------------------
# Combine multiple records into one dataset
# -------------------------------------------------------
def build_dataset(base_dir, records):
    X_all, y_all = [], []
    fs = None
    for rec in records:
        print(f"[INFO] Loading {rec}...")
        X, y, fs_curr = load_record(base_dir, rec)
        if fs is None:
            fs = fs_curr
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if len(X_all) == 0:
        raise SystemExit("[ERROR] No valid data found. Check record names or dataset path.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    X = X[..., np.newaxis]
    print(f"[DATA] Shape: {X.shape}, Labels count: {np.bincount(y.astype(int))}")
    return X, y, fs


# -------------------------------------------------------
# CNN Model Architecture
# -------------------------------------------------------
def make_model(input_len):
    model = models.Sequential([
        layers.Input(shape=(input_len, 1)),
        layers.Conv1D(32, 7, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 7, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 7, activation='relu'),
        layers.MaxPooling1D(2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# -------------------------------------------------------
# Main Training Routine
# -------------------------------------------------------
def main():
    base_dir = "apnea-ecg-database-1.0.0"
    records = ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10"]
    epochs = 20
    batch_size = 32

    X, y, fs = build_dataset(base_dir, records)

    # ---------- Check label balance ----------
    unique, counts = np.unique(y, return_counts=True)
    label_dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[CHECK] Label distribution: {label_dist}")
    if len(unique) < 2:
        raise SystemExit("[ERROR] Only one class present! Choose records containing both A and N.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Compute class weights
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: weights[i] for i in range(len(weights))}
    print(f"[INFO] Class Weights: {class_weights}")

    # Build & train model
    model = make_model(X.shape[1])
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # ---------- Evaluate ----------
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    acc = np.mean(y_pred == y_test)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = float("nan")
    print(f"[RESULT] Test Accuracy: {acc:.3f}, ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model.save("apnea_model.h5")
    print("[SAVE] Model saved as apnea_model.h5")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()