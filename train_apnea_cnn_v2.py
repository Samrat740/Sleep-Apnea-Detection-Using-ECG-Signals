import os
import numpy as np
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def load_record(base_dir, record_name):
    """
    Load ECG record and apnea annotations using wfdb.
    """
    ann_path = os.path.join(base_dir, record_name)
    
    try:
        # Load annotation properly
        ann = wfdb.rdann(ann_path, 'apn')
        labels = [1 if sym == 'A' else 0 for sym in ann.symbol]
    except Exception as e:
        print(f"[WARN] Could not read annotations for {record_name}: {e}")
        return np.array([]), np.array([]), None

    # Load ECG signal
    try:
        record = wfdb.rdrecord(ann_path)
        signal = record.p_signal[:, 0]
        fs = record.fs
    except Exception as e:
        print(f"[WARN] Could not read ECG for {record_name}: {e}")
        return np.array([]), np.array([]), None

    # Segment length = 1 minute
    segment_length = fs * 60
    num_segments = len(labels)

    X, y = [], []
    for i in range(num_segments):
        start = int(i * segment_length)
        end = int(start + segment_length)
        if end <= len(signal):
            seg = signal[start:end]
            seg = (seg - np.mean(seg)) / np.std(seg)
            X.append(seg)
            y.append(labels[i])

    print(f"[INFO] {record_name}: A={sum(y)}, N={len(y)-sum(y)}")
    return np.array(X), np.array(y, dtype=int), fs


def build_dataset(base_dir, record_list):
    X_all, y_all = [], []
    for rec in record_list:
        X, y, _ = load_record(base_dir, rec)
        X_all.append(X)
        y_all.append(y)
    
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    print(f"[DATA] Shape: {X_all.shape}, Labels count: {np.bincount(y_all)}")
    return X_all, y_all


def create_improved_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 7, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(256, 3, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    base_dir = "apnea-ecg-database-1.0.0"
    records = ["a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10"]

    X, y = build_dataset(base_dir, records)
    X = np.expand_dims(X, axis=2)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = dict(enumerate(class_weights))
    print("[INFO] Class Weights:", class_weights)

    model = create_improved_cnn((X.shape[1], 1))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint("apnea_cnn_v2_best.h5", save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=32,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    model.save("apnea_cnn_v2.h5")
    print("✅ Improved model saved as apnea_cnn_v2.h5")

if __name__ == "__main__":
    main()