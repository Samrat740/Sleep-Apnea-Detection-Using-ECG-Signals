import wfdb
import pandas as pd
import numpy as np
import os

BASE_DIR = "model/apnea-ecg-database-1.0.0"
fs = 100        # Hz
duration = 60   # seconds


def extract_segment(record_name, label_type, output_name):
    print(f"\n[INFO] Extracting {label_type} segment from {record_name}...")

    record_path = os.path.join(BASE_DIR, record_name)

    # ✅ Load ECG signal
    record = wfdb.rdrecord(record_path)
    ecg = record.p_signal[:, 0]

    # ✅ Load apnea annotations (from .apn)
    ann = wfdb.rdann(record_path, "apn")
    labels = [x.strip().upper() for x in ann.symbol if x.strip() in ["A", "N"]]

    print(f"[DEBUG] Found {len(labels)} labels: {set(labels)}")

    if not labels:
        print(f"⚠️ No labels found for {record_name}")
        return

    # Determine minute indices for target label
    if label_type == "A":
        valid_indices = [i for i, l in enumerate(labels) if l == "A"]
    elif label_type == "N":
        valid_indices = [i for i, l in enumerate(labels) if l == "N"]
    elif label_type == "MIX":
        valid_indices = [i for i, l in enumerate(labels) if l in ["A", "N"]]
    else:
        raise ValueError("Invalid label_type")

    print(f"[DEBUG] Found {len(valid_indices)} valid minutes for {label_type}")

    if not valid_indices:
        print(f"⚠️ No {label_type} segments found in {record_name}")
        return

    # ✅ Pick a random 60-second window
    minute_idx = np.random.choice(valid_indices)
    start = minute_idx * fs * duration
    segment = ecg[int(start): int(start + fs * duration)]

    pd.DataFrame({"ECG": segment}).to_csv(output_name, index=False)
    print(f"✅ Saved {output_name} ({len(segment)} samples, label={label_type})")


def main():
    print("🩺 Generating realistic ECG examples from annotations...\n")

    extract_segment("c01", "N", "normal.csv")         # Normal subject
    extract_segment("b01", "MIX", "likely_apnea.csv") # Borderline subject
    extract_segment("a01", "A", "apnea.csv")          # Apnea subject

    print("\n🎉 All ECG CSVs ready! Upload them to your Streamlit app.")


if __name__ == "__main__":
    main()
