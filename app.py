import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained CNN model
MODEL_PATH = "apnea_cnn_v2_best.h5"
model = load_model(MODEL_PATH)

st.set_page_config(page_title="Sleep Apnea Detector", page_icon="💤", layout="centered")

st.title("💤 Sleep Apnea Detection from ECG")
st.write("Upload an ECG CSV file to visualize and analyze for potential Sleep Apnea risk.")

uploaded_file = st.file_uploader("📁 Upload ECG CSV", type=["csv"])

# Only show "Analyze" button after file is uploaded
if uploaded_file is not None:
    st.success("✅ File uploaded successfully.")
    analyze = st.button("🔍 Analyze ECG")

    if analyze:
        try:
            # Load ECG data
            data = pd.read_csv(uploaded_file)
            if "ECG" not in data.columns:
                st.error("❌ Invalid file format. Please upload a CSV containing one 'ECG' column.")
            else:
                ecg_signal = data["ECG"].values

                # Normalize ECG before feeding to model
                ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
                ecg_signal = ecg_signal[:6000]  # Use 60 seconds (100 Hz * 60)
                ecg_signal = np.expand_dims(ecg_signal, axis=(0, 2))

                # Predict apnea probability
                with st.spinner("🧠 Analyzing ECG..."):
                    prob = model.predict(ecg_signal)[0][0] * 100

                # 🩺 ECG Visualization — Medical Paper Style
                st.subheader("📈 ECG Waveform (Medical View)")

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(data["ECG"], color="#e63946", linewidth=1.2)

                # ECG-style background grid
                ax.set_facecolor("#fffafa")  # light white-pink background
                ax.set_xticks(np.arange(0, len(data), 50))
                ax.set_yticks(np.arange(int(min(data["ECG"])), int(max(data["ECG"])) + 1, 0.2))
                ax.grid(which='major', color='#ffb3b3', linestyle='-', linewidth=0.6)
                ax.grid(which='minor', color='#ffe6e6', linestyle='--', linewidth=0.4)
                ax.minorticks_on()

                # Hide axes and borders
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlim(0, len(data))
                ax.set_title("Lead II ECG (Sample Segment)", fontsize=14, fontweight="bold", color="#333")
                for spine in ax.spines.values():
                    spine.set_visible(False)

                st.pyplot(fig)

                # 🧠 Prediction Output
                st.subheader("🧠 Model Result")
                st.write(f"**Apnea Probability:** `{prob:.2f}%`")

                if prob > 70:
                    st.error("🚨 Sleep Apnea Detected")
                elif 30 <= prob <= 70:
                    st.warning("⚠️ Likely Apnea Condition")
                else:
                    st.success("✅ Normal ECG Pattern")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")

else:
    st.info("⬆️ Please upload an ECG CSV file to begin.")
