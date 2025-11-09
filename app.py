from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import io

# ===============================
# ⚙️ Initialize FastAPI app
# ===============================
app = FastAPI(title="Sleep Apnea Detection API", description="Detects sleep apnea from ECG signals.")

# Allow frontend (React / Firebase) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace with your Firebase domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 🧠 Load Trained CNN Model
# ===============================
MODEL_PATH = "model/apnea_cnn_v2_best.h5"
model = load_model(MODEL_PATH)

# ===============================
# 🩺 Root Endpoint
# ===============================
@app.get("/")
def home():
    return {"message": "Sleep Apnea Detection API is running!"}

# ===============================
# 🔍 ECG Prediction Endpoint
# ===============================
@app.post("/predict")
async def predict_apnea(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        contents = await file.read()
        data = pd.read_csv(io.BytesIO(contents))

        # Check for ECG column
        if "ECG" not in data.columns:
            return {"error": "Invalid CSV format. Must contain 'ECG' column."}

        # Preprocess ECG
        ecg_signal = data["ECG"].values
        ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)
        ecg_signal = ecg_signal[:6000]  # Use 60 seconds (100 Hz * 60)
        ecg_signal = np.expand_dims(ecg_signal, axis=(0, 2))

        # Predict apnea probability
        prob = float(model.predict(ecg_signal)[0][0] * 100)

        # Determine result
        if prob > 70:
            status = "Sleep Apnea Detected"
        elif 30 <= prob <= 70:
            status = "Likely Apnea Condition"
        else:
            status = "Normal ECG Pattern"

        return {
            "probability": round(prob, 2),
            "status": status
        }

    except Exception as e:
        return {"error": f"Error processing file: {str(e)}"}
