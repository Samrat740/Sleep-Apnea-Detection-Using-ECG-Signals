
# Sleep Apnea Detection Using ECG Signals

This project implements an AI-powered deep learning system to detect Sleep Apnea using ECG (Electrocardiogram) signals.
It combines a Convolutional Neural Network (CNN)–based model trained on the Apnea-ECG database and a FastAPI backend for serving predictions.
A separate frontend (Sleep Apnea UI) built with React + TypeScript visualizes ECG waveforms and shows apnea detection results.

#### ApneaView - https://apneaview.vercel.app 
#### Frontend Repository - https://github.com/Samrat740/Sleep_Apnea_UI


## Features

✅ CNN-based Sleep Apnea Detection from ECG signals
✅ FastAPI Backend for real-time prediction
✅ ECG Data Generator for creating labeled test samples
✅ Streamlit / React Frontend integration support
✅ Pretrained Model (apnea_cnn_v2_best.h5) for immediate inference


## Model Details

The model is a 1D Convolutional Neural Network trained on the Apnea-ECG Database (PhysioNet).

Dataset - https://physionet.org/content/apnea-ecg/1.0.0/

It takes 60-second ECG segments (6000 samples @ 100 Hz) as input and outputs the probability of apnea presence.

Training Script:
train_apnea_cnn_v2.py

Input: Raw ECG signals

Layers: Conv1D, BatchNorm, MaxPooling, Dense

Optimizer: Adam

Loss: Binary Crossentropy

Output: Probability of Sleep Apnea (0–1)