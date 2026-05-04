# Bangla Speech Emotion Recognition Using 2D CNN-Transformer

This project is a Bangla Speech Emotion Recognition (SER) web application built with Flask.  
It uses a hybrid **2D CNN-Transformer** deep learning model to classify Bangla speech into different emotion categories from audio input.

## Project Overview

Speech Emotion Recognition helps identify human emotions from speech signals.  
In this project, Bangla audio is processed using acoustic feature extraction techniques, and a trained deep learning model predicts the corresponding emotion class.

The system uses:

- Mel-spectrogram features
- Delta features
- Delta-delta features
- Hybrid 2D CNN-Transformer architecture
- Flask-based web interface for audio upload and prediction

## Features

- Upload Bangla audio files through a web interface
- Preprocess audio automatically
- Extract three-channel acoustic features
- Predict emotion using a trained `.keras` model
- Supports multiple audio formats
- Simple and lightweight Flask deployment

## Emotion Classes

The model is designed to classify speech into the following emotion classes:

- Angry
- Disgusted
- Fearful
- Happy
- Neutral
- Sad
- Surprised

## Required Files

Make sure the following files are available in the project directory:

```text
bangla_cnn_transformer_emotion.keras
bangla_emotion_labels.json
requirements.txt
app.py
