# Bangla CNN + Transformer Flask App

## Required files in this folder
- `bangla_cnn_transformer_emotion.keras`
- `bangla_emotion_labels.json`
- optionally an FFmpeg `bin` folder if FFmpeg is not on PATH

## Run
```bash
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Notes
- The app expects the same preprocessing used during training:
  - sample rate 16000
  - duration 4 seconds
  - mel + delta + delta-delta with 128 mel bins
- Supported uploads: wav, mp3, m4a, ogg, flac, webm
