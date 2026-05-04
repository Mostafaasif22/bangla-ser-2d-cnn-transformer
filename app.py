import os
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import librosa
import tensorflow as tf
import keras
from flask import Flask, render_template, request, jsonify
from pydub import AudioSegment


# =========================================================
# Keras compatibility patch
# =========================================================
try:
    from keras.layers import Layer as PublicKerasLayer
except Exception:
    PublicKerasLayer = None

try:
    from keras.src.layers.layer import Layer as InternalKerasLayer
except Exception:
    InternalKerasLayer = None


def patch_keras_layer_init():
    patched = set()

    for LayerCls in [PublicKerasLayer, InternalKerasLayer]:
        if LayerCls is None or LayerCls in patched:
            continue

        original_init = LayerCls.__init__

        def _patched_init(self, *args, __original_init=original_init, **kwargs):
            kwargs.pop("quantization_config", None)
            return __original_init(self, *args, **kwargs)

        LayerCls.__init__ = _patched_init
        patched.add(LayerCls)


patch_keras_layer_init()


# =========================================================
# Paths and config
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bangla_cnn_transformer_emotion.keras"
LABEL_PATH = BASE_DIR / "bangla_emotion_labels.json"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_MB = 20
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "ogg", "flac", "webm", "mp4"}

# Must match training
SR = 16000
DURATION = 4
SAMPLES = SR * DURATION
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256


# =========================================================
# FFmpeg auto-detection
# =========================================================
def configure_ffmpeg() -> None:
    candidate_bin_dirs = [
        BASE_DIR / "ffmpeg" / "bin",
        BASE_DIR / "ffmpeg-7.1.1-essentials_build" / "bin",
        BASE_DIR / "ffmpeg-7.1.1-essentials_build" / "ffmpeg-7.1.1-essentials_build" / "bin",
        BASE_DIR / "ffmpeg-7.1.1-essentials_build - 2" / "ffmpeg-7.1.1-essentials_build" / "ffmpeg-7.1.1-essentials_build" / "bin",
        BASE_DIR / "ffmpeg-7.1.1-essentials_build - 2(1)" / "ffmpeg-7.1.1-essentials_build" / "ffmpeg-7.1.1-essentials_build" / "bin",
    ]

    for bin_dir in candidate_bin_dirs:
        ffmpeg_exe = bin_dir / "ffmpeg.exe"
        ffprobe_exe = bin_dir / "ffprobe.exe"

        if ffmpeg_exe.exists() and ffprobe_exe.exists():
            AudioSegment.converter = str(ffmpeg_exe)
            AudioSegment.ffmpeg = str(ffmpeg_exe)
            AudioSegment.ffprobe = str(ffprobe_exe)
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

            print(f"[INFO] Using local FFmpeg from: {bin_dir}")
            print(f"[INFO] ffmpeg.exe: {ffmpeg_exe}")
            print(f"[INFO] ffprobe.exe: {ffprobe_exe}")
            return

    print("[INFO] Local FFmpeg folder not found. Falling back to system PATH.")


# =========================================================
# Custom layers for loading the hybrid model
# =========================================================
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pos_emb = tf.keras.layers.Embedding(input_dim=max_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d_model),
        ])

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.att(x, x)
        attn_output = self.drop1(attn_output, training=training)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        return self.norm2(x + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate,
        })
        return config


# =========================================================
# Load labels and model
# =========================================================
configure_ffmpeg()

if not LABEL_PATH.exists():
    raise FileNotFoundError(f"Missing label file: {LABEL_PATH}")

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    LABEL_INFO = json.load(f)

LABELS = LABEL_INFO["labels"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")


def load_emotion_model():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "PositionalEmbedding": PositionalEmbedding,
                "TransformerEncoder": TransformerEncoder,
            },
            compile=False,
            safe_mode=False,
        )
        print("[INFO] Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {MODEL_PATH}\n"
            f"Reason: {e}"
        ) from e


MODEL = load_emotion_model()


# =========================================================
# Audio preprocessing - must match training
# =========================================================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav(input_path: Path) -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    ext = input_path.suffix.lower()
    print(f"[INFO] Converting file: {input_path} (ext={ext})")

    if ext == ".wav":
        audio = AudioSegment.from_wav(str(input_path))
    elif ext == ".webm":
        audio = AudioSegment.from_file(str(input_path), format="webm")
    elif ext == ".mp4":
        audio = AudioSegment.from_file(str(input_path), format="mp4")
    elif ext == ".m4a":
        audio = AudioSegment.from_file(str(input_path), format="m4a")
    elif ext == ".mp3":
        audio = AudioSegment.from_file(str(input_path), format="mp3")
    elif ext == ".ogg":
        audio = AudioSegment.from_file(str(input_path), format="ogg")
    elif ext == ".flac":
        audio = AudioSegment.from_file(str(input_path), format="flac")
    else:
        audio = AudioSegment.from_file(str(input_path))

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(SR)

    wav_path = input_path.with_suffix(".wav")
    audio.export(str(wav_path), format="wav")

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV export failed: {wav_path}")

    print(f"[INFO] WAV created: {wav_path}")
    return wav_path


def load_audio(file_path: Path, sr: int = SR, target_len: int = SAMPLES) -> np.ndarray:
    y, _ = librosa.load(str(file_path), sr=sr)

    if len(y) == 0:
        return np.zeros(target_len, dtype=np.float32)

    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y.astype(np.float32)


def audio_to_feature3(y: np.ndarray, sr: int = SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    delta1 = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    delta1 = (delta1 - delta1.min()) / (delta1.max() - delta1.min() + 1e-8)
    delta2 = (delta2 - delta2.min()) / (delta2.max() - delta2.min() + 1e-8)

    feat = np.stack([mel_db, delta1, delta2], axis=-1)
    return feat.astype(np.float32)


def predict_emotion(file_path: Path):
    y = load_audio(file_path)
    feat = audio_to_feature3(y)
    feat = np.expand_dims(feat, axis=0)

    probs = MODEL.predict(feat, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs


# =========================================================
# Flask app
# =========================================================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]

    if not audio_file or audio_file.filename == "":
        return jsonify({"error": "Empty file."}), 400

    if not allowed_file(audio_file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    temp_dir = Path(tempfile.mkdtemp(prefix="emotion_"))

    try:
        ext = Path(audio_file.filename).suffix.lower() or ".webm"
        raw_path = temp_dir / f"input{ext}"
        audio_file.save(str(raw_path))

        print(f"[INFO] Saved raw file: {raw_path}")
        print(f"[INFO] Raw file exists: {raw_path.exists()}")

        wav_path = convert_to_wav(raw_path)
        pred_label, confidence, probs = predict_emotion(wav_path)

        result = {
            "predicted_emotion": pred_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                label: round(float(prob) * 100, 2)
                for label, prob in zip(LABELS, probs)
            },
        }
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)