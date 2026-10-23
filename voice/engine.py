import wave
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parents[1] / "data" / "voice" / "piper"


def default_model(voice_dir=VOICE_DIR):
    found = sorted(Path(voice_dir).glob("*.onnx"))
    return found[0] if found else None


def load_voice(model_path):
    from piper import PiperVoice
    return PiperVoice.load(str(model_path))


def speak(voice, text, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as wav:
        voice.synthesize_wav(text, wav)
    return out
