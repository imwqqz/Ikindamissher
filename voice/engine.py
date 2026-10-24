import asyncio
import re
import wave
from pathlib import Path

VOICE_DIR = Path(__file__).resolve().parents[1] / "data" / "voice" / "piper"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


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


def split_sentences(text):
    # one synthesis per sentence: real pauses at punctuation, first audio sooner
    return [p for p in SENTENCE_SPLIT.split(text.strip()) if p]


def looks_hallucinated(text):
    # whisper on non-speech repeats one short phrase; real speech does not
    words = text.lower().split()
    if len(words) < 6:
        return False
    if len(set(words)) <= 2:
        return True
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return max(trigrams.count(t) for t in set(trigrams)) >= 3


def edge_speak(text, voice, out_path, rate="+0%", pitch="+0Hz", gap=0.06):
    # edge-tts only emits mp3; libsndfile >= 1.1 decodes it, so no ffmpeg
    import edge_tts
    import numpy as np
    import soundfile as sf

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp3 = out.with_suffix(".mp3")
    pieces = []
    rate_hz = 24000
    try:
        for sentence in split_sentences(text):
            asyncio.run(edge_tts.Communicate(sentence, voice, rate=rate,
                                             pitch=pitch).save(str(mp3)))
            data, rate_hz = sf.read(str(mp3), dtype="float32", always_2d=True)
            pieces.append(data)
            pieces.append(np.zeros((int(rate_hz * gap), data.shape[1]),
                                   dtype="float32"))
    finally:
        mp3.unlink(missing_ok=True)
    if not pieces:
        raise ValueError("nothing to speak")
    sf.write(str(out), np.concatenate(pieces), rate_hz, subtype="PCM_16")
    return out
