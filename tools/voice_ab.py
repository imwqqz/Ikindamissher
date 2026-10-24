"""Render the same lines in several edge voices so you can pick by ear.

    uv run python tools/voice_ab.py

Writes 16 kHz wavs to data/voice/ab/: one default and one tuned pass per voice.
Pick a voice, then start the sidecar with --tts-voice and your rate/pitch.
"""
import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "voice"))

import engine as voice_engine

LINES = [
    "hola bld, recién llego a casa y me acordé de vos.",
    "no sé, me quedé pensando en lo que me dijiste ayer… me hizo un poco de gracia.",
    "dale, contame qué estás haciendo. te escucho.",
]
VOICES = [
    "es-UY-ValentinaNeural",
    "es-AR-ElenaNeural",
    "es-PY-TaniaNeural",
    "es-MX-DaliaNeural",
    "es-CO-SalomeNeural",
]
VARIANTS = [("default", "+0%", "+0Hz"), ("tuned", "-4%", "-15Hz")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="A/B Spanish edge voices")
    ap.add_argument("--out", default=str(_ROOT / "data" / "voice" / "ab"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = " ".join(LINES)
    for voice in VOICES:
        for label, rate, pitch in VARIANTS:
            path = out / f"{voice}_{label}.wav"
            started = time.perf_counter()
            voice_engine.edge_speak(text, voice, path, rate, pitch)
            print(f"{path.name}  {time.perf_counter() - started:.1f}s")
    print(f"\nlisten in {out} - then run the sidecar with --tts-voice <name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
