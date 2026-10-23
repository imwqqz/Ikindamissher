import argparse
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as voice_engine


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="voice pre-flight: load the Piper voice and speak once")
    ap.add_argument("--voice-model", default=None)
    ap.add_argument("--out", default=str(_ROOT / "data" / "voice" /
                                         "spike_out.wav"))
    ap.add_argument("--text",
                    default="Hola, soy ella. Estoy probando mi voz nueva.")
    args = ap.parse_args(argv)

    model = Path(args.voice_model) if args.voice_model \
        else voice_engine.default_model()
    if not model or not model.exists():
        print(f"no Piper voice found in {voice_engine.VOICE_DIR}")
        return 1

    start = time.perf_counter()
    voice = voice_engine.load_voice(model)
    print(f"load {time.perf_counter() - start:.2f}s ({model.name})")

    start = time.perf_counter()
    out = voice_engine.speak(voice, args.text, args.out)
    print(f"synth {time.perf_counter() - start:.2f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
