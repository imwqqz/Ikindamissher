import argparse
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_VOICE = Path(__file__).resolve().parent
sys.path.insert(0, str(_VOICE))

import engine as voice_engine

_speak_lock = threading.Lock()
_listen_lock = threading.Lock()
_pipe = None


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _wav(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.split("?")[0] != "/health":
            return self._json(404, {"error": "unknown endpoint"})
        return self._json(200, {"ok": True, "tts": "piper",
                                "voice": Path(self.server.S["model"]).name})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/speak":
                return self._speak()
            if path == "/listen":
                return self._listen()
            return self._json(404, {"error": "unknown endpoint"})
        except Exception as exc:
            print(f"[voice] {type(exc).__name__}: {exc}")
            return self._json(500, {"error": "voice failed"})

    def _speak(self):
        text = str(self._body().get("text", "")).strip()
        if not text:
            return self._json(400, {"error": "text must not be empty"})
        out = Path(tempfile.gettempdir()) / f"her_speak_{time.time_ns()}.wav"
        started = time.perf_counter()
        with _speak_lock:
            voice_engine.speak(self.server.S["voice"], text, out)
        data = out.read_bytes()
        out.unlink(missing_ok=True)
        print(f"[voice] spoke {len(text)} chars in "
              f"{time.perf_counter() - started:.2f}s")
        return self._wav(data)

    def _listen(self):
        return self._json(200, {"text": self._record_and_transcribe(
            self.server)})

    def _record_and_transcribe(self, server):
        import numpy as np
        import sounddevice as sd

        rate = 16000
        block = 0.1
        quiet_needed = server.S["silence"]
        quiet = 0.0
        frames = []
        with _listen_lock, sd.InputStream(samplerate=rate, channels=1,
                                          dtype="float32") as stream:
            start = time.time()
            while time.time() - start < server.S["max_seconds"]:
                chunk, _ = stream.read(int(rate * block))
                frames.append(chunk.copy())
                rms = float(np.sqrt((chunk ** 2).mean()))
                quiet = quiet + block if rms < server.S["threshold"] else 0.0
                if quiet >= quiet_needed and len(frames) > 5:
                    break
        audio = np.concatenate(frames).reshape(-1)
        global _pipe
        if _pipe is None:
            from transformers import pipeline
            _pipe = pipeline("automatic-speech-recognition",
                             model=server.S["whisper"],
                             device=0 if server.S["device"] == "cuda" else -1)
        result = _pipe({"array": audio, "sampling_rate": rate},
                       generate_kwargs={"language": server.S["language"]})
        return str(result.get("text", "")).strip()


def build_parser():
    ap = argparse.ArgumentParser(
        description="local voice service: Piper TTS + Whisper STT")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8779)
    ap.add_argument("--voice-model", default=None,
                    help="Piper .onnx voice (default: first in data/voice/piper)")
    ap.add_argument("--language", default="es", help="Whisper language")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Whisper device; TTS is always CPU")
    ap.add_argument("--whisper", default="openai/whisper-small")
    ap.add_argument("--silence", type=float, default=1.2,
                    help="seconds of quiet that ends the recording")
    ap.add_argument("--threshold", type=float, default=0.02,
                    help="mic RMS below this counts as silence")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    model = Path(args.voice_model) if args.voice_model \
        else voice_engine.default_model()
    if not model or not model.exists():
        print(f"no Piper voice found in {voice_engine.VOICE_DIR}")
        print("download one with:\n  uv run python -m piper.download_voices "
              "es_AR-daniela-high --download-dir data/voice/piper")
        return 1

    started = time.perf_counter()
    voice = voice_engine.load_voice(model)
    print(f"piper ready in {time.perf_counter() - started:.2f}s "
          f"({model.name})")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.S = {
        "voice": voice,
        "model": model,
        "language": args.language,
        "device": args.device,
        "whisper": args.whisper,
        "silence": args.silence,
        "threshold": args.threshold,
        "max_seconds": args.max_seconds,
    }
    print(f"voice ready on http://{args.host}:{args.port} "
          f"(language: {args.language})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
