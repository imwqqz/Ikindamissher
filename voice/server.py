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
_pipe_lock = threading.Lock()


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
        S = self.server.S
        voice = S["tts_voice"] if S["tts"] == "edge" \
            else Path(S["model"]).name
        return self._json(200, {"ok": True, "tts": S["tts"], "voice": voice,
                                "last_tts": S["last_tts"],
                                "state": S["state"], "rms": S["rms"],
                                "since": S["since"]})

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/speak":
                return self._speak()
            if path == "/listen":
                return self._listen()
            if path == "/listen/stop":
                return self._stop()
            return self._json(404, {"error": "unknown endpoint"})
        except Exception as exc:
            print(f"[voice] {type(exc).__name__}: {exc}")
            return self._json(500, {"error": "voice failed"})

    def _speak(self):
        text = str(self._body().get("text", "")).strip()
        if not text:
            return self._json(400, {"error": "text must not be empty"})
        S = self.server.S
        out = Path(tempfile.gettempdir()) / f"her_speak_{time.time_ns()}.wav"
        started = time.perf_counter()
        with _speak_lock:
            try:
                if S["tts"] == "edge":
                    voice_engine.edge_speak(text, S["tts_voice"], out,
                                            S["tts_rate"], S["tts_pitch"])
                    S["last_tts"] = "edge"
                else:
                    voice_engine.speak(S["voice"], text, out)
                    S["last_tts"] = "piper"
            except Exception as exc:
                if S["tts"] != "edge" or S["voice"] is None:
                    raise
                print(f"[voice] edge failed ({type(exc).__name__}: {exc}); "
                      "falling back to piper", flush=True)
                voice_engine.speak(S["voice"], text, out)
                S["last_tts"] = "piper"
        data = out.read_bytes()
        out.unlink(missing_ok=True)
        print(f"[voice] spoke {len(text)} chars in "
              f"{time.perf_counter() - started:.2f}s", flush=True)
        return self._wav(data)

    def _listen(self):
        return self._json(200, {"text": self._record_and_transcribe(
            self.server)})

    def _stop(self):
        # no _listen_lock here or a stop during a recording would deadlock
        stop = self.server.S.get("stop")
        if stop is not None:
            stop.set()
        return self._json(200, {"ok": True})

    def _record_and_transcribe(self, server):
        import numpy as np
        import sounddevice as sd

        rate = 16000
        block = 0.1
        quiet_needed = server.S["silence"]
        quiet = 0.0
        voiced = 0.0
        peak = 0.0
        frames = []
        try:
            with _listen_lock:
                stop = threading.Event()
                server.S["stop"] = stop
                server.S["state"] = "recording"
                server.S["since"] = time.time()
                server.S["rms"] = 0.0
                with sd.InputStream(samplerate=rate, channels=1,
                                    dtype="float32") as stream:
                    start = time.time()
                    while time.time() - start < server.S["max_seconds"] \
                            and not stop.is_set():
                        chunk, _ = stream.read(int(rate * block))
                        frames.append(chunk.copy())
                        rms = float(np.sqrt((chunk ** 2).mean()))
                        server.S["rms"] = rms
                        peak = max(peak, rms)
                        if rms >= server.S["threshold"]:
                            voiced += block
                            quiet = 0.0
                        else:
                            quiet += block
                        if quiet >= quiet_needed and len(frames) > 5:
                            break
                server.S["state"] = "transcribing"
                server.S["since"] = time.time()
                audio = np.concatenate(frames).reshape(-1) if frames \
                    else np.zeros(0, dtype="float32")
                # whisper invents text from silence ("www.mooji.org"); never
                # hand it a clip where nothing crossed the threshold
                if voiced < 0.25:
                    print(f"[voice] no speech ({voiced:.2f}s voiced, "
                          f"peak {peak:.3f} vs threshold "
                          f"{server.S['threshold']:.3f})", flush=True)
                    return ""
                global _pipe
                if _pipe is None:
                    with _pipe_lock:
                        if _pipe is None:
                            from transformers import pipeline
                            _pipe = pipeline(
                                "automatic-speech-recognition",
                                model=server.S["whisper"],
                                device=0 if server.S["device"] == "cuda" else -1)
                result = _pipe({"array": audio, "sampling_rate": rate},
                               generate_kwargs={"language": server.S["language"]})
                text = str(result.get("text", "")).strip()
                if voice_engine.looks_hallucinated(text):
                    print(f"[voice] dropped degenerate transcript "
                          f"{text[:40]!r}", flush=True)
                    return ""
                print(f"[voice] heard {len(audio) / rate:.1f}s "
                      f"rms {server.S['rms']:.3f} -> {text[:60]!r}", flush=True)
                return text
        finally:
            server.S["stop"] = None
            server.S["state"] = "idle"
            server.S["since"] = time.time()


def build_parser():
    ap = argparse.ArgumentParser(
        description="local voice service: edge TTS (Piper fallback) + Whisper")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8779)
    ap.add_argument("--tts", default="edge", choices=["edge", "piper"],
                    help="edge = Microsoft neural voice (network, the text "
                         "leaves the machine); piper = offline fallback")
    ap.add_argument("--tts-voice", default="es-UY-ValentinaNeural",
                    help="edge voice id; see tools/voice_ab.py to A/B voices")
    ap.add_argument("--tts-rate", default="-4%", help="edge speaking rate")
    ap.add_argument("--tts-pitch", default="-15Hz", help="edge pitch offset")
    ap.add_argument("--voice-model", default=None,
                    help="Piper .onnx voice (default: first in data/voice/piper)")
    ap.add_argument("--language", default="es", help="Whisper language")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Whisper device; Piper is always CPU")
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
        "tts": args.tts,
        "tts_voice": args.tts_voice,
        "tts_rate": args.tts_rate,
        "tts_pitch": args.tts_pitch,
        "last_tts": args.tts,
        "language": args.language,
        "device": args.device,
        "whisper": args.whisper,
        "silence": args.silence,
        "threshold": args.threshold,
        "max_seconds": args.max_seconds,
        "state": "idle",
        "rms": 0.0,
        "since": time.time(),
        "stop": None,
    }
    tts_label = f"edge/{args.tts_voice}" if args.tts == "edge" else "piper"
    print(f"voice ready on http://{args.host}:{args.port} "
          f"(tts: {tts_label}, language: {args.language})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
