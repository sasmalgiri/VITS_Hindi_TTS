"""Lightweight HTTP server wrapping a loaded TTSEngine.

Endpoints:
    GET  /health             — liveness
    GET  /info               — engine metadata
    POST /speak              — JSON {text: str} → audio/wav (binary)
    POST /render-srt         — multipart SRT upload → audio/wav (binary)

Auto-generated OpenAPI docs at /docs.

Start with:
    hindi-tts-builder serve my_voice --host 127.0.0.1 --port 8765

Or directly from Python:
    from hindi_tts_builder.cli.server import run_server
    run_server("projects/my_voice/engine")
"""
from __future__ import annotations
from pathlib import Path
import io
import tempfile


def create_app(engine_dir: Path):
    """Build the FastAPI app bound to the engine at `engine_dir`.

    Imports are deferred so merely importing this module doesn't require
    fastapi to be installed.
    """
    try:
        from fastapi import FastAPI, HTTPException, UploadFile, File
        from fastapi.responses import Response
        from pydantic import BaseModel
    except ImportError as e:
        raise ImportError(
            "fastapi + pydantic required for HTTP server. "
            "Install: pip install fastapi uvicorn pydantic"
        ) from e

    from hindi_tts_builder.inference.engine import TTSEngine
    from hindi_tts_builder.inference.srt_renderer import SRTRenderer

    engine = TTSEngine.load(engine_dir)

    app = FastAPI(
        title="Hindi TTS Builder — Inference API",
        version="1.0.0",
        description="Generate audio from Hindi text or SRT files.",
    )

    class SpeakRequest(BaseModel):
        text: str
        validate: bool = True
        seed: int | None = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        return {
            "project": engine.manifest.project_name,
            "language": engine.manifest.language,
            "sample_rate": engine.manifest.sample_rate,
            "model_type": engine.manifest.model_type,
            "vocab_size": engine.tokenizer.vocab_size,
            "device": engine.device,
        }

    @app.post("/speak")
    def speak(req: SpeakRequest):
        if not req.text or not req.text.strip():
            raise HTTPException(400, "text must be non-empty")
        try:
            result = engine.speak(req.text, validate=req.validate, seed=req.seed)
        except Exception as e:
            raise HTTPException(500, str(e))

        import soundfile as sf  # type: ignore
        buf = io.BytesIO()
        sf.write(buf, result.audio, result.sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)

        headers = {
            "X-Retries": str(result.retries),
            "X-Sample-Rate": str(result.sample_rate),
        }
        if result.validation is not None:
            headers["X-Validation-Passed"] = str(result.validation.passed).lower()
            headers["X-Validation-CER"] = f"{result.validation.cer:.4f}"
        return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)

    @app.post("/render-srt")
    async def render_srt(srt: UploadFile = File(...), mode: str = "fit_to_cue"):
        content = await srt.read()
        with tempfile.TemporaryDirectory() as td:
            srt_path = Path(td) / "input.srt"
            srt_path.write_bytes(content)
            out_path = Path(td) / "output.wav"
            renderer = SRTRenderer(engine, mode=mode)
            try:
                summary = renderer.render(srt_path, out_path)
            except Exception as e:
                raise HTTPException(500, str(e))
            audio_bytes = out_path.read_bytes()
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Cues-Rendered": str(summary["cues_rendered"]),
                "X-Duration-Sec": f"{summary['duration_sec']:.2f}",
                "X-Validation-Failures": str(summary["validation_failures"]),
            },
        )

    return app


def run_server(engine_dir: Path | str, host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        import uvicorn  # type: ignore
    except ImportError as e:
        raise ImportError(
            "uvicorn required to run server. Install: pip install uvicorn[standard]"
        ) from e
    app = create_app(Path(engine_dir))
    uvicorn.run(app, host=host, port=port)
