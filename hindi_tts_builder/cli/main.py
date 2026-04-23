"""`hindi-tts-builder` command-line interface.

Sub-commands:
    new             Create a new project
    add-sources     Register YouTube URLs + SRT transcripts into a project
    prepare         Run the full data pipeline (download → QC → training set)
    train           Run VITS training
    export          Bundle the trained model into an `engine/` folder
    speak           Generate audio from text using an engine
    render-srt      Render an SRT file to a timed WAV using an engine
    serve           Start a FastAPI HTTP server
    studio          Start the browser-based training launcher (URLs + SRT → train)
    doctor          Diagnose environment (GPU, dependencies, etc.)

Every sub-command is discoverable via `hindi-tts-builder <cmd> --help`.
"""
from __future__ import annotations
from pathlib import Path
import json
import sys

import click

# All heavy imports are inside each sub-command body.


def _projects_root() -> Path:
    """Current-working-directory-relative projects root, resolved at call time."""
    return Path.cwd() / "projects"


def _project_path(name: str) -> Path:
    return _projects_root() / name


def _echo_ok(msg: str) -> None:
    click.secho(f"✓ {msg}", fg="green")


def _echo_err(msg: str) -> None:
    click.secho(f"✗ {msg}", fg="red", err=True)


def _echo_info(msg: str) -> None:
    click.secho(f"  {msg}", fg="cyan")


# =========================================================================
# Root group
# =========================================================================
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
def cli():
    """hindi-tts-builder — build a private Hindi TTS engine from YouTube + SRT."""


# =========================================================================
# new
# =========================================================================
@cli.command()
@click.argument("name")
def new(name: str):
    """Create a new project directory."""
    from hindi_tts_builder.utils.project import create_project
    if _project_path(name).exists():
        _echo_err(f"Project '{name}' already exists at {_project_path(name)}")
        sys.exit(1)
    root = _projects_root()
    root.mkdir(parents=True, exist_ok=True)
    paths = create_project(root, name)
    _echo_ok(f"Created project '{name}' at {paths.root}")
    _echo_info("Next: add sources with `hindi-tts-builder add-sources {name} --urls urls.txt --transcripts ./transcripts/`".format(name=name))


# =========================================================================
# add-sources
# =========================================================================
@cli.command(name="add-sources")
@click.argument("name")
@click.option("--urls", "urls_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="File with one YouTube URL per line.")
@click.option("--transcripts", "transcripts_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing .srt files matching URLs in sorted order.")
def add_sources(name: str, urls_file: Path, transcripts_dir: Path):
    """Register YouTube URLs + SRT transcripts into a project's manifest."""
    from hindi_tts_builder.data.pipeline import add_sources_from_files
    from hindi_tts_builder.utils.project import ProjectPaths

    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist. Run: hindi-tts-builder new {name}")
        sys.exit(1)
    paths = ProjectPaths(pp)
    paths.ensure_all()
    try:
        added = add_sources_from_files(paths, urls_file, transcripts_dir)
        _echo_ok(f"Added {added} sources to '{name}'")
    except Exception as e:
        _echo_err(str(e))
        sys.exit(1)


# =========================================================================
# prepare
# =========================================================================
@cli.command()
@click.argument("name")
@click.option("--no-whisperx", is_flag=True, help="Skip WhisperX alignment (use SRT timestamps as-is)")
@click.option("--no-whisper-qc", is_flag=True, help="Skip Whisper round-trip QC filtering")
def prepare(name: str, no_whisperx: bool, no_whisper_qc: bool):
    """Run full data pipeline: download → align → segment → QC → build dataset."""
    from hindi_tts_builder.data.pipeline import run_pipeline
    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    try:
        summary = run_pipeline(
            pp,
            use_whisperx=not no_whisperx,
            use_whisper_qc=not no_whisper_qc,
        )
    except Exception as e:
        _echo_err(f"Pipeline failed: {e}")
        sys.exit(1)

    _echo_ok("Pipeline complete")
    _echo_info(f"Download:  {summary['download']}")
    _echo_info(f"Align:     {summary['align']}")
    _echo_info(f"Segment:   {summary['segment']}")
    _echo_info(f"QC:        {summary['qc']}")
    _echo_info(f"Dataset:   {summary['dataset']}")


# =========================================================================
# train
# =========================================================================
@cli.command()
@click.argument("name")
@click.option("--prepare-only", is_flag=True, help="Only fit tokenizer and validate data; do not start training.")
def train(name: str, prepare_only: bool):
    """Train VITS on the project's prepared data. Resumable."""
    from hindi_tts_builder.train.trainer import Trainer
    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    try:
        trainer = Trainer(pp)
        ready = trainer.prepare()
        _echo_ok("Training prepared")
        for k, v in ready.items():
            _echo_info(f"{k}: {v}")
        if prepare_only:
            return
        trainer.train()
        _echo_ok("Training complete")
    except ImportError as e:
        _echo_err(str(e))
        sys.exit(1)
    except Exception as e:
        _echo_err(f"Training failed: {e}")
        raise


# =========================================================================
# export
# =========================================================================
@cli.command()
@click.argument("name")
def export(name: str):
    """Package the trained model into an engine/ folder for inference."""
    from hindi_tts_builder.train.trainer import Trainer
    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    try:
        trainer = Trainer(pp)
        engine_dir = trainer.export_engine()
        _echo_ok(f"Engine exported to {engine_dir}")
    except Exception as e:
        _echo_err(f"Export failed: {e}")
        sys.exit(1)


# =========================================================================
# speak
# =========================================================================
@cli.command()
@click.argument("name")
@click.option("--text", "-t", required=True, help="Hindi text to speak.")
@click.option("--out", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Output WAV path.")
@click.option("--no-validate", is_flag=True, help="Disable Whisper round-trip validation.")
@click.option("--seed", type=int, help="Random seed for reproducible output.")
def speak(name: str, text: str, out: Path, no_validate: bool, seed: int | None):
    """Generate audio for a single text string."""
    from hindi_tts_builder.inference.engine import TTSEngine
    pp = _project_path(name)
    engine_dir = pp / "engine"
    if not engine_dir.exists():
        _echo_err(f"No engine at {engine_dir}. Run `export` first.")
        sys.exit(1)
    try:
        engine = TTSEngine.load(engine_dir, enable_roundtrip=not no_validate)
        result = engine.speak(text, output=out, validate=not no_validate, seed=seed)
    except Exception as e:
        _echo_err(f"Generation failed: {e}")
        sys.exit(1)
    _echo_ok(f"Wrote {out}")
    _echo_info(f"sample_rate={result.sample_rate}Hz")
    _echo_info(f"duration={len(result.audio) / result.sample_rate:.2f}s")
    if result.validation is not None:
        status = "passed" if result.validation.passed else f"FAILED ({result.validation.reason})"
        _echo_info(f"validation={status} cer={result.validation.cer:.3f}")
    if result.retries > 0:
        _echo_info(f"retries={result.retries}")


# =========================================================================
# render-srt
# =========================================================================
@cli.command(name="render-srt")
@click.argument("name")
@click.option("--srt", "srt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", "-o", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--mode", type=click.Choice(["fit_to_cue", "natural"]), default="fit_to_cue")
@click.option("--gap-ms", type=int, default=200, help="(natural mode) silence between cues.")
def render_srt(name: str, srt_path: Path, out: Path, mode: str, gap_ms: int):
    """Render an SRT file to a timed WAV."""
    from hindi_tts_builder.inference.engine import TTSEngine
    from hindi_tts_builder.inference.srt_renderer import SRTRenderer

    pp = _project_path(name)
    engine_dir = pp / "engine"
    if not engine_dir.exists():
        _echo_err(f"No engine at {engine_dir}. Run `export` first.")
        sys.exit(1)

    engine = TTSEngine.load(engine_dir)
    renderer = SRTRenderer(engine, mode=mode, gap_ms_between_cues=gap_ms)

    last_i = [0]
    def _cb(p):
        if p.cue_index % 10 == 0 or p.cue_index == p.total_cues:
            click.echo(f"  [{p.cue_index}/{p.total_cues}] {'✓' if p.validation_passed else '✗'} {p.cue_text[:60]}")
        last_i[0] = p.cue_index

    try:
        summary = renderer.render(srt_path, out, progress_callback=_cb)
    except Exception as e:
        _echo_err(f"Render failed: {e}")
        sys.exit(1)
    _echo_ok(f"Wrote {out}")
    _echo_info(f"cues={summary['cues_rendered']} duration={summary['duration_sec']:.1f}s validation_failures={summary['validation_failures']}")


# =========================================================================
# serve
# =========================================================================
@cli.command()
@click.argument("name")
@click.option("--host", default="127.0.0.1")
@click.option("--port", type=int, default=8765)
def serve(name: str, host: str, port: int):
    """Start a local HTTP API server."""
    from hindi_tts_builder.cli.server import run_server
    pp = _project_path(name)
    engine_dir = pp / "engine"
    if not engine_dir.exists():
        _echo_err(f"No engine at {engine_dir}. Run `export` first.")
        sys.exit(1)
    _echo_ok(f"Serving {name} at http://{host}:{port} (OpenAPI: /docs)")
    run_server(engine_dir, host=host, port=port)


# =========================================================================
# studio (training launcher web UI)
# =========================================================================
@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8770, show_default=True)
@click.option("--projects-root", type=click.Path(file_okay=False, path_type=Path),
              default=None,
              help="Directory holding project subdirs. Defaults to ./projects relative to CWD.")
def studio(host: str, port: int, projects_root: Path | None):
    """Launch the browser-based training studio (paste URLs + SRTs, click train)."""
    from hindi_tts_builder.web.app import run_studio
    root = projects_root or _projects_root()
    root.mkdir(parents=True, exist_ok=True)
    _echo_ok(f"Studio at http://{host}:{port}  (projects: {root})")
    _echo_info("Open the URL in your browser. Ctrl+C to stop.")
    run_studio(root, host=host, port=port)


# =========================================================================
# doctor
# =========================================================================
@cli.command()
def doctor():
    """Diagnose the environment: GPU, CUDA, dependencies, disk."""
    import shutil

    def _check(label, ok, detail=""):
        (_echo_ok if ok else _echo_err)(f"{label}: {detail}")

    # Python
    _echo_info(f"Python: {sys.version.split()[0]}")

    # torch + CUDA
    try:
        import torch  # type: ignore
        cuda = torch.cuda.is_available()
        _check(
            "torch+CUDA", cuda,
            f"torch {torch.__version__}, "
            f"GPU: {torch.cuda.get_device_name(0) if cuda else 'none'}, "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if cuda else "CUDA not available"
        )
    except ImportError:
        _check("torch", False, "not installed — pip install torch torchaudio")

    # ffmpeg
    _check("ffmpeg", shutil.which("ffmpeg") is not None,
           shutil.which("ffmpeg") or "not found on PATH")

    # yt-dlp
    _check("yt-dlp", shutil.which("yt-dlp") is not None,
           shutil.which("yt-dlp") or "not found — pip install yt-dlp")

    # Coqui TTS
    try:
        import TTS  # type: ignore
        _check("coqui-tts", True, f"version {getattr(TTS, '__version__', 'unknown')}")
    except ImportError:
        _check("coqui-tts", False, "not installed — pip install coqui-tts")

    # faster-whisper
    try:
        import faster_whisper  # type: ignore
        _check("faster-whisper", True, "installed")
    except ImportError:
        try:
            import whisper  # type: ignore
            _check("faster-whisper", False, "openai-whisper installed instead (works but slower)")
        except ImportError:
            _check("whisper", False, "not installed — pip install faster-whisper")

    # Disk
    du = shutil.disk_usage(Path.cwd())
    free_gb = du.free / 1e9
    _check("disk-space", free_gb > 50.0, f"{free_gb:.1f} GB free (need ≥50 GB for 50h project)")


if __name__ == "__main__":
    cli()
