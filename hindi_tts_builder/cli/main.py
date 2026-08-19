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
@click.option("--skip-qc", is_flag=True, help="Skip QC entirely; mark every clip as passed (quick experiments only)")
@click.option("--trust-srt", is_flag=True, help="Trust SRT timestamps as-is. Skip Stage 2 (align) and use trusted_srt_cut "
              "(50ms left + 100ms right pad) in Stage 3. Use when SRT timing is verified word-to-word.")
@click.option("--mode", "segmentation_mode", default=None,
              type=click.Choice(["cue", "sentence", "aligned_words"]),
              help="How to choose clip boundaries. cue=one clip per SRT cue (legacy default). "
                   "sentence=merge cues to sentence ends. aligned_words=cut from ASR word "
                   "timings (the only mode that reaches a boundary inside a cue). "
                   "Overrides config's segmentation.mode.")
@click.option("--transcribe/--no-transcribe", "transcribe", default=None,
              help="Force Stage 1.5 ASR transcription on or off. Default: auto — on when the "
                   "mode needs word timings or a source has no SRT.")
def prepare(name: str, no_whisperx: bool, no_whisper_qc: bool, skip_qc: bool, trust_srt: bool,
            segmentation_mode: str | None, transcribe: bool | None):
    """Run full data pipeline: download → [transcribe] → align → segment → QC → build dataset."""
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
            skip_qc=skip_qc,
            trust_srt=trust_srt,
            segmentation_mode=segmentation_mode,
            transcribe=transcribe,
        )
    except Exception as e:
        _echo_err(f"Pipeline failed: {e}")
        sys.exit(1)

    _echo_ok("Pipeline complete")
    _echo_info(f"Download:   {summary['download']}")
    _echo_info(f"Transcribe: {summary['transcribe']}")
    _echo_info(f"Align:      {summary['align']}")
    _echo_info(f"Segment:    {summary['segment']}")
    _echo_info(f"QC:         {summary['qc']}")
    _echo_info(f"Dataset:    {summary['dataset']}")

    gate = summary.get("gate")
    if gate and not gate["ok"]:
        _echo_err(f"Pre-train gate: {len(gate['blockers'])} blocker(s) — training is blocked")
        for b in gate["blockers"]:
            _echo_err(f"  - {b}")
        _echo_info("Run `hindi-tts-builder gate " + name + "` for the full report.")
    elif gate:
        _echo_ok("Pre-train gate passed")


# =========================================================================
# add-url  (no-SRT path: bare YouTube links)
# =========================================================================
@cli.command(name="add-url")
@click.argument("name")
@click.argument("urls", nargs=-1, required=True)
def add_url(name: str, urls: tuple[str, ...]):
    """Register YouTube URLs with NO transcript — text comes from ASR.

    Use for a fresh voice where you have video but no SRT. `prepare` will run
    Stage 1.5 (Whisper transcription with word timings) automatically, then cut
    on real sentence boundaries.
    """
    from hindi_tts_builder.data.manifest import Manifest
    from hindi_tts_builder.utils.project import ProjectPaths

    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist. Run: hindi-tts-builder new {name}")
        sys.exit(1)
    paths = ProjectPaths(pp)
    paths.ensure_all()
    manifest = Manifest(paths.sources / "manifest.json")
    existing = {s.url for s in manifest.sources}
    added = 0
    for i, url in enumerate(urls, start=len(manifest.sources)):
        if url in existing:
            _echo_info(f"already present, skipping: {url}")
            continue
        src = manifest.add(url=url, transcript_path=None, index=i)
        src.transcript_origin = "asr"
        added += 1
    manifest.save()
    _echo_ok(f"Added {added} URL(s) to '{name}' (transcript: ASR)")
    _echo_info(f"Next: hindi-tts-builder prepare {name} --mode aligned_words")


# =========================================================================
# srt-health / gate / transcribe / resegment
# =========================================================================
@cli.command(name="srt-health")
@click.argument("target", type=click.Path(exists=True, path_type=Path))
def srt_health_cmd(target: Path):
    """Check SRT timelines for fabricated timings before trusting them."""
    from hindi_tts_builder.data.srt_health import analyze_timeline
    from hindi_tts_builder.utils.srt import parse_srt

    files = sorted(target.glob("*.srt")) if target.is_dir() else [target]
    if not files:
        _echo_err(f"no .srt found under {target}")
        sys.exit(1)
    bad = 0
    for f in files:
        click.echo(f"\n=== {f.name}")
        rep = analyze_timeline(parse_srt(f))
        click.echo(rep.describe())
        if not rep.trustworthy:
            bad += 1
    click.echo("")
    if bad:
        _echo_err(f"{bad}/{len(files)} timeline(s) should not be used for cutting audio")
        _echo_info("Use `prepare --mode aligned_words` to cut from ASR word timings instead.")
        sys.exit(1)
    _echo_ok(f"all {len(files)} timeline(s) look measured, not generated")


@cli.command(name="gate")
@click.argument("name")
@click.option("--profile", default=None, help="Check train_<profile>.csv instead of train.csv")
@click.option("--force", is_flag=True, help="Demote blockers to warnings (deliberate override)")
def gate_cmd(name: str, profile: str | None, force: bool):
    """Check whether a prepared corpus is fit to train on."""
    from hindi_tts_builder.data.gate import check_corpus
    from hindi_tts_builder.utils.project import load_config

    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    cfg = load_config(pp)
    res = check_corpus(pp, profile=profile, force=force, **dict(cfg["gate"]))
    click.echo(res.describe())
    if not res.ok:
        sys.exit(1)


@cli.command(name="transcribe")
@click.argument("name")
@click.option("--model", default=None, help="Whisper model (default: config asr.model)")
@click.option("--all", "do_all", is_flag=True, help="Transcribe every source, even those with a trustworthy SRT")
@click.option("--overwrite", is_flag=True, help="Re-transcribe sources that already have word timings")
def transcribe_cmd(name: str, model: str | None, do_all: bool, overwrite: bool):
    """Stage 1.5: ASR-transcribe audio into word-level timings."""
    from hindi_tts_builder.data.manifest import Manifest
    from hindi_tts_builder.data.transcribe import transcribe_sources
    from hindi_tts_builder.utils.project import ProjectPaths, load_config

    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    cfg = load_config(pp)
    paths = ProjectPaths(pp)
    manifest = Manifest(paths.sources / "manifest.json")
    asr, seg = cfg["asr"], cfg["segmentation"]
    try:
        s = transcribe_sources(
            paths, manifest,
            language=cfg.get("language", "hi"),
            model=model or asr["model"],
            compute_type=asr["compute_type"],
            beam_size=asr["beam_size"],
            min_seconds=seg["min_seconds"],
            max_seconds=seg["max_seconds"],
            max_gap_seconds=seg["max_gap_seconds"],
            skip_existing=not overwrite,
            only_untrusted=not do_all,
        )
    except Exception as e:
        _echo_err(f"Transcription failed: {e}")
        sys.exit(1)
    _echo_ok(f"Transcribed {s['sources_transcribed']} source(s) with {s['model_used']}")
    _echo_info(f"words={s['words_total']} units={s['units_total']} "
               f"sentence-terminated={100 * s['terminator_ratio']:.1f}%")
    if s["latin_ratio"] > 0.02:
        _echo_err(f"{100 * s['latin_ratio']:.1f}% Latin-script letters — transliterate before training")
    if s["untrainable_chars"]:
        _echo_info(f"folded/dropped characters: {s['untrainable_chars']}")


@cli.command(name="sources")
@click.argument("name")
@click.option("--exclude", multiple=True, help="Source id to exclude from training (repeatable)")
@click.option("--include", multiple=True, help="Re-include a previously excluded source (repeatable)")
def sources_cmd(name: str, exclude: tuple[str, ...], include: tuple[str, ...]):
    """List sources, or exclude/include them without editing the manifest by hand.

    Excluding marks a source so later stages skip it. Its audio and word timings
    stay on disk, so re-including costs nothing.
    """
    from hindi_tts_builder.data.manifest import Manifest
    from hindi_tts_builder.data.srt_health import analyze_timeline
    from hindi_tts_builder.utils.project import ProjectPaths
    from hindi_tts_builder.utils.srt import parse_srt

    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    paths = ProjectPaths(pp)
    manifest = Manifest(paths.sources / "manifest.json")

    changed = 0
    for sid in exclude:
        src = manifest.find(sid)
        if not src:
            _echo_err(f"unknown source: {sid}")
            continue
        src.excluded = True
        changed += 1
        _echo_info(f"excluded {sid}")
    for sid in include:
        src = manifest.find(sid)
        if not src:
            _echo_err(f"unknown source: {sid}")
            continue
        src.excluded = False
        changed += 1
        _echo_info(f"included {sid}")
    if changed:
        manifest.save()
        _echo_ok(f"updated {changed} source(s)")

    click.echo("")
    for src in manifest:
        mark = "EXCLUDED" if src.excluded else "        "
        note = ""
        if src.transcript_path:
            srt = paths.root / src.transcript_path
            if srt.exists():
                try:
                    rep = analyze_timeline(parse_srt(srt))
                    note = (f"{rep.terminators_per_1k:5.2f} term/1k  "
                            f"{rep.verdict:<22} {rep.recommended_mode}")
                except Exception as e:
                    note = f"(unreadable SRT: {e})"
        else:
            note = "no transcript (ASR path)"
        click.echo(f"  {mark}  {src.id:<24} {note}")


@cli.command(name="force-align")
@click.argument("name")
@click.option("--overwrite", is_flag=True, help="Re-align sources that already have word timings")
@click.option("--chunk-minutes", default=20.0, show_default=True, type=float)
def force_align_cmd(name: str, overwrite: bool, chunk_minutes: float):
    """Recover real word timings for an EXISTING transcript.

    Use when the transcript text is good but its timings are not. Far more
    accurate than re-transcribing when the SRT already carries punctuation, and
    roughly 36x faster.
    """
    from hindi_tts_builder.data.force_align import force_align_sources
    from hindi_tts_builder.data.manifest import Manifest
    from hindi_tts_builder.utils.project import ProjectPaths, load_config

    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    cfg = load_config(pp)
    paths = ProjectPaths(pp)
    manifest = Manifest(paths.sources / "manifest.json")
    seg = cfg["segmentation"]
    try:
        s = force_align_sources(
            paths, manifest,
            language=cfg.get("language", "hi"),
            min_seconds=seg["min_seconds"],
            max_seconds=seg["max_seconds"],
            max_gap_seconds=seg["max_gap_seconds"],
            chunk_minutes=chunk_minutes,
            skip_existing=not overwrite,
        )
    except Exception as e:
        _echo_err(f"Forced alignment failed: {e}")
        sys.exit(1)
    _echo_ok(f"Force-aligned {s['sources_aligned']} source(s)")
    _echo_info(f"words={s['words_total']} units={s['units_total']} "
               f"sentence-terminated={100 * s['terminator_ratio']:.1f}%")
    if s["reclaimed_min"]:
        _echo_info(f"reclaimed {s['reclaimed_min']:.1f} min from fabricated inter-cue gaps")
    if s["sources_failed"]:
        _echo_err(f"{s['sources_failed']} source(s) failed — see logs/force_align.log")


@cli.command(name="resegment")
@click.argument("name")
@click.option("--mode", "segmentation_mode", required=True,
              type=click.Choice(["cue", "sentence", "aligned_words"]))
@click.option("--force", is_flag=True, help="Delete existing clips for each source before recutting")
@click.option("--rebuild-dataset/--no-rebuild-dataset", default=True, show_default=True)
def resegment_cmd(name: str, segmentation_mode: str, force: bool, rebuild_dataset: bool):
    """Recut an existing corpus under a different segmentation policy.

    Without --force, sources whose clips were cut under another policy are
    skipped and flagged rather than silently mixed.
    """
    import shutil

    from hindi_tts_builder.data.dataset import build_training_set
    from hindi_tts_builder.data.manifest import Manifest
    from hindi_tts_builder.data.segment import segment_clips
    from hindi_tts_builder.frontend.pipeline import HindiFrontend
    from hindi_tts_builder.utils.project import ProjectPaths, load_config

    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    cfg = load_config(pp)
    paths = ProjectPaths(pp)
    manifest = Manifest(paths.sources / "manifest.json")
    seg = cfg["segmentation"]

    if force:
        for src in manifest:
            d = paths.aligned / src.id
            if d.exists():
                shutil.rmtree(d)
                _echo_info(f"cleared existing clips for {src.id}")
            src.segmentation_fingerprint = None
            src.segmentation_state = None
        manifest.save()

    s = segment_clips(
        paths, manifest,
        sample_rate=cfg["target_sample_rate"],
        loudness_lufs=cfg["target_loudness_lufs"],
        min_seconds=cfg["clip_min_seconds"],
        max_seconds=cfg["clip_max_seconds"],
        trust_srt=segmentation_mode != "aligned_words",
        trust_srt_pad_left_ms=seg["pad_left_ms"],
        trust_srt_pad_right_ms=seg["pad_right_ms"],
        segmentation_mode=segmentation_mode,
        sentence_min_seconds=seg["min_seconds"],
        sentence_max_gap_seconds=seg["max_gap_seconds"],
        sentence_max_interior_gap_seconds=seg["max_interior_gap_seconds"],
        sentence_max_interior_silence_ratio=seg["max_interior_silence_ratio"],
        close_synthetic_gaps=seg["close_synthetic_gaps"],
        refuse_untrusted_timeline=seg["refuse_untrusted_timeline"],
        skip_existing=not force,
    )
    _echo_ok(f"resegment [{segmentation_mode}]: {s['clips_created']} clips, "
             f"sentence-terminated {100 * s['terminator_ratio']:.1f}%")
    if s["sources_conflicted"]:
        _echo_err(f"{s['sources_conflicted']} source(s) skipped as policy conflicts — rerun with --force")
    if rebuild_dataset:
        # Recut clips invalidate the QC report: same clip ids, different audio.
        # Re-score before rebuilding, or the dataset inherits durations and
        # pass/fail verdicts belonging to audio that no longer exists.
        from hindi_tts_builder.data.pipeline import write_qc_meta
        from hindi_tts_builder.data.qc import quality_filter

        qc_cfg = cfg["qc"]
        _echo_info("re-running QC (recut clips invalidate the previous report)")
        quality_filter(
            paths, manifest,
            min_snr_db=qc_cfg["min_snr_db"],
            max_cer_vs_whisper=qc_cfg["max_cer_vs_whisper"],
            max_silence_ratio=qc_cfg["max_silence_ratio"],
            min_seconds=cfg["clip_min_seconds"],
            max_seconds=cfg["clip_max_seconds"],
            use_whisper=True,
            language=cfg.get("language", "hi"),
        )
        write_qc_meta(paths, "full", thresholds=dict(qc_cfg))
        d = build_training_set(paths, frontend=HindiFrontend(apply_prosody=False))
        _echo_ok(f"rebuilt training set: {d}")
    _echo_info(f"Next: hindi-tts-builder gate {name}")


# =========================================================================
# Voice library (LoRA multi-voice)
# =========================================================================
def _default_library_dir() -> Path:
    """Default voice library: <cwd>/voice_library/, project-agnostic."""
    return Path.cwd() / "voice_library"


@cli.command(name="train-indic-parler-lora")
@click.argument("project")
@click.option("--profile", default="10h_clean", show_default=True,
              type=click.Choice(["2h_clean", "10h_clean", "30h_clean", "60h_clean", "100h_clean"]))
@click.option("--adapter-name", required=True,
              help="Voice name for the trained adapter (e.g. male_narrator_16h, female_dramatic_150h).")
@click.option("--smoke", is_flag=True, help="Tiny 64-sample smoke train (~50 steps).")
@click.option("--max-steps", default=30_000, show_default=True,
              help="LoRA needs more steps than full FT (typical: 20k-50k).")
@click.option("--lr", "learning_rate", default=1e-4, show_default=True, type=float,
              help="LoRA tolerates higher LR than full FT (typical: 1e-4 vs 5e-6).")
@click.option("--rank", "lora_rank", default=32, show_default=True, type=int)
@click.option("--alpha", "lora_alpha", default=64, show_default=True, type=int)
@click.option("--dtype", default="float16", show_default=True,
              type=click.Choice(["float16", "bfloat16", "float32"]))
@click.option("--overwrite-export", is_flag=True)
@click.option("--speaker-name", default="Narrator", show_default=True)
@click.option("--per-device-batch", type=str, default="auto", show_default=True,
              help="Per-device batch size. 'auto' picks from free VRAM + corpus size. "
                   "Override with an int like 4 or 8.")
@click.option("--grad-accum", type=str, default="auto", show_default=True,
              help="Gradient accumulation steps. 'auto' picks to hit a quality-tuned effective batch.")
def train_indic_parler_lora(project: str, profile: str, adapter_name: str,
                             smoke: bool, max_steps: int, learning_rate: float,
                             lora_rank: int, lora_alpha: int, dtype: str,
                             overwrite_export: bool, speaker_name: str,
                             per_device_batch: str, grad_accum: str):
    """Train a LoRA adapter for one voice on Indic-Parler-TTS.

    Output adapter (~50 MB) goes to:
      projects/<project>/voice_library/adapters/<adapter_name>/

    Use `voice-add` to register it in your global voice library, then
    `speak-voice` to render with it.
    """
    from hindi_tts_builder.train.backends.indic_parler_lora import (
        IndicParlerLoraBackend, IndicParlerLoraConfig,
    )
    pp = _project_path(project)
    if not pp.exists():
        _echo_err(f"Project '{project}' does not exist.")
        sys.exit(1)
    # Parse per-device-batch / grad-accum: "auto" → None (let dynamic_batch decide)
    explicit_pdb = None if per_device_batch == "auto" else int(per_device_batch)
    explicit_ga = None if grad_accum == "auto" else int(grad_accum)
    backend = IndicParlerLoraBackend(pp, IndicParlerLoraConfig(
        project_root=pp,
        profile=profile,
        adapter_name=adapter_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        dtype=dtype,
        speaker_name=speaker_name,
        smoke=smoke,
        overwrite_output_dir=overwrite_export,
        explicit_per_device_batch=explicit_pdb,
        explicit_grad_accum=explicit_ga,
    ))
    try:
        backend._train(smoke=smoke, overwrite_export=overwrite_export)
    except Exception as e:
        _echo_err(f"LoRA training failed: {e}")
        raise
    _echo_ok(f"LoRA training finished for adapter '{adapter_name}'.")
    _echo_info(f"Export with: hindi-tts-builder voice-export {project} --adapter-name {adapter_name}")


@cli.command(name="voice-export")
@click.argument("project")
@click.option("--adapter-name", required=True)
def voice_export_cmd(project: str, adapter_name: str):
    """Bundle the trained LoRA checkpoint into the per-project voice library."""
    from hindi_tts_builder.train.backends.indic_parler_lora import (
        IndicParlerLoraBackend, IndicParlerLoraConfig,
    )
    pp = _project_path(project)
    backend = IndicParlerLoraBackend(pp, IndicParlerLoraConfig(
        project_root=pp,
        adapter_name=adapter_name,
        # profile/etc filled with defaults; export only needs adapter_name + run_root
    ))
    out = backend.export_engine()
    _echo_ok(f"Adapter exported to {out}")
    _echo_info(f"Add to global library with: hindi-tts-builder voice-add {out} --library-dir <library>")


@cli.command(name="voice-add")
@click.argument("adapter_dir", type=click.Path(path_type=Path, exists=True))
@click.option("--library-dir", type=click.Path(path_type=Path), default=None,
              help="Voice library root (default: <cwd>/voice_library/)")
@click.option("--name", "voice_name", default=None,
              help="Override the voice name (default: adapter dir basename).")
def voice_add_cmd(adapter_dir: Path, library_dir: Path | None, voice_name: str | None):
    """Register an adapter directory into the voice library."""
    from hindi_tts_builder.inference.backends.voice_library import VoiceLibrary
    lib_dir = library_dir or _default_library_dir()
    # Touch the library structure without loading the base model
    (lib_dir / "adapters").mkdir(parents=True, exist_ok=True)
    import shutil
    name = voice_name or adapter_dir.name
    dst = lib_dir / "adapters" / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(adapter_dir, dst)
    _echo_ok(f"Voice '{name}' added to library {lib_dir}")


@cli.command(name="voice-list")
@click.option("--library-dir", type=click.Path(path_type=Path), default=None)
def voice_list_cmd(library_dir: Path | None):
    """List voices available in the library."""
    lib_dir = library_dir or _default_library_dir()
    adapters = lib_dir / "adapters"
    if not adapters.exists():
        _echo_info(f"No library at {lib_dir}.")
        return
    voices = sorted(p for p in adapters.iterdir() if p.is_dir())
    if not voices:
        _echo_info(f"Library empty: {lib_dir}")
        return
    _echo_ok(f"Voices in {lib_dir}:")
    for v in voices:
        card_path = v / "voice_card.json"
        info = ""
        if card_path.exists():
            try:
                card = json.loads(card_path.read_text(encoding="utf-8"))
                info = (f"  rank={card.get('lora_rank', '?')} "
                        f"steps={card.get('training_steps', '?')} "
                        f"profile={card.get('training_profile', '?')}")
            except Exception:
                pass
        _echo_info(f"  {v.name}{info}")


@cli.command(name="voice-remove")
@click.argument("voice_name")
@click.option("--library-dir", type=click.Path(path_type=Path), default=None)
def voice_remove_cmd(voice_name: str, library_dir: Path | None):
    """Delete a voice from the library."""
    import shutil
    lib_dir = library_dir or _default_library_dir()
    adapter_dir = lib_dir / "adapters" / voice_name
    if not adapter_dir.exists():
        _echo_err(f"Voice '{voice_name}' not in {lib_dir}/adapters/")
        sys.exit(1)
    shutil.rmtree(adapter_dir)
    _echo_ok(f"Removed voice '{voice_name}' from {lib_dir}")


@cli.command(name="speak-voice")
@click.argument("text")
@click.option("--voice", required=True, help="Voice name from the library.")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--library-dir", type=click.Path(path_type=Path), default=None)
@click.option("--description", default=None,
              help="Override the voice card's default description.")
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--dtype", type=click.Choice(["auto", "bfloat16", "float16", "float32"]),
              default="auto", show_default=True)
def speak_voice_cmd(text: str, voice: str, out_path: Path, library_dir: Path | None,
                    description: str | None, temperature: float, dtype: str):
    """Render text with a specific voice from the library."""
    try:
        from hindi_tts_builder.inference.backends.voice_library import VoiceLibrary
    except ImportError as e:
        if "parler_tts" in str(e) or "peft" in str(e):
            _echo_err(
                "parler_tts/peft not importable in this venv. Run from the parler venv:\n"
                f"    /root/parler-venv/bin/python3 -m hindi_tts_builder.cli.main "
                f"speak-voice {text!r} --voice {voice} --out {out_path}"
            )
            sys.exit(1)
        raise
    lib_dir = library_dir or _default_library_dir()
    lib = VoiceLibrary(library_dir=lib_dir, dtype=dtype)
    if voice not in lib.list_voices():
        _echo_err(f"Voice '{voice}' not in library. Available: {lib.list_voices()}")
        sys.exit(1)
    final = lib.speak_to_file(text, out_path, voice=voice,
                               description=description, temperature=temperature)
    _echo_ok(f"Wrote {final}")


# =========================================================================
# corpus-ingest (bulk ingest + watcher for large 50-150h corpora)
# =========================================================================
@cli.command(name="corpus-ingest")
@click.argument("name")
@click.option("--staging-dir", required=True, type=click.Path(path_type=Path),
              help="Directory containing (id)/(media,srt) pairs or flat media+srt files.")
@click.option("--watch", is_flag=True,
              help="Long-running watch mode — polls every --poll-sec and picks up new pairs.")
@click.option("--poll-sec", type=float, default=30.0, show_default=True,
              help="Watch-mode poll interval in seconds.")
@click.option("--no-rebuild-csvs", is_flag=True,
              help="Skip the train/val/test CSV rebuild after ingest (faster; do it once at the end).")
def corpus_ingest_cmd(name: str, staging_dir: Path, watch: bool, poll_sec: float,
                      no_rebuild_csvs: bool):
    """Bulk-ingest a large Hindi corpus (50-150h+) into a project.

    Drop (media, srt) pairs into --staging-dir; this command finds them,
    extracts audio, segments by SRT timestamps (trust-srt mode — no
    WhisperX), validates clips, and appends to the project's metadata.

    CPU-only — safe to run alongside an active GPU training job.

    Layout examples:

      staging/ep001/ep001.mp4 + staging/ep001/ep001.srt   (nested)
      staging/ep001.mp4 + staging/ep001.srt                (flat)

    Use --watch to keep ingesting new files as you drop them in.
    """
    from hindi_tts_builder.data.corpus_ingest import ingest_directory, watch_directory
    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist. Create it first: "
                  f"hindi-tts-builder new {name}")
        sys.exit(1)
    if not staging_dir.exists():
        _echo_err(f"Staging dir not found: {staging_dir}")
        sys.exit(1)
    rebuild = not no_rebuild_csvs

    if watch:
        _echo_ok(f"watching {staging_dir} every {poll_sec:.0f}s — Ctrl-C to stop")
        try:
            for summary in watch_directory(
                pp, staging_dir,
                poll_interval_sec=poll_sec,
                rebuild_training_set_after=rebuild,
            ):
                _echo_info(
                    f"pass: +{summary.sources_processed} new sources, "
                    f"+{summary.total_clips_added} clips, "
                    f"+{summary.total_hours_added:.2f}h "
                    f"(skipped {summary.sources_skipped})"
                )
        except KeyboardInterrupt:
            _echo_ok("watcher stopped")
        return

    summary = ingest_directory(pp, staging_dir, rebuild_training_set_after=rebuild)
    _echo_ok(f"ingested {summary.sources_processed} sources "
             f"(+{summary.total_clips_added} clips, +{summary.total_hours_added:.2f}h)")
    _echo_info(f"skipped: {summary.sources_skipped}")
    for r in summary.per_source[:20]:
        if r.skipped_reason:
            _echo_info(f"  {r.source_id}: SKIP ({r.skipped_reason})")
        else:
            _echo_info(f"  {r.source_id}: +{r.accepted} clips ({r.hours:.2f}h), {r.rejected} rejected")
    if len(summary.per_source) > 20:
        _echo_info(f"  ... and {len(summary.per_source) - 20} more")


# =========================================================================
# audit
# =========================================================================
@cli.command()
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of human-readable.")
@click.option("--sample-n", type=int, default=30, show_default=True,
              help="Number of clips to sample for loudness + speaker-consistency checks.")
def audit(name: str, as_json: bool, sample_n: int):
    """Pre-training dataset audit. Catches silent-collapse bugs before training.

    Reports total clean hours, accept/reject reasons, duration distribution,
    loudness range, text-char coverage vs CharactersConfig, and a speaker-
    consistency proxy. Read-only; safe to run anytime.
    """
    from hindi_tts_builder.data.audit import audit_project, print_report
    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    try:
        report = audit_project(pp, sample_n_for_speaker=sample_n)
    except Exception as e:
        _echo_err(f"Audit failed: {e}")
        raise
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)


# =========================================================================
# make-profile (staged dataset subsampling)
# =========================================================================
@cli.command(name="make-profile")
@click.argument("name")
@click.argument("profile", type=click.Choice(["2h_clean", "10h_clean", "30h_clean", "60h_clean", "100h_clean"]))
@click.option("--seed", type=int, default=1234, show_default=True,
              help="RNG seed for reproducible sampling.")
def make_profile(name: str, profile: str, seed: int):
    """Materialise a staged dataset profile (e.g. 30h_clean) by subsampling
    the project's full train/val/test CSVs source-stratified.

    Output goes to `training_set/{train,val,test}_<profile>.csv` alongside
    the originals. Re-run with the same seed to reproduce the same sample.
    """
    from hindi_tts_builder.data.profiles import make_profile as _make
    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)
    try:
        summary = _make(pp, profile, seed=seed)
    except KeyError as e:
        _echo_err(str(e))
        sys.exit(1)
    except Exception as e:
        _echo_err(f"Profile build failed: {e}")
        raise
    _echo_ok(f"Profile '{summary['profile']}' ready (target {summary['target_hours']}h)")
    for split, info in summary["splits"].items():
        _echo_info(f"  {split}: {info.get('hours', 0.0)}h, {info.get('rows', 0)} rows -> {info.get('out', '<skipped>')}")


# =========================================================================
# train
# =========================================================================
@cli.command()
@click.argument("name")
@click.option("--prepare-only", is_flag=True, help="Only fit tokenizer and validate data; do not start training.")
@click.option("--backend", type=click.Choice(["coqui-vits", "indic-parler", "f5-hindi"]),
              default="coqui-vits", show_default=True,
              help="Trainer backend. coqui-vits is the current from-scratch path; "
              "indic-parler / f5-hindi are fine-tune adapters (v5).")
@click.option("--skip-gate", is_flag=True,
              help="Train despite pre-train gate blockers. You are choosing to spend GPU "
                   "time on a corpus with a known defect.")
def train(name: str, prepare_only: bool, backend: str, skip_gate: bool):
    """Train on the project's prepared data. Resumable. Backend-selectable."""
    from hindi_tts_builder.train.backends import get_backend
    pp = _project_path(name)
    if not pp.exists():
        _echo_err(f"Project '{name}' does not exist.")
        sys.exit(1)

    # The gate exists because two full training runs were spent on a corpus whose
    # defects were all measurable beforehand.
    try:
        from hindi_tts_builder.data.gate import check_corpus
        from hindi_tts_builder.utils.project import load_config

        res = check_corpus(pp, force=skip_gate, **dict(load_config(pp)["gate"]))
        if res.blockers:
            _echo_err("Pre-train gate FAILED — refusing to start training:")
            click.echo(res.describe())
            _echo_info("Fix the blockers, or re-run with --skip-gate to override.")
            sys.exit(1)
        for w in res.warnings:
            _echo_info(f"gate warning: {w}")
    except SystemExit:
        raise
    except Exception as e:
        # Never fall through to the trainer. A typo in the gate config used to
        # raise here, print one informational line, and start training anyway —
        # the safety net silently removed by a misspelled key.
        _echo_err(f"Pre-train gate could not be evaluated: {type(e).__name__}: {e}")
        _echo_info("Fix config.yaml's `gate:` section, or re-run with --skip-gate to override.")
        sys.exit(1)

    try:
        Backend = get_backend(backend)
        trainer = Backend(pp)
        ready = trainer.prepare()
        _echo_ok(f"Training prepared (backend={backend})")
        for k, v in ready.items():
            _echo_info(f"{k}: {v}")
        if prepare_only:
            return
        trainer.train()
        _echo_ok("Training complete")
    except ImportError as e:
        _echo_err(str(e))
        sys.exit(1)
    except NotImplementedError as e:
        _echo_err(f"Backend '{backend}' is a stub — see ROADMAP.md v5: {e}")
        sys.exit(2)
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
# Indic-Parler track (the v5 pinnacle sprint)
# Each command is a thin click wrapper around IndicParlerBackend so the
# generic `train --backend indic-parler` route stays in sync with these.
# =========================================================================
@cli.command(name="export-indic-parler")
@click.argument("project")
@click.option("--profile", default="10h_clean", show_default=True,
              type=click.Choice(["2h_clean", "10h_clean", "30h_clean", "60h_clean", "100h_clean"]))
@click.option("--overwrite", is_flag=True, help="Wipe existing export dir before writing.")
@click.option("--speaker-name", default="Divya", show_default=True)
def export_indic_parler(project: str, profile: str, overwrite: bool, speaker_name: str):
    """Export an existing project profile into a local Indic-Parler HF dataset."""
    from hindi_tts_builder.train.backends.indic_parler import (
        IndicParlerBackend, IndicParlerBackendConfig,
    )
    pp = _project_path(project)
    if not pp.exists():
        _echo_err(f"Project '{project}' does not exist.")
        sys.exit(1)
    backend = IndicParlerBackend(pp, IndicParlerBackendConfig(
        project_root=pp, profile=profile, speaker_name=speaker_name,
    ))
    try:
        out = backend.export_dataset(overwrite=overwrite)
    except FileExistsError as e:
        _echo_err(f"{e} (re-run with --overwrite to wipe)")
        sys.exit(1)
    _echo_ok(f"Indic-Parler dataset exported: {out}")


@cli.command(name="train-indic-parler")
@click.argument("project")
@click.option("--profile", default="10h_clean", show_default=True,
              type=click.Choice(["2h_clean", "10h_clean", "30h_clean", "60h_clean", "100h_clean"]))
@click.option("--smoke", is_flag=True, help="Tiny 64-sample smoke train (~50 steps); proves wiring works.")
@click.option("--max-steps", default=10_000, show_default=True)
@click.option("--lr", "learning_rate", default=1e-5, show_default=True, type=float)
@click.option("--dtype", default="float16", show_default=True,
              type=click.Choice(["float16", "bfloat16", "float32"]))
@click.option("--overwrite-export", is_flag=True, help="Re-export dataset even if one exists.")
@click.option("--speaker-name", default="Divya", show_default=True)
def train_indic_parler(project: str, profile: str, smoke: bool, max_steps: int,
                       learning_rate: float, dtype: str, overwrite_export: bool,
                       speaker_name: str):
    """Train Indic-Parler on one profile (2h → 10h → 30h → 60h → 100h)."""
    from hindi_tts_builder.train.backends.indic_parler import (
        IndicParlerBackend, IndicParlerBackendConfig,
    )
    pp = _project_path(project)
    if not pp.exists():
        _echo_err(f"Project '{project}' does not exist.")
        sys.exit(1)
    backend = IndicParlerBackend(pp, IndicParlerBackendConfig(
        project_root=pp,
        profile=profile,
        max_steps=max_steps,
        learning_rate=learning_rate,
        dtype=dtype,
        speaker_name=speaker_name,
        smoke=smoke,
    ))
    try:
        backend._train(smoke=smoke, overwrite_export=overwrite_export)
    except Exception as e:
        _echo_err(f"Training failed: {e}")
        raise
    _echo_ok("Indic-Parler training finished.")


@cli.command(name="export-indic-parler-engine")
@click.argument("project")
@click.option("--profile", default="10h_clean", show_default=True,
              type=click.Choice(["2h_clean", "10h_clean", "30h_clean", "60h_clean", "100h_clean"]))
def export_indic_parler_engine(project: str, profile: str):
    """Export the latest Indic-Parler checkpoint as an engine bundle."""
    from hindi_tts_builder.train.backends.indic_parler import (
        IndicParlerBackend, IndicParlerBackendConfig,
    )
    pp = _project_path(project)
    if not pp.exists():
        _echo_err(f"Project '{project}' does not exist.")
        sys.exit(1)
    backend = IndicParlerBackend(pp, IndicParlerBackendConfig(
        project_root=pp, profile=profile,
    ))
    try:
        out = backend._export_engine()
    except FileNotFoundError as e:
        _echo_err(str(e))
        sys.exit(1)
    _echo_ok(f"Indic-Parler engine exported: {out}")


@cli.command(name="speak-indic-parler")
@click.argument("project")
@click.argument("text")
@click.option("--profile", default="10h_clean", show_default=True,
              type=click.Choice(["2h_clean", "10h_clean", "30h_clean", "60h_clean", "100h_clean"]))
@click.option("--out", "out_path", default="out.wav", show_default=True,
              type=click.Path(dir_okay=False, path_type=Path))
@click.option("--description", default=None,
              help="Override the engine's default voice description (e.g. for style A/B).")
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--compile/--no-compile", "compile_model", default=False,
              help="torch.compile the model. First call is slow (~1 min) but subsequent calls are 1.5-2x faster.")
@click.option("--dtype", type=click.Choice(["auto", "bfloat16", "float16", "float32"]),
              default="auto", show_default=True,
              help="Model dtype. 'auto' picks bf16 on Ampere+ GPUs, fp32 on CPU.")
def speak_indic_parler(project: str, text: str, profile: str, out_path: Path,
                       description: str | None, temperature: float,
                       compile_model: bool, dtype: str):
    """Generate speech with an exported Indic-Parler engine.

    Must be run from the parler venv (parler-tts hard-pins transformers==4.46.1
    which conflicts with coqui-tts). Default install path:
        /root/parler-venv/bin/python3 -m hindi_tts_builder.cli.main speak-indic-parler ...
    """
    try:
        from hindi_tts_builder.inference.backends.indic_parler import IndicParlerEngine
    except ImportError as e:
        if "parler_tts" in str(e):
            _echo_err(
                "parler_tts not importable in this venv. Run from the parler venv:\n"
                "    /root/parler-venv/bin/python3 -m hindi_tts_builder.cli.main "
                f"speak-indic-parler {project} {text!r} --profile {profile} --out {out_path}"
            )
            sys.exit(1)
        raise
    pp = _project_path(project)
    engine_dir = pp / "engines" / "indic_parler" / profile
    if not engine_dir.exists():
        _echo_err(f"No engine at {engine_dir}. Run `export-indic-parler-engine` first.")
        sys.exit(1)
    eng = IndicParlerEngine.load(engine_dir, compile_model=compile_model, dtype=dtype)
    final = eng.speak_to_file(text, out_path, description=description, temperature=temperature)
    _echo_ok(f"Wrote {final}")


# =========================================================================
# eval-long (long-form regression detector)
# =========================================================================
@cli.command(name="eval-long")
@click.argument("name")
@click.option("--srt", "srt_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="SRT file to render and audit.")
@click.option("--out-dir", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Where to write final_mix.wav, failed_cues.csv, duration_mismatch.csv, cer_report.csv.")
@click.option("--cer-threshold", type=float, default=0.20, show_default=True,
              help="Per-cue CER above which the cue is marked failed.")
@click.option("--duration-tolerance", type=float, default=0.30, show_default=True,
              help="Duration mismatch tolerance, e.g. 0.30 = 30% drift either way.")
def eval_long(name: str, srt_path: Path, out_dir: Path, cer_threshold: float, duration_tolerance: float):
    """Long-form eval: render SRT, then audit per cue. Flags failures for
    manual review. A model that sounds great on 10s clips can still fall
    apart in a 30-minute video — this command catches that."""
    import csv as _csv
    from hindi_tts_builder.inference.engine import TTSEngine
    from hindi_tts_builder.inference.srt_renderer import SRTRenderer
    from hindi_tts_builder.utils.srt import parse_srt

    pp = _project_path(name)
    engine_dir = pp / "engine"
    if not engine_dir.exists():
        _echo_err(f"No engine at {engine_dir}. Run `export` first.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    final_mix = out_dir / "final_mix.wav"
    failed_csv = out_dir / "failed_cues.csv"
    dur_mismatch_csv = out_dir / "duration_mismatch.csv"
    cer_csv = out_dir / "cer_report.csv"

    cues = parse_srt(srt_path)
    engine = TTSEngine.load(engine_dir, cer_threshold=cer_threshold)
    renderer = SRTRenderer(engine, mode="natural", gap_ms_between_cues=200)

    failed_rows = []
    dur_rows = []
    cer_rows = []

    def _cb(p):
        # p exposes per-cue cer + cue duration vs synth duration
        cue = cues[p.cue_index - 1] if p.cue_index - 1 < len(cues) else None
        cue_dur = cue.duration if cue else None
        synth_dur = getattr(p, "synth_duration_sec", None)
        cer = getattr(p, "cer", None)
        cer_rows.append({"cue_index": p.cue_index, "text": p.cue_text, "cer": cer if cer is not None else ""})
        if cer is not None and cer > cer_threshold:
            failed_rows.append({"cue_index": p.cue_index, "text": p.cue_text,
                                "reason": f"cer>{cer_threshold}", "cer": round(cer, 3)})
        if cue_dur and synth_dur:
            drift = abs(synth_dur - cue_dur) / cue_dur
            if drift > duration_tolerance:
                dur_rows.append({"cue_index": p.cue_index, "text": p.cue_text,
                                 "cue_seconds": round(cue_dur, 3),
                                 "synth_seconds": round(synth_dur, 3),
                                 "drift_pct": round(drift * 100, 1)})
        if p.cue_index % 25 == 0 or p.cue_index == p.total_cues:
            click.echo(f"  [{p.cue_index}/{p.total_cues}] cer={cer if cer else '-'}")

    summary = renderer.render(srt_path, final_mix, progress_callback=_cb)

    def _write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(failed_csv, failed_rows)
    _write_csv(dur_mismatch_csv, dur_rows)
    _write_csv(cer_csv, cer_rows)

    _echo_ok(f"Wrote {final_mix}")
    _echo_info(f"cues={summary['cues_rendered']} duration={summary['duration_sec']:.1f}s")
    _echo_info(f"failed_cues: {len(failed_rows)}  duration_mismatch: {len(dur_rows)}")
    _echo_info(f"  reports -> {failed_csv.name}, {dur_mismatch_csv.name}, {cer_csv.name}")


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
