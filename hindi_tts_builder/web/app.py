"""Training-launcher FastAPI app.

Single-page UI plus a JSON+SSE API:

    GET  /                               HTML page
    GET  /api/projects                   list known projects
    POST /api/projects                   create project + add sources (multipart)
    GET  /api/projects/{name}            project summary (config + manifest counts)
    POST /api/projects/{name}/start      kick off prepare + train
    POST /api/projects/{name}/stop       terminate the running job
    GET  /api/projects/{name}/status     job state
    GET  /api/projects/{name}/logs       SSE stream of the run log
"""
from pathlib import Path
from typing import List, Optional
import asyncio
import json

from hindi_tts_builder.web.jobs import JobRegistry


def create_app(projects_root: Path):
    try:
        from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
        from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    except ImportError as e:
        raise ImportError(
            "fastapi required for the studio UI. Install: pip install fastapi uvicorn python-multipart"
        ) from e

    projects_root = Path(projects_root).resolve()
    projects_root.mkdir(parents=True, exist_ok=True)
    registry = JobRegistry()

    # If the studio was just restarted while a pipeline is mid-flight, the
    # in-memory registry is empty — but the prepare/train subprocess is
    # still running orphaned. Re-attach so the UI shows the right state.
    n = registry.reattach_orphans(projects_root)
    if n:
        print(f"[studio] re-attached {n} orphan job(s) from previous studio session", flush=True)

    app = FastAPI(
        title="Hindi TTS Builder — Training Studio",
        version="1.0.0",
        description="Paste YouTube URLs, upload SRT transcripts, and start training.",
    )

    template_path = Path(__file__).parent / "templates" / "index.html"

    # =========================================================================
    # HTML
    # =========================================================================
    @app.get("/", response_class=HTMLResponse)
    def index():
        return template_path.read_text(encoding="utf-8")

    # =========================================================================
    # Projects: list + create
    # =========================================================================
    @app.get("/api/projects")
    def list_projects():
        items = []
        for p in sorted(projects_root.iterdir()) if projects_root.exists() else []:
            if not p.is_dir():
                continue
            cfg_file = p / "config.yaml"
            if not cfg_file.exists():
                continue
            items.append(_project_summary(p, registry))
        return {"projects": items}

    @app.post("/api/projects")
    async def create_project(
        name: str = Form(...),
        urls: str = Form(...),
        srt_files: List[UploadFile] = File(...),
    ):
        import shutil
        import sys
        import traceback
        from hindi_tts_builder.utils.project import create_project as create_proj
        from hindi_tts_builder.data.pipeline import add_sources_from_files

        name = name.strip()
        if not name or not _is_safe_name(name):
            raise HTTPException(400, "name must be alphanumeric/underscore/hyphen, non-empty")
        url_lines = [u.strip() for u in urls.splitlines() if u.strip() and not u.strip().startswith("#")]
        if not url_lines:
            raise HTTPException(400, "at least one YouTube URL is required")
        if len(url_lines) != len(srt_files):
            raise HTTPException(
                400,
                f"URL/SRT count mismatch: {len(url_lines)} URLs vs {len(srt_files)} SRT uploads. "
                "Upload one .srt per URL, in the same order.",
            )
        for f in srt_files:
            if not (f.filename or "").lower().endswith(".srt"):
                raise HTTPException(400, f"file '{f.filename}' is not a .srt")

        proj_dir = projects_root / name
        if proj_dir.exists():
            raise HTTPException(409, f"project '{name}' already exists")

        # Wrap the whole creation flow so unexpected errors come back as 500s
        # with a useful message instead of a bare "Internal Server Error".
        try:
            paths = create_proj(projects_root, name)

            staging = paths.root / "_staging"
            staging.mkdir(parents=True, exist_ok=True)
            urls_file = staging / "urls.txt"
            urls_file.write_text("\n".join(url_lines) + "\n", encoding="utf-8")
            for i, f in enumerate(srt_files):
                data = await f.read()
                target = staging / f"{i:05d}_{_safe_filename(f.filename or f'cue_{i}.srt')}"
                target.write_bytes(data)

            try:
                added = add_sources_from_files(paths, urls_file, staging)
            finally:
                shutil.rmtree(staging, ignore_errors=True)

            return {"name": name, "added": added, "summary": _project_summary(proj_dir, registry)}
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            shutil.rmtree(proj_dir, ignore_errors=True)
            raise HTTPException(500, f"{type(e).__name__}: {e}")

    @app.get("/api/projects/{name}")
    def get_project(name: str):
        p = projects_root / name
        if not (p / "config.yaml").exists():
            raise HTTPException(404, f"no project '{name}'")
        return _project_summary(p, registry)

    # =========================================================================
    # Pipeline control
    # =========================================================================
    @app.post("/api/projects/{name}/start")
    def start_pipeline(name: str, payload: Optional[dict] = None):
        p = projects_root / name
        if not (p / "config.yaml").exists():
            raise HTTPException(404, f"no project '{name}'")
        opts = payload or {}
        try:
            state = registry.start_pipeline(
                name,
                projects_root,
                skip_train=bool(opts.get("skip_train", False)),
                no_whisperx=bool(opts.get("no_whisperx", False)),
                no_whisper_qc=bool(opts.get("no_whisper_qc", False)),
                skip_qc=bool(opts.get("skip_qc", False)),
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return state.to_dict()

    @app.post("/api/projects/{name}/stop")
    def stop_pipeline(name: str):
        ok = registry.stop(name)
        return {"signalled": ok}

    _AVATAR_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")
    _AVATAR_MEDIA = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                     "webp": "image/webp", "gif": "image/gif", "svg": "image/svg+xml"}

    def _avatar_dir(proj_dir: Path) -> Path:
        return proj_dir / "avatars"

    def _list_avatars(proj_dir: Path) -> list:
        """Return sorted list of avatar files in the project's avatars/ dir."""
        d = _avatar_dir(proj_dir)
        if not d.exists():
            return []
        return sorted(p for p in d.iterdir()
                      if p.is_file() and p.suffix.lower() in _AVATAR_EXTS)

    @app.post("/api/projects/{name}/avatar")
    async def upload_avatars(name: str, files: list[UploadFile] = File(...)):
        """Upload one or more avatar images. Stored as
        projects/<name>/avatars/avatar_00.<ext> ... avatar_NN.<ext> in upload
        order. Multiple images become "maturity stages" — the voice card
        shows image N where N = floor(maturity * len/100). Single upload
        works the same way (acts as both first and only stage).

        Replaces any previously uploaded avatars.
        """
        proj_dir = projects_root / name
        if not (proj_dir / "config.yaml").exists():
            raise HTTPException(404, f"no project '{name}'")
        if not files:
            raise HTTPException(400, "no files in upload")
        # Validate all extensions before we wipe the old ones
        for f in files:
            ext = Path(f.filename or "").suffix.lower()
            if ext not in _AVATAR_EXTS:
                raise HTTPException(400, f"unsupported image type {ext!r} for {f.filename!r}; use png/jpg/webp/gif/svg")
        # Reset existing avatars (both the new dir and any legacy avatar.*)
        adir = _avatar_dir(proj_dir)
        if adir.exists():
            for old in adir.iterdir():
                try: old.unlink()
                except OSError: pass
        for legacy in proj_dir.glob("avatar.*"):
            try: legacy.unlink()
            except OSError: pass
        adir.mkdir(parents=True, exist_ok=True)
        # Save each
        saved = []
        for i, f in enumerate(files):
            ext = Path(f.filename or "").suffix.lower()
            data = await f.read()
            if len(data) > 8 * 1024 * 1024:
                raise HTTPException(400, f"{f.filename!r} > 8 MiB")
            target = adir / f"avatar_{i:02d}{ext}"
            target.write_bytes(data)
            saved.append({"index": i, "filename": target.name, "bytes": len(data)})
        return {"name": name, "count": len(saved), "stages": saved}

    @app.get("/api/projects/{name}/avatar")
    def get_avatar(name: str, stage: int | None = None):
        """Return the avatar image. With ?stage=N returns the N-th uploaded
        image (clamped to the available range). Without it returns stage 0
        (or the legacy single avatar if any). 404 if no avatar uploaded."""
        from fastapi.responses import FileResponse
        proj_dir = projects_root / name
        files = _list_avatars(proj_dir)
        if files:
            if stage is None:
                stage = 0
            stage = max(0, min(len(files) - 1, int(stage)))
            p = files[stage]
            media = _AVATAR_MEDIA[p.suffix.lstrip(".").lower()]
            return FileResponse(p, media_type=media)
        # Legacy single-file fallback
        for ext in _AVATAR_EXTS:
            p = proj_dir / f"avatar{ext}"
            if p.exists():
                return FileResponse(p, media_type=_AVATAR_MEDIA[ext.lstrip(".")])
        raise HTTPException(404, "no avatar uploaded")

    @app.delete("/api/projects/{name}/avatar")
    def delete_avatar(name: str):
        proj_dir = projects_root / name
        removed = 0
        adir = _avatar_dir(proj_dir)
        if adir.exists():
            for p in adir.iterdir():
                try: p.unlink(); removed += 1
                except OSError: pass
        for legacy in proj_dir.glob("avatar.*"):
            try: legacy.unlink(); removed += 1
            except OSError: pass
        return {"removed": removed}

    @app.get("/api/projects/{name}/status")
    def status(name: str):
        st = registry.get(name)
        if st is None:
            return {"running": False, "state": None}
        return {"running": st.running, "state": st.to_dict()}

    @app.get("/api/projects/{name}/logs")
    async def stream_logs(name: str, request: Request, tail: int = 200):
        """Stream live log lines via SSE.

        Replays only the last `tail` lines on connect (not the whole file)
        so a 3 MB log doesn't pin uvicorn for several seconds on every page
        load and starve other API requests. Then follows the file like
        `tail -f`. Pass ?tail=0 to start strictly from the end, or
        ?tail=-1 to replay the entire log (legacy behavior).
        """
        st = registry.get(name)
        log_path = st.log_path if st else (projects_root / name / "logs" / "studio_run.log")
        if not log_path.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch()

        async def event_gen():
            # Send the last N lines synchronously up front, then seek to end
            # and follow like tail -f. Reading via tell()/seek() avoids
            # re-streaming everything on every reconnect.
            with open(log_path, "rb") as fp:
                if tail and tail != 0:
                    # Read the whole file (we accept a one-time cost) then
                    # keep only the last `tail` lines. -1 = unlimited.
                    data = fp.read()
                    lines = data.decode("utf-8", errors="replace").splitlines()
                    if tail > 0:
                        lines = lines[-tail:]
                    for line in lines:
                        yield f"data: {json.dumps(line)}\n\n"
                # fp is now at end-of-file; loop and tail new bytes
                while True:
                    if await request.is_disconnected():
                        break
                    chunk = fp.read(4096)
                    if chunk:
                        for line in chunk.decode("utf-8", errors="replace").splitlines():
                            yield f"data: {json.dumps(line)}\n\n"
                    else:
                        cur = registry.get(name)
                        if cur and not cur.running:
                            extra = fp.read()
                            if extra:
                                for line in extra.decode("utf-8", errors="replace").splitlines():
                                    yield f"data: {json.dumps(line)}\n\n"
                            yield f"event: done\ndata: {json.dumps(cur.to_dict())}\n\n"
                            break
                        await asyncio.sleep(0.5)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return app


def _is_safe_name(name: str) -> bool:
    return all(c.isalnum() or c in "-_" for c in name)


def _safe_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def _project_summary(p: Path, registry: JobRegistry) -> dict:
    import yaml
    cfg = {}
    try:
        cfg = yaml.safe_load((p / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        pass

    # max_steps from training_config.yaml (falls back to default 500k)
    train_cfg = {}
    try:
        train_cfg = yaml.safe_load((p / "training_config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    max_steps = int(train_cfg.get("max_steps", 500_000))

    manifest_file = p / "sources" / "manifest.json"
    sources_count = 0
    counts = {"downloaded": 0, "aligned": 0, "segmented": 0, "qc_passed": 0}
    if manifest_file.exists():
        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            srcs = data.get("sources", [])
            sources_count = len(srcs)
            for s in srcs:
                st = s.get("status", {})
                if st.get("downloaded"): counts["downloaded"] += 1
                if st.get("aligned"): counts["aligned"] += 1
                if st.get("segmented"): counts["segmented"] += 1
                if st.get("qc_passed"): counts["qc_passed"] += 1
        except Exception:
            pass

    job = registry.get(p.name)
    avatar_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")
    avatars_dir = p / "avatars"
    avatar_count = 0
    if avatars_dir.exists():
        avatar_count = sum(1 for f in avatars_dir.iterdir()
                           if f.is_file() and f.suffix.lower() in avatar_exts)
    if avatar_count == 0:
        # Legacy single-file path
        avatar_count = sum(1 for ext in avatar_exts if (p / f"avatar{ext}").exists())
    return {
        "name": p.name,
        "language": cfg.get("language"),
        "sample_rate": cfg.get("target_sample_rate"),
        "sources_count": sources_count,
        "stage_counts": counts,
        "max_steps": max_steps,
        "engine_exported": (p / "engine" / "manifest.json").exists(),
        "has_avatar": avatar_count > 0,
        "avatar_count": avatar_count,
        "job": job.to_dict() if job else None,
    }


def run_studio(projects_root, host: str = "127.0.0.1", port: int = 8770) -> None:
    try:
        import uvicorn  # type: ignore
    except ImportError as e:
        raise ImportError(
            "uvicorn required to run the studio. Install: pip install uvicorn[standard]"
        ) from e
    app = create_app(Path(projects_root))
    uvicorn.run(app, host=host, port=port)
