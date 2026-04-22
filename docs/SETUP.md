# Setup Guide — WSL2 on Windows 11

One-time setup takes about 30 minutes. After that, every session is just
`wsl` in a PowerShell window.

## 1. Install WSL2 + Ubuntu 22.04

Open PowerShell **as Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart when prompted. On first boot, Ubuntu asks you to pick a username and
password — these are for inside Linux only and don't need to match Windows.

Verify:

```powershell
wsl --list --verbose
```

You should see `Ubuntu-22.04` with `VERSION 2`.

## 2. Install the NVIDIA driver on Windows (not inside WSL)

WSL2 accesses your GPU through the Windows driver. Install the latest NVIDIA
Game Ready or Studio driver from nvidia.com. Do NOT install a separate
NVIDIA driver inside Ubuntu — that will break CUDA.

After driver install, open WSL and verify:

```bash
nvidia-smi
```

You should see your 12GB GPU. If the command is missing or errors, the
Windows driver is either too old or not installed correctly.

## 3. Install system dependencies inside WSL

```bash
sudo apt update
sudo apt install -y \
    python3.11 python3.11-venv python3-pip \
    ffmpeg \
    build-essential \
    git wget curl \
    sox libsndfile1
```

## 4. Clone this project

```bash
cd ~
git clone <your-repo-url> hindi-tts
cd hindi-tts
```

Or if you downloaded a zip, extract it into `~/hindi-tts/`.

## 5. Create Python virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The first install takes 10–15 minutes (torch + torchaudio are large).

## 6. Install the package in editable mode

```bash
pip install -e .
```

This makes `hindi-tts-builder` available as a command.

## 7. Verify GPU access from Python

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0, 'GB')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA GeForce RTX 3060 (or similar 12GB card)
VRAM: 12.0 GB
```

## 8. Run the test suite

```bash
pytest tests/ -v
```

All tests should pass. If any fail, something is wrong with the installation.

## Windows/Linux file paths

WSL2 can access your Windows files at `/mnt/c/Users/<you>/`. However:

- **For this project, keep files inside WSL2's native filesystem** (e.g. `~/hindi-tts/`). Native filesystem is 3–5× faster for the tens of thousands of small files this project produces.
- Copy your OneDrive transcripts into WSL2 once at the start. Don't try to train directly against files on `/mnt/c/`.

From Windows, you can browse WSL2 files in File Explorer via:
```
\\wsl$\Ubuntu-22.04\home\<your-wsl-username>\hindi-tts
```

## Common setup issues

**`nvidia-smi` not found inside WSL:** Update your Windows NVIDIA driver to the latest version. Driver must be a version that supports WSL2 (all drivers from 2021+ do).

**CUDA available: False in Python:** You installed `torch` without CUDA support. Reinstall:
```bash
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**`ffmpeg: command not found`:** `sudo apt install ffmpeg` inside WSL.

**Pip install takes forever:** Network-bound. Be patient; the torch wheel is ~2GB.
