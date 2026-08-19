#!/usr/bin/env bash
# Render full long-form through base AI4Bharat, then CER-eval.
set -e
set -x

cd /root/hindi-tts
/root/parler-venv/bin/python scripts/render_base_long_form.py
echo "==== CER ===="
/root/hindi-tts/venv/bin/python scripts/eval_base_cer.py
