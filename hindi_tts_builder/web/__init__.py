"""Training-launcher web UI.

A small FastAPI app that lets you create a project, paste YouTube URLs,
upload matching SRT transcripts, and kick off the full prepare + train
pipeline from the browser. Logs stream back via Server-Sent Events.

Start it with:
    hindi-tts-builder studio --host 127.0.0.1 --port 8770
"""
