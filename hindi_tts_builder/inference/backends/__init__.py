"""Inference backends.

The legacy `inference.engine.TTSEngine` covers Coqui-VITS. This package
holds backend-specific engines for non-VITS models. Imports are lazy so
loading one backend doesn't pull in the other's dependencies.
"""
