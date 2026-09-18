#!/usr/bin/env bash
# Linux/macOS dev runner (Normal mode). Needs: pip install -r requirements.txt -r requirements-ocr.txt
cd "$(dirname "$0")/.." && exec python3 -m ragly_backend
