"""
Gemini multimodal VLM helpers (Gemini 3 Pro and compatible models).

Requires: pip install google-generativeai pillow

API key: GEMINI_API_KEY or GOOGLE_API_KEY
Model: GEMINI_VLM_MODEL (default: gemini-3-pro-preview)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def get_gemini_api_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def default_vlm_model() -> str:
    return (os.environ.get("GEMINI_VLM_MODEL") or "gemini-3-pro-preview").strip()


def generate_content_with_images(
    *,
    system_instruction: str,
    user_text: str,
    image_paths: List[str],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is required for Gemini VLM. Install: pip install google-generativeai"
        ) from exc
    from PIL import Image

    key = (api_key or get_gemini_api_key()).strip()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini VLM.")
    model_id = (model or default_vlm_model()).strip()
    genai.configure(api_key=key)
    model_obj = genai.GenerativeModel(model_id)
    combined_prompt = f"{system_instruction.rstrip()}\n\n---\n\n{user_text}"
    parts: List = [combined_prompt]
    for path in image_paths:
        parts.append(Image.open(Path(path)))
    response = model_obj.generate_content(parts)
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise RuntimeError("Empty response from Gemini VLM")
    return text.strip()
