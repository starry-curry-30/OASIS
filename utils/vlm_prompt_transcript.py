"""
Unified VLM/LLM text transcript (append-only stream; images not stored).
Shared by NavRunner, post-rotation check, relocalization helpers, etc.
Env OASIS_VLM_PROMPT_TRANSCRIPT is set by oasis.core at session start and inherited by child processes.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_PROMPT_DIR = PROJECT_ROOT / "task_prompt"
ENV_KEY = "OASIS_VLM_PROMPT_TRANSCRIPT"


def get_path_from_env() -> Optional[str]:
    p = os.environ.get(ENV_KEY)
    return p if p else None


def init_session_header(path: str, inquiry: str) -> None:
    """Write session header for the first file in a run (Core calls once per navigation task)."""
    try:
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            f.write("=== OASIS VLM session (all LLM/VLM text; images not stored) ===\n")
            f.write(f"started: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"inquiry: {inquiry}\n\n")
            f.write("--- streaming log ---\n\n")
        print(f"[VLMTranscript] Session file: {path}", flush=True)
    except Exception as e:
        print(f"[VLMTranscript] Failed to init session header: {e}", file=sys.stderr, flush=True)


def init_standalone_nav_file(path: str, task: str, model: str) -> None:
    """Header for standalone NavRunner (no Core session)."""
    try:
        outp = Path(path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            f.write("=== OASIS Nav Task Transcript (text only; images not stored) ===\n")
            f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"task: {task}\n")
            f.write(f"model: {model}\n")
            f.write("\n--- streaming log ---\n\n")
        print(f"[VLMTranscript] Standalone transcript: {path}", flush=True)
    except Exception as e:
        print(f"[VLMTranscript] Failed to init standalone file: {e}", file=sys.stderr, flush=True)


def append_transcript(path: Optional[str], text: str) -> None:
    """Append a chunk and flush immediately."""
    if not path or not text:
        return
    try:
        with Path(path).open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")
            f.flush()
    except Exception as e:
        print(f"[VLMTranscript] Failed to append: {e}", file=sys.stderr, flush=True)
