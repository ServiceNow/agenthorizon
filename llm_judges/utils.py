"""Shared utilities for LLM-as-Judge preprocessing and evaluation.

Provides:
    - Trajectory loading from JSONL
    - Image encoding (base64 for API payloads)
    - Token estimation
    - API key loading from dotenv files
    - Output format helpers compatible with Phase 1 analysis
"""

import base64
import io
import json
import os
import re
from pathlib import Path

from PIL import Image


# ── Token estimation ──────────────────────────────────────────────────────
# Rough heuristics for multimodal token budgets. These are intentionally
# conservative (over-estimate) so we stay within context limits.

# Text: ~4 chars per token (works for English, close enough for code)
CHARS_PER_TOKEN = 4

# Images: vision counters differ across providers. We use an area-based
# estimator (tokens scale smoothly with pixel area) rather than a discrete
# tile bucketing, so the auto-resolution binary search can find resolutions
# anywhere between 64 and 512 px. Calibrated against Gemini 3.1 Flash Lite,
# where a 512x332 JPEG costs ~1080 prompt tokens. Image_tokens =
# IMAGE_BASE_TOKENS + (w * h) / PIXELS_PER_TOKEN. This matches observed
# Gemini usage to within a few percent and stays a slight over-estimate for
# OpenAI / Gemma / Qwen-VL tokenizers, so payloads sized under the budget
# stay under each provider's real context window.
IMAGE_BASE_TOKENS = 85
PIXELS_PER_TOKEN = 170  # 512*332 / ~1000 -> hits ~1085 estimated at 512x332


def estimate_text_tokens(text: str) -> int:
    """Rough token estimate for a text string."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate tokens for an image based on its pixel dimensions.

    Area-based: IMAGE_BASE_TOKENS + (w * h) / PIXELS_PER_TOKEN. Calibrated
    against observed Gemini 3.1 Flash Lite usage (a 512x332 JPEG costs
    ~1080 prompt tokens).
    """
    return IMAGE_BASE_TOKENS + (width * height) // PIXELS_PER_TOKEN


def estimate_image_tokens_from_bytes(image_bytes: bytes) -> int:
    """Estimate tokens for an image from its raw bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    return estimate_image_tokens(img.width, img.height)


# ── Image processing ──────────────────────────────────────────────────────

def load_and_resize_image(
    path: str | Path,
    max_width: int = 512,
    max_height: int = 512,
    quality: int = 60,
) -> tuple[bytes, int, int]:
    """Load an image, resize to fit within max dimensions, return JPEG bytes.

    Returns:
        (jpeg_bytes, new_width, new_height)
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_width, max_height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), img.width, img.height


def image_to_base64(image_bytes: bytes) -> str:
    """Encode raw image bytes as a base64 string."""
    return base64.b64encode(image_bytes).decode("ascii")


def load_image_as_base64(
    path: str | Path,
    max_width: int | None = None,
    max_height: int | None = None,
    quality: int = 85,
) -> tuple[str, int, int]:
    """Load image, optionally resize, return (base64_str, width, height).

    If max_width/max_height are None, loads at original resolution.
    """
    if max_width is not None and max_height is not None:
        img_bytes, w, h = load_and_resize_image(
            path, max_width, max_height, quality
        )
    else:
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        img_bytes = buf.getvalue()
        w, h = img.width, img.height
    return image_to_base64(img_bytes), w, h


# ── Trajectory loading ────────────────────────────────────────────────────

def load_trajectories(jsonl_path: str | Path, limit: int | None = None) -> list[dict]:
    """Load trajectories from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file (e.g., data/standard/agenthorizon.jsonl)
        limit: If set, only load first N trajectories

    Returns:
        List of trajectory dicts
    """
    trajectories = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            trajectories.append(json.loads(line))
    return trajectories


def load_local_image_map(labels_path: str | Path) -> dict[str, str]:
    """Build a trajectory_id -> original_id map from the labels file.

    The original_id is the directory name under data/media/images/ where the
    screenshot files live. This is the only field we read from labels at
    preprocess time -- never the verdict-revealing fields. See
    CLAUDE.md "Standardized data format".
    """
    mapping: dict[str, str] = {}
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tid = d.get("trajectory_id")
            oid = d.get("original_id")
            if tid and oid:
                mapping[tid] = oid
    return mapping


def get_screenshot_path(
    step: dict,
    base_dir: str | Path = ".",
    local_image_id: str | None = None,
    local_image_root: str | Path = "data/media/images",
) -> Path | None:
    """Get the absolute screenshot path for a step.

    Two cases:
    1. local_image_id is provided (the recommended path): resolve to
       <base_dir>/<local_image_root>/<local_image_id>/step_<step_id+1>.png.
       Step files are 1-indexed on disk while step_id is 0-indexed in the
       JSONL, matching scripts/render_trajectories.py.
    2. step['screenshot'] is already a local relative path (legacy data
       layout). Resolve it under base_dir directly.
    """
    # Preferred path: use local_image_id from labels.
    if local_image_id is not None:
        step_id = step.get("step_id")
        if step_id is None:
            return None
        try:
            file_index = int(step_id) + 1
        except (TypeError, ValueError):
            file_index = step_id
        p = Path(base_dir) / local_image_root / local_image_id / f"step_{file_index}.png"
        if p.exists():
            return p
        return None

    # Legacy path: trust the screenshot field as a local relative path.
    screenshot = step.get("screenshot") or ""
    if not screenshot or screenshot.startswith("http"):
        return None
    p = Path(base_dir) / screenshot
    if p.exists():
        return p
    return None


def format_step_text(step: dict, step_idx: int | None = None) -> str:
    """Format a single step as text (no image).

    Mirrors the format in the markdown files used by Phase 1, using the same
    action formatting as render_trajectories.py.
    """
    if step_idx is None:
        step_idx = step.get("step_id", "?")

    action = step.get("action", {})
    action_type = action.get("type", "unknown")
    params = action.get("parameters", {})

    # Build action description (matching render_trajectories.py format)
    if action_type == "click":
        button = params.get("button", "left")
        x, y = params.get("x", "?"), params.get("y", "?")
        n = params.get("num_clicks", 1)
        click_str = f"{'right' if button == 'right' else 'left'}-click"
        suffix = f" x{n}" if n and n > 1 else ""
        action_str = f"{click_str} ({x}, {y}){suffix}"
    elif action_type == "type":
        text = params.get("text", "")
        if len(text) > 80:
            text = text[:77] + "..."
        action_str = f'type "{text}"'
    elif action_type == "press":
        keys = params.get("keys", [])
        action_str = f"press {' '.join(keys)}"
    elif action_type == "hotkey":
        keys = params.get("keys", [])
        action_str = f"hotkey {'+'.join(keys)}"
    elif action_type == "scroll":
        direction = params.get("direction", "?")
        amount = params.get("amount", "?")
        x, y = params.get("x", "?"), params.get("y", "?")
        action_str = f"scroll {direction} {amount}px at ({x}, {y})"
    elif action_type == "drag":
        sx = params.get("start_x", "?")
        sy = params.get("start_y", "?")
        ex = params.get("end_x", "?")
        ey = params.get("end_y", "?")
        action_str = f"drag ({sx}, {sy}) -> ({ex}, {ey})"
    elif action_type in ("key_down", "key_up"):
        key = params.get("key", "?")
        action_str = f"{action_type.replace('_', '')} {key}"
    else:
        action_str = action_type

    # Build step text
    lines = [
        f"### Step {step_idx}",
        f"**Action:** `{action_str}`",
    ]

    # Timestamp (stored as microseconds in JSONL)
    timestamp_us = step.get("timestamp_us")
    if timestamp_us is not None:
        try:
            ts_ms = int(timestamp_us) / 1000
            lines.append(f"**Timestamp:** {ts_ms:.0f} ms")
        except (ValueError, TypeError):
            pass

    return "\n".join(lines)


def build_instruction_text(trajectory: dict) -> str:
    """Extract the task instruction from a trajectory dict."""
    return trajectory["task"]["instruction"]


# ── API key loading ───────────────────────────────────────────────────────

def load_api_key(name: str) -> str:
    """Load an API key from the appropriate dotenv file.

    Supported names:
        OPENROUTER_API_KEY -- from ~/.claude/.env
        GEMINI_API_KEY     -- from ~/.gemini/.env
        VLLM_BEARER_TOKEN  -- from ~/.vllm/.env  (self-hosted vLLM cluster)

    Falls back to os.environ if the dotenv file doesn't exist.
    """
    dotenv_paths = {
        "OPENROUTER_API_KEY": Path.home() / ".claude" / ".env",
        "GEMINI_API_KEY": Path.home() / ".gemini" / ".env",
        "VLLM_BEARER_TOKEN": Path.home() / ".vllm" / ".env",
    }
    dotenv_path = dotenv_paths.get(name)
    if dotenv_path and dotenv_path.exists():
        with open(dotenv_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{name}="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value

    # Fallback to environment
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(
            f"API key {name} not found in {dotenv_path} or environment. "
            f"Please set it before running."
        )
    return value


# ── Output format ─────────────────────────────────────────────────────────

def parse_judge_response(response_text: str) -> dict:
    """Parse the judge's JSON response, stripping markdown fences if present.

    Returns the parsed dict, or a dict with 'raw_response' if parsing fails.
    """
    text = response_text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": text}


def save_result(
    result: dict,
    deliverable_id: str,
    output_dir: Path,
    meta: dict | None = None,
) -> Path:
    """Save a judge result in Phase 1-compatible format.

    The output JSON has:
        success: bool
        reasoning: str
        deliverable_id: str
        _meta: dict (optional)
    """
    result["deliverable_id"] = deliverable_id
    if meta:
        result["_meta"] = meta
    out_path = output_dir / f"{deliverable_id}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return out_path
