"""Evaluate trajectory markdown files using an LLM judge CLI.

For each .md trajectory in --md-dir, spawns a judge subprocess (claude or gemini)
that reads the trajectory via stdin and writes a structured JSON evaluation.
Supports concurrent workers, dry-run mode, and automatic resumption
(existing outputs are skipped). At the end, writes a _summary.json and
generates success/failure CSVs in the results/ directory.

Usage (sandbox mode is the default):
    uv run python scripts/evaluate_trajectories.py \
        --harness claude \
        --model claude-opus-4-7 \
        --effort xhigh \
        --platform anthropic_console \
        --use-console-key \
        --workers 2

    # Gemini CLI harness:
    uv run python scripts/evaluate_trajectories.py \
        --harness gemini \
        --model gemini-3.1-pro-preview \
        --platform google_gemini_direct \
        --workers 4

    # Multi-split run (val + public_test + private_test in ONE exp_dir):
    # The UI / experiment-status skill still group rows per split via
    # trajectory_id → split lookup; this just keeps everything under one
    # exp_id instead of fragmenting across three.
    uv run python scripts/evaluate_trajectories.py \
        --harness opencode \
        --model qwen36_vllm/Qwen/Qwen3.6-27B \
        --splits val,public_test,private_test \
        --workers 4

    # Phased multi-split (val first as pilot, then resume into public_test):
    uv run python scripts/evaluate_trajectories.py --split val ...     # exp_id=X
    uv run python scripts/evaluate_trajectories.py --resume-exp X --split public_test ...

    # Legacy (--no-sandbox) flow:
    uv run python scripts/evaluate_trajectories.py \
        --no-sandbox \
        --md-dir data/standard/agenthorizon_md \
        --output-dir results/eval_agenthorizon_opus \
        --harness claude --model opus --workers 1

Output structure:
    <output-dir>/
        <deliverable_id>.json      # per-trajectory evaluation result
        sessions/<did>.jsonl       # session transcript (claude only)
        _summary.json              # run metadata (model, counts, elapsed time)
    results/successes/success_<suffix>.csv
    results/failures/failure_<suffix>.csv
"""

import argparse
import asyncio
import functools
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Console API spending cap (user directive 2026-04-17, see
# .claude/projects/.../memory/project_console_api_limit.md)
# Log location: project-root/console_usage.jsonl (visible, in the repo).
CONSOLE_USAGE_LOG = Path(__file__).resolve().parent.parent / "console_usage.jsonl"
CONSOLE_SPEND_CAP_USD = 2500.0  # 2026-04-29: hard cap on local ledger (no real-billing scaling). User directive: count what we count, abort at this number. Lifetime cap for sk-ant-...9d1ff4c2.
CONSOLE_SPEND_CHECK_INTERVAL_SEC = 300  # re-check cumulative spend every 5 minutes during a run

# Codex weekly rate-limit cap for --platform chatgpt_subscription runs.
# Codex session JSONLs under ~/.codex/sessions/ carry a token_count event whose
# rate_limits.secondary.used_percent reports the current 7-day-window usage
# (window_minutes = 10080). We abort batches once that crosses CODEX_WEEKLY_CAP_PERCENT
# to avoid bumping the ChatGPT subscription into a hard-lock state.
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
CODEX_STATE_DB = Path.home() / ".codex" / "state_5.sqlite"
CODEX_WEEKLY_CAP_PERCENT = 80.0
CODEX_WEEKLY_WARN_PERCENT = 80.0
CODEX_WEEKLY_CHECK_INTERVAL_SEC = 120  # re-check weekly usage every 2 minutes during a run


def _read_dotenv_key(path: Path, name: str) -> str | None:
    """Return the value of `name` in a dotenv file, or None if absent."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            val = line.split("=", 1)[1].strip()
            return val.strip("'").strip('"')
    return None


def _load_console_key() -> str | None:
    """Read CONSOLE_ANTHROPIC_API_KEY from ~/.claude/.env.

    We don't trust os.environ because Claude Code may override ANTHROPIC_API_KEY
    with its own subscription credentials.
    """
    return _read_dotenv_key(Path.home() / ".claude" / ".env", "CONSOLE_ANTHROPIC_API_KEY")


def _load_gemini_key() -> str | None:
    """Read GEMINI_API_KEY from ~/.gemini/.env (fallback to os.environ)."""
    return _read_dotenv_key(Path.home() / ".gemini" / ".env", "GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")


def _load_openrouter_key() -> str | None:
    """Read OPENROUTER_API_KEY from ~/.claude/.env (fallback to os.environ).

    Claude Code may override OPENROUTER_API_KEY in its session env, so we read
    the dotenv first and fall through to the process env only as a safety net.
    """
    return _read_dotenv_key(Path.home() / ".claude" / ".env", "OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")


def _load_openrouter_byok_key() -> str | None:
    """Read OPENROUTER_GEMINI_KEY (BYOK) from ~/.claude/.env.

    The BYOK key routes requests through OpenRouter but bills the upstream
    provider (Google for Gemini models). Used when --platform openrouter_byok
    is set.
    """
    return _read_dotenv_key(Path.home() / ".claude" / ".env", "OPENROUTER_GEMINI_KEY") or os.environ.get("OPENROUTER_GEMINI_KEY")


def get_console_cumulative_spend() -> float:
    """Read experiments/console_usage.jsonl and sum cost_usd."""
    if not CONSOLE_USAGE_LOG.exists():
        return 0.0
    total = 0.0
    for line in CONSOLE_USAGE_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            total += float(rec.get("cost_usd", 0) or 0)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return total


def log_console_usage(deliverable_id: str, cost_usd: float, model: str) -> None:
    """Append a usage entry to console_usage.jsonl at the repo root."""
    CONSOLE_USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "deliverable_id": deliverable_id,
        "model": model,
        "cost_usd": cost_usd,
    }
    with CONSOLE_USAGE_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _recent_openai_codex_sessions(limit: int = 20) -> list[Path]:
    """Return up to `limit` most-recently-updated OpenAI-authed Codex session files.

    Uses Codex's own sqlite index (~/.codex/state_5.sqlite, table `threads`)
    which tracks model_provider + rollout_path per thread — much faster than
    scanning tens of thousands of rollout-*.jsonl files by hand. Only sessions
    with model_provider == "openai" are returned (these are the ones that carry
    rate_limits in their token_count event_msgs).
    """
    if not CODEX_STATE_DB.exists():
        return []
    try:
        import sqlite3
        conn = sqlite3.connect(f"file:{CODEX_STATE_DB}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT rollout_path FROM threads "
                "WHERE model_provider='openai' AND rollout_path IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            return [Path(r[0]) for r in cur.fetchall() if r[0]]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


def get_codex_weekly_usage() -> dict | None:
    """Return the most recent Codex rate_limits snapshot, or None if unavailable.

    Codex CLI (>=0.118) writes rate-limit state into the `token_count` event_msg
    of sessions under ~/.codex/sessions/<year>/<month>/<day>/rollout-*.jsonl.
    Only sessions with model_provider == "openai" (i.e. `codex login` auth, not
    OpenRouter) emit these. We query Codex's own thread index (state_5.sqlite)
    for the 20 most recent OpenAI sessions, then scan each newest-first for a
    token_count event with a rate_limits block.

    If no OpenAI session is found, or none have a rate_limits snapshot, returns
    None — the watchdog treats that as "no data" rather than "0 percent".

    Shape:
        {
          "limit_id": "codex",
          "primary":   {"used_percent": float, "window_minutes": int, "resets_at": int},
          "secondary": {"used_percent": float, "window_minutes": 10080, "resets_at": int},
          "plan_type": "team" | "plus" | ...
        }

    If the returned snapshot's `secondary.resets_at` is in the past, the weekly
    window has rolled over since the snapshot was written. In that case the
    usage value no longer reflects reality; callers should treat it as stale
    (see `get_codex_weekly_used_percent` which returns None in that case).
    """
    for path in _recent_openai_codex_sessions():
        if not path.exists():
            continue
        last_rate_limits: dict | None = None
        try:
            with path.open() as f:
                for line in f:
                    if "token_count" not in line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = ev.get("payload", {}) if ev.get("type") == "event_msg" else {}
                    if payload.get("type") == "token_count" and payload.get("rate_limits"):
                        last_rate_limits = payload["rate_limits"]
        except OSError:
            continue
        # Require a usable weekly block before accepting this session. Some
        # snapshots (e.g. limit_id=premium without a window) have secondary=None
        # and tell us nothing about weekly usage; fall through to the next one.
        if last_rate_limits is None:
            continue
        secondary = last_rate_limits.get("secondary")
        if isinstance(secondary, dict) and secondary.get("used_percent") is not None:
            return last_rate_limits
    return None


def get_codex_weekly_used_percent() -> float | None:
    """Return the weekly (secondary) used_percent from the most recent session.

    Returns None if:
    - No OpenAI-authed Codex session was found in the recent-session window, or
    - The snapshot's `secondary.resets_at` has already passed (window rolled
      over, so the recorded used_percent no longer reflects current usage).
    """
    rl = get_codex_weekly_usage()
    if not rl:
        return None
    secondary = rl.get("secondary") or {}
    resets_at = secondary.get("resets_at")
    try:
        if resets_at is not None and int(resets_at) <= int(time.time()):
            return None
    except (TypeError, ValueError):
        pass
    pct = secondary.get("used_percent")
    try:
        return float(pct) if pct is not None else None
    except (TypeError, ValueError):
        return None


# ── Sandbox-based experiment layout ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = REPO_ROOT / "experiments"
EXP_META_DIR = EXP_ROOT / ".meta"
EXP_TEMPLATE_DIR = EXP_META_DIR / "template"       # full exp_dir shape template
EXP_ID_REGISTRY = EXP_META_DIR / "exp_ids.jsonl"


def _model_slug(model: str) -> str:
    """Filesystem-safe version of a model name.

    OpenRouter preset IDs like `@preset/gemini-3-1-pro-ai-studio` resolve to
    a specific upstream model at inference time. We don't want the preset
    prefix to fork experiment paths — the preset is a routing mechanism, not
    a different model. Map known preset IDs to their canonical model slug.
    """
    PRESET_TO_MODEL = {
        "@preset/gemini-3-1-pro-ai-studio":        "google/gemini-3.1-pro-preview",
        "@preset/gemini-3-1-flash-lite-ai-studio": "google/gemini-3.1-flash-lite-preview",
    }
    canonical = PRESET_TO_MODEL.get(model, model)
    return canonical.replace("/", "_").replace(":", "_").lstrip("@")


VALID_PLATFORMS = {
    "anthropic_console",        # Claude Code with CONSOLE_ANTHROPIC_API_KEY (--bare)
    "anthropic_subscription",   # Claude Code with OAuth / subscription
    "google_gemini_direct",     # Direct Google Gemini API (GEMINI_API_KEY / GOOGLE_GENERATIVE_AI_API_KEY)
    "openrouter",               # Main OpenRouter (OPENROUTER_API_KEY)
    "openrouter_byok",          # OpenRouter BYOK key (OPENROUTER_GEMINI_KEY) - billed to external provider
    "chatgpt_subscription",     # Codex with ChatGPT Plus auth (no env key)
    "other",
}


def _new_exp_id() -> str:
    """Generate a fresh experiment id: <YYYY-MM-DD_HHMM>_<6-char-hex>."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    return f"{ts}_{secrets.token_hex(3)}"


def _prompt_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def _find_matching_exp(harness: str, model: str, prompt_path: Path,
                       platform: str | None,
                       sandbox_version: str = "s2") -> str | None:
    """Auto-resume helper for --continue. Returns the exp_id of the most
    recent matching experiment, or None if no match.

    Identity tuple: (harness, model_slug, platform, prompt_hash, sandbox_version).
    Excludes registry entries whose path contains `_merged_into_` (already
    consolidated by `merge_split_runs.py`) or `_archived` (intentionally
    retired by the user). Verifies the directory still exists on disk.

    If the most recent match already has 1700 result files (= full coverage),
    raises SystemExit so the user has to explicitly opt into appending more
    or starting a new exp.
    """
    slug = _model_slug(model)
    prompt_hash_val = _prompt_hash(prompt_path.read_text())
    if not EXP_ID_REGISTRY.exists():
        return None
    candidates: list[dict] = []
    for line in EXP_ID_REGISTRY.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("harness") != harness:
            continue
        if d.get("model_slug") != slug:
            continue
        if d.get("platform") != platform:
            continue
        if d.get("prompt_hash") != prompt_hash_val:
            continue
        if d.get("sandbox_version") != sandbox_version:
            continue
        path = d.get("path") or ""
        if "_merged_into_" in path or "_archived" in path:
            continue
        full = REPO_ROOT / path
        if not full.exists():
            continue
        candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.get("started") or "", reverse=True)
    most_recent = candidates[0]
    # Refuse if results dir is already at full coverage (1700 files).
    results_dir = REPO_ROOT / (most_recent.get("path") or "") / "results"
    n_results = 0
    if results_dir.exists():
        for f in results_dir.iterdir():
            if f.suffix == ".json" and not f.name.startswith("_"):
                n_results += 1
    if n_results >= 1700:
        raise SystemExit(
            f"--continue refused: most recent matching exp "
            f"'{most_recent['exp_id']}' already has {n_results} result files "
            f"(>=1700, full coverage). Pass --resume-exp explicitly to keep "
            f"appending, or omit --continue to start a fresh exp_id."
        )
    return most_recent["exp_id"]


def _capture_env_snapshot() -> dict:
    """Snapshot of the environment at experiment start time.

    Captures git commit, Python version, hostname, CLI tool versions, and
    which API auth secrets are present (by prefix only, never the full key).
    Best-effort: missing tools just report null.
    """
    def _cmd(args: list[str]) -> str | None:
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
            return result.stdout.strip() or None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _key_fingerprint(val: str | None) -> str | None:
        """Return first 10 chars + SHA-256-hash prefix so we can identify which key
        was used without storing the secret. None if not set."""
        if not val:
            return None
        h = hashlib.sha256(val.encode()).hexdigest()[:8]
        prefix = val[:10]
        return f"{prefix}...(sha256:{h})"

    # Read CONSOLE_ANTHROPIC_API_KEY from ~/.claude/.env (not env vars, since Claude
    # Code overrides ANTHROPIC_API_KEY) and other keys from env / dotenv files.
    console_key = _load_console_key()
    gemini_key = None
    gemini_env_path = Path.home() / ".gemini" / ".env"
    if gemini_env_path.exists():
        for line in gemini_env_path.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                gemini_key = line.split("=", 1)[1].strip().strip("'").strip('"')
                break

    import platform as _platform

    snap = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "hostname": _platform.node(),
        "os": {
            "system": _platform.system(),
            "release": _platform.release(),
            "machine": _platform.machine(),
        },
        "python": {
            "version": _platform.python_version(),
            "implementation": _platform.python_implementation(),
        },
        "git": {
            "commit": _cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
            "branch": _cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_cmd(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])),
        },
        "tools": {
            "claude": _cmd(["claude", "--version"]),
            "gemini": _cmd(["gemini", "--version"]),
            "codex": _cmd(["codex", "--version"]),
            "opencode": _cmd([os.path.expanduser("~/.opencode/bin/opencode"), "--version"]),
            "openhands": _cmd(["openhands", "--version"]),
        },
        # Ground truth: which API keys were ACCESSIBLE at experiment start.
        # Only the prefix + hash is stored. Tells us which auth was in scope
        # without leaking the secret itself.
        "api_keys_available": {
            # Anthropic
            "ANTHROPIC_API_KEY_env": _key_fingerprint(os.environ.get("ANTHROPIC_API_KEY")),
            "CONSOLE_ANTHROPIC_API_KEY_file": _key_fingerprint(console_key),
            # OpenRouter
            "OPENROUTER_API_KEY_env": _key_fingerprint(os.environ.get("OPENROUTER_API_KEY")),
            "OPENROUTER_GEMINI_KEY_file": _key_fingerprint(
                _load_dotenv_value(Path.home() / ".claude" / ".env", "OPENROUTER_GEMINI_KEY")
            ),
            # Google Gemini
            "GEMINI_API_KEY_env": _key_fingerprint(os.environ.get("GEMINI_API_KEY")),
            "GEMINI_API_KEY_file": _key_fingerprint(gemini_key),
            "GOOGLE_GENERATIVE_AI_API_KEY_env": _key_fingerprint(os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")),
        },
    }
    return snap


def _load_dotenv_value(path: Path, var: str) -> str | None:
    """Read a single VAR=value from a .env-style file."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(var + "="):
            return line.split("=", 1)[1].strip().strip("'").strip('"')
    return None


def _provision_experiment(
    harness: str,
    model: str,
    prompt_path: Path,
    *,
    exp_id: str | None = None,
    notes: str = "",
    extra_config: dict | None = None,
) -> dict:
    """Create an experiment directory rooted at experiments/<harness>/<model_slug>/<exp_id>/.

    - rsync's .meta/template/ into <exp_dir>/ (copies sandbox, empty result dirs, errors.jsonl)
    - Writes <exp_dir>/prompt.md (also updates sandbox/prompt.md)
    - Writes <exp_dir>/config.json and <exp_dir>/env.json
    - Appends a line to .meta/exp_ids.jsonl
    - Returns the config dict (with 'path' = absolute exp_dir)

    If exp_id is provided and a config already exists at that path, it's
    treated as resume (no rewrite of config/env/registry).
    """
    if not EXP_TEMPLATE_DIR.exists():
        raise FileNotFoundError(
            f"Experiment template missing at {EXP_TEMPLATE_DIR}. Aborting."
        )

    slug = _model_slug(model)
    exp_id = exp_id or _new_exp_id()
    exp_dir = EXP_ROOT / harness / slug / exp_id
    config_path = exp_dir / "config.json"

    # Resume case: directory exists and has a config. Just return the config.
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        cfg["path"] = str(exp_dir)
        return cfg

    # Fresh experiment: rsync the whole template into the exp_dir.
    # -a preserves symlinks and perms. The template has sandbox/ (with the
    # 555/444 locked trajectories) plus empty results/, stdout/, stderr/,
    # analysis/ (each with a .gitkeep) and an empty errors.jsonl.
    exp_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", str(EXP_TEMPLATE_DIR) + "/", str(exp_dir) + "/"],
        check=True,
    )

    # Copy prompt as prompt.md inside exp_dir (and refresh the one inside sandbox too).
    prompt_text = prompt_path.read_text()
    sandbox_dir = exp_dir / "sandbox"
    # Temporarily chmod prompt.md writable (inherited 444 from template)
    (exp_dir / "prompt.md").write_text(prompt_text) if not (exp_dir / "prompt.md").exists() else None
    (exp_dir / "prompt.md").chmod(0o644)
    (exp_dir / "prompt.md").write_text(prompt_text)
    if (sandbox_dir / "prompt.md").exists():
        (sandbox_dir / "prompt.md").chmod(0o644)
    (sandbox_dir / "prompt.md").write_text(prompt_text)

    # env.json: snapshot at start
    (exp_dir / "env.json").write_text(json.dumps(_capture_env_snapshot(), indent=2) + "\n")

    config = {
        "exp_id": exp_id,
        "harness": harness,
        "model": model,
        "model_slug": slug,
        "platform": (extra_config or {}).get("platform"),  # Explicit, set from --platform CLI arg
        "prompt_hash": _prompt_hash(prompt_text),
        "prompt_file": "prompt.md",
        "sandbox_version": "s2",
        "started": datetime.now(timezone.utc).isoformat(),
        "completed": None,
        "path": str(exp_dir.relative_to(REPO_ROOT)),
        "notes": notes,
    }
    if extra_config:
        config.update(extra_config)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    # Append to registry
    EXP_ID_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with EXP_ID_REGISTRY.open("a") as f:
        f.write(json.dumps(config) + "\n")

    config["path"] = str(exp_dir)  # return absolute for in-process use
    return config


def _mark_experiment_complete(exp_dir: Path, counts: dict) -> None:
    """Patch config.json and the registry entry with completion timestamp + counts."""
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return
    cfg = json.loads(config_path.read_text())
    cfg["completed"] = datetime.now(timezone.utc).isoformat()
    cfg["counts"] = counts
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")

    # Update the matching registry line in-place
    if not EXP_ID_REGISTRY.exists():
        return
    lines = EXP_ID_REGISTRY.read_text().splitlines()
    updated = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            updated.append(line)
            continue
        if rec.get("exp_id") == cfg["exp_id"]:
            rec.update({"completed": cfg["completed"], "counts": counts})
            updated.append(json.dumps(rec))
        else:
            updated.append(line)
    EXP_ID_REGISTRY.write_text("\n".join(updated) + "\n")


# Shared state for the periodic cap checker
_CAP_ABORT_EVENT: asyncio.Event | None = None
_CAP_USD_OVERRIDE: float | None = None  # set by --cap-usd CLI flag


async def _cap_watchdog(interval: float = CONSOLE_SPEND_CHECK_INTERVAL_SEC) -> None:
    """Re-read cumulative spend every `interval` seconds. If >= cap, set abort event."""
    assert _CAP_ABORT_EVENT is not None
    cap = _CAP_USD_OVERRIDE if _CAP_USD_OVERRIDE is not None else CONSOLE_SPEND_CAP_USD
    while not _CAP_ABORT_EVENT.is_set():
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        spend = get_console_cumulative_spend()
        if spend >= cap:
            print(f"\n[CAP HIT] Cumulative Console spend ${spend:.2f} >= ${cap}. Aborting.", flush=True)
            _CAP_ABORT_EVENT.set()
            return
        if spend >= 0.8 * cap:
            print(f"[CAP WARN] Spend at ${spend:.2f} ({spend/cap*100:.0f}% of cap ${cap})", flush=True)


async def _codex_weekly_watchdog(interval: float = CODEX_WEEKLY_CHECK_INTERVAL_SEC) -> None:
    """Re-read latest Codex weekly usage every `interval` seconds. If >= cap, abort.

    Mirrors `_cap_watchdog` for the ChatGPT / Codex subscription plan. Reads the
    most recent `rate_limits.secondary.used_percent` from ~/.codex/sessions/. If
    the weekly window is at or above CODEX_WEEKLY_CAP_PERCENT, sets the shared
    abort event so in-flight workers exit gracefully at the next trajectory.
    """
    assert _CAP_ABORT_EVENT is not None
    while not _CAP_ABORT_EVENT.is_set():
        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            return
        pct = get_codex_weekly_used_percent()
        if pct is None:
            # Fresh install / no session files yet. Nothing to check.
            continue
        if pct >= CODEX_WEEKLY_CAP_PERCENT:
            rl = get_codex_weekly_usage() or {}
            resets_at = (rl.get("secondary") or {}).get("resets_at")
            resets_str = (
                datetime.fromtimestamp(int(resets_at), tz=timezone.utc).isoformat()
                if resets_at else "unknown"
            )
            print(
                f"\n[CODEX CAP HIT] Weekly usage {pct:.1f}% >= {CODEX_WEEKLY_CAP_PERCENT:.0f}%. "
                f"Aborting. Window resets at {resets_str}.",
                flush=True,
            )
            _CAP_ABORT_EVENT.set()
            return
        if pct >= CODEX_WEEKLY_WARN_PERCENT:
            print(f"[CODEX CAP WARN] Weekly usage at {pct:.1f}% (cap {CODEX_WEEKLY_CAP_PERCENT:.0f}%)", flush=True)

# Flush stdout on every print so background runs show live progress
print = functools.partial(print, flush=True)


# Judge CLI configuration
HARNESS_CONFIG = {
    "claude": {
        "cmd": "claude",
        "response_key": "result",
        "supports_session_id": True,
        "env_unset": ["CLAUDECODE"],
        "uses_stdin": True,
    },
    "gemini": {
        "cmd": "gemini",
        "response_key": "response",
        "supports_session_id": False,
        "env_unset": ["GEMINI_CLI"],
        "uses_stdin": True,
    },
    "codex": {
        "cmd": "codex",
        "response_key": None,  # JSONL output, parsed differently
        "supports_session_id": False,
        "env_unset": [],
        "uses_stdin": True,  # trajectory piped via stdin, prompt as CLI arg
    },
    "opencode": {
        "cmd": os.path.expanduser("~/.opencode/bin/opencode"),
        "response_key": None,  # JSONL output, parsed differently
        "supports_session_id": False,
        "env_unset": [],
        "uses_stdin": True,  # trajectory piped via stdin, prompt as CLI arg
    },
    "openhands": {
        "cmd": "openhands",
        "response_key": None,  # custom JSON event output
        "supports_session_id": False,
        "env_unset": [],
        "uses_stdin": False,  # uses -t flag for prompt
    },
    "pi": {
        "cmd": "pi",
        "response_key": None,  # JSONL event stream, parsed separately
        "supports_session_id": False,
        "env_unset": [],
        "uses_stdin": False,  # prompt passed as positional arg
    },
}


def build_cmd(judge: str, prompt_text: str, model: str, session_id: str,
              trajectory_content: str | None = None,
              effort: str | None = None,
              use_console_key: bool = False,
              platform: str | None = None) -> list[str]:
    """Build the CLI command for the given judge.

    Args:
        effort: reasoning effort flag for claude (xhigh|high|medium|low|minimal)
        use_console_key: if True, add --bare to force env-var auth (Console API,
            no subscription/keychain fallback)
        platform: explicit platform declaration. For codex, this drives the
            `-c model_provider=...` override (chatgpt_subscription vs openrouter_byok).
    """
    cfg = HARNESS_CONFIG[judge]
    if judge == "codex":
        # Codex 0.117+: prompt as arg, trajectory piped via stdin, model via -m
        # --skip-git-repo-check: sandbox CWD is not a git repo; required to let
        # Codex run there.
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]
        # Provider override. `~/.codex/config.toml` defaults to openrouter, but
        # chatgpt_subscription runs (GPT-5.x on the user's ChatGPT Plus plan)
        # need the built-in OpenAI provider which reads ~/.codex/auth.json.
        # Platforms of the form `vllm_yul201_<preset>` route to a self-hosted
        # vLLM server via a matching `[model_providers.<preset>_vllm]` entry in
        # ~/.codex/config.toml.
        if platform == "chatgpt_subscription":
            cmd.extend(["-c", 'model_provider = "openai"'])
        elif platform and platform.startswith("vllm_yul201_"):
            preset = platform[len("vllm_yul201_"):]
            cmd.extend(["-c", f'model_provider = "{preset}_vllm"'])
        # Reasoning effort override (GPT-5.x supports minimal|low|medium|high|xhigh;
        # default in ~/.codex/config.toml is "high"). Forward --effort when present.
        if effort:
            cmd.extend(["-c", f'model_reasoning_effort = "{effort}"'])
        if model and model != "sonnet":  # "sonnet" is the default placeholder
            cmd.extend(["-m", model])
        cmd.append(prompt_text)
        return cmd
    if judge == "opencode":
        # OpenCode: prompt as positional arg, trajectory piped via stdin, model via -m
        cmd = [cfg["cmd"], "run", "--format", "json"]
        if model and model != "sonnet":
            cmd.extend(["-m", model])
        cmd.append(prompt_text)
        return cmd
    if judge == "openhands":
        # OpenHands: prompt + trajectory combined in -t flag, model via env vars
        combined = prompt_text + "\n\n" + (trajectory_content or "")
        return [
            "openhands", "--headless", "--override-with-envs", "--json",
            "-t", combined,
        ]
    if judge == "pi":
        # pi (mariozechner/pi-coding-agent): non-interactive + JSON event stream.
        # Prompt as positional arg; model + provider via flags. `--no-*` disable
        # user-local extension/skill/theme discovery so runs are reproducible.
        cmd = [cfg["cmd"], "--mode", "json", "--print",
               "--no-extensions", "--no-skills", "--no-prompt-templates",
               "--no-themes", "--no-session"]
        # Model routing: accept "provider/model" or "model" (with --provider).
        if "/" in model:
            cmd.extend(["--model", model])
        else:
            # Default provider is google; accept gemini-* plain names as Google.
            cmd.extend(["--provider", "google", "--model", model])
        if effort:
            cmd.extend(["--thinking", effort])
        cmd.append(prompt_text)
        return cmd
    if judge == "gemini":
        # Gemini CLI "-p ... Appended to input on stdin (if any)" puts the
        # prompt AFTER the trajectory, which is the wrong order for judging.
        # Fix: concatenate prompt + trajectory into stdin ourselves (prompt first)
        # and use a minimal `-p` anchor to force non-interactive mode. The
        # model then sees: prompt -> trajectory (matching every other harness).
        combined = prompt_text + "\n\n" + (trajectory_content or "")
        # --approval-mode yolo: auto-approve any tool calls the model emits.
        # Without this, default mode blocks on approval in non-interactive runs
        # and tool calls would hang. We keep the full default tool set available
        # (read-file, shell, etc.) for parity with other harnesses.
        # Whitelist the canonical screenshots dir so Gemini-CLI's filesystem
        # sandbox treats `data/media/images/<id>/step_N.png` symlinks (which
        # point out of the per-exp sandbox to <repo>/data/media/images) as
        # readable. Without this, read_file silently returns empty content for
        # the symlinked path and weaker models (Flash Lite) give up before
        # finding a working alternative.
        images_dir = REPO_ROOT / "data" / "media" / "images"
        cmd = [cfg["cmd"], "-p", "Respond to the instructions above.",
               "--model", model, "--output-format", "json",
               "--approval-mode", "yolo",
               "--include-directories", str(images_dir)]
        # Caller needs to know to pipe `combined` as stdin, not just trajectory.
        # We signal that by returning a special tuple-like list; see evaluate_one.
        return cmd
    # claude CLI
    cmd = [cfg["cmd"], "-p"]
    if judge == "claude":
        # Empirical test 2026-04-18 confirmed Claude Code honors
        # ANTHROPIC_API_KEY for billing even without --bare, and keeps its
        # standard scaffolding (system prompt + tool schemas, ~37k tokens)
        # which is essential for judge accuracy. Dropping --bare recovers
        # ~15pp on Opus 4.7 xhigh vs the stripped-down Console-billed setup.
        #
        # Permission mode: `--dangerously-skip-permissions` because each eval
        # creates a brand-new project dir (via `--session-id <uuid>`) whose
        # trusted-paths cache is empty. `--permission-mode auto` then denies
        # Read on PNGs ("hasn't been granted yet"), which silently caps the
        # model's vision. Bypassing permissions is safe: the subprocess is
        # scripted, single-shot, and confined to the exp sandbox dir. Verified
        # 2026-04-24 by observing Haiku 4.5's PNG reads returning ERR_PERM
        # under `auto` despite the skipAutoPermissionPrompt flag in
        # ~/.claude/settings.json.
        cmd.append("--dangerously-skip-permissions")
        if effort:
            cmd.extend(["--effort", effort])
    cmd.extend([prompt_text, "--model", model, "--output-format", "json"])
    if cfg["supports_session_id"]:
        cmd.extend(["--session-id", session_id])
    return cmd


def _extract_balanced_objects(text: str):
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 0
            start = i
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start : j + 1]
                        i = j + 1
                        break
            else:
                return
        else:
            i += 1


def _extract_verdict_from_response(text: str) -> dict | None:
    """Extract judge verdict dict from possibly-prose response. Returns None if
    no valid JSON dict found. Prefers blocks with a "success" key.

    Handles: pure JSON, surrounding ```fences```, fences inside prose,
    and balanced {...} substrings.
    """
    if not text:
        return None
    s = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", s)
    stripped = re.sub(r"\n?```\s*$", "", stripped).strip()
    try:
        d = json.loads(stripped)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    for block in reversed(re.findall(r"```(?:json)?\s*\n(.*?)\n```", s, flags=re.DOTALL)):
        try:
            d = json.loads(block.strip())
            if isinstance(d, dict) and "success" in d:
                return d
        except json.JSONDecodeError:
            continue
    candidates = list(_extract_balanced_objects(s))
    for block in reversed(candidates):
        try:
            d = json.loads(block)
            if isinstance(d, dict) and "success" in d:
                return d
        except json.JSONDecodeError:
            continue
    for block in candidates:
        try:
            d = json.loads(block)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return None


def extract_claude_session_thinking(session_path: Path) -> str | None:
    """Extract concatenated extended-thinking blocks from a Claude Code session JSONL.

    Claude Code writes each assistant turn as a JSONL line where
    message.content is an array of content blocks; thinking blocks have
    type == "thinking" and carry the text under a "thinking" key.
    Returns the joined thinking text, or None if the session has no thinking
    blocks (e.g., extended thinking disabled).
    """
    if not session_path.exists():
        return None
    chunks: list[str] = []
    try:
        for line in session_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "assistant":
                continue
            msg = entry.get("message") or {}
            content = msg.get("content") or []
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    text = block.get("thinking") or block.get("text") or ""
                    if text:
                        chunks.append(text)
    except OSError:
        return None
    return "\n\n".join(chunks) if chunks else None


def parse_codex_output(raw_output: str) -> tuple[str, dict]:
    """Parse Codex JSONL output. Returns (response_text, metadata).

    Supports both old format (v0.38, msg.type=agent_message) and
    new format (v0.117+, type=item.completed with item.text).

    metadata fields:
      - usage: final token usage dict
      - thinking: concatenated reasoning chunks (if any)
      - turns: count of turn.completed events (proxy for API round-trips)
      - tool_calls: count of item.completed events with tool-type items
                    (command_execution, function_call, mcp_tool_call)
      - tool_names_by_count: {tool_name: count} for the tool calls
    """
    TOOL_ITEM_TYPES = {"command_execution", "function_call", "mcp_tool_call"}
    response_text = ""
    metadata = {}
    thinking_chunks: list[str] = []
    turns = 0
    tool_calls = 0
    tool_names: dict[str, int] = {}
    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Old format (v0.38): first line has run config
        if "model" in event and "sandbox" in event:
            metadata["model"] = event.get("model")
            metadata["provider"] = event.get("provider")
            continue
        # Old format: agent_message
        msg = event.get("msg", {})
        if msg.get("type") == "agent_message":
            response_text = msg.get("message", "")
        # Old format: agent_reasoning
        if msg.get("type") in ("agent_reasoning", "reasoning"):
            chunk = msg.get("text") or msg.get("message") or ""
            if chunk:
                thinking_chunks.append(chunk)
        # New format (v0.117+): item.completed with agent_message or tool item
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            itype = item.get("type")
            if itype == "agent_message":
                response_text = item.get("text", "")
            elif itype in ("reasoning", "agent_reasoning"):
                chunk = item.get("text") or item.get("summary") or ""
                if chunk:
                    thinking_chunks.append(chunk)
            elif itype in TOOL_ITEM_TYPES:
                tool_calls += 1
                # Record tool name. For command_execution use the command,
                # for function_call use function.name, for mcp_tool_call use name.
                name = (
                    item.get("name")
                    or (item.get("function") or {}).get("name")
                    or itype
                )
                tool_names[name] = tool_names.get(name, 0) + 1
        # New format: turn.completed has usage + signals a turn boundary
        if event.get("type") == "turn.completed":
            turns += 1
            usage = event.get("usage", {})
            if usage:
                metadata["usage"] = usage
    if thinking_chunks:
        metadata["thinking"] = "\n\n".join(thinking_chunks)
    metadata["turns"] = turns
    metadata["tool_calls"] = tool_calls
    if tool_names:
        metadata["tool_names_by_count"] = tool_names
    return response_text, metadata


def parse_pi_output(raw_output: str) -> tuple[str, dict]:
    """Parse pi (mariozechner/pi-coding-agent) --mode json --print output.

    Pi emits a JSONL event stream with types:
      - session, agent_start, turn_start, message_start, message_update,
        message_end, turn_end, agent_end, tool_call_*, tool_result_*
    Each `message_end` with role=assistant has the fully-formed message and its
    per-message usage + cost. We take the last assistant message's text as the
    response and aggregate usage across all assistant messages.
    """
    response_text = ""
    metadata: dict = {}
    thinking_chunks: list[str] = []
    total_in = total_out = 0
    total_cached = 0
    total_cost = 0.0
    turns = 0
    tool_calls = 0
    tool_names: dict[str, int] = {}
    model_used: str | None = None

    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "message_end":
            msg = event.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            # Text content: concatenate every text part
            parts = msg.get("content") or []
            text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
            if text_parts:
                response_text = "\n".join(text_parts)
            # Reasoning parts (if the provider surfaces them)
            for p in parts:
                if isinstance(p, dict) and p.get("type") in ("thinking", "reasoning"):
                    chunk = p.get("text") or p.get("content") or ""
                    if chunk:
                        thinking_chunks.append(chunk)
            usage = msg.get("usage") or {}
            total_in += int(usage.get("input") or 0)
            total_out += int(usage.get("output") or 0)
            total_cached += int(usage.get("cacheRead") or 0)
            cost_info = usage.get("cost") or {}
            total_cost += float(cost_info.get("total") or 0.0)
            model_used = msg.get("model") or model_used
        elif etype == "turn_end":
            turns += 1
        elif etype == "tool_execution_end":
            tool_calls += 1
            name = event.get("toolName") or "tool"
            tool_names[name] = tool_names.get(name, 0) + 1

    metadata["usage"] = {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cached_input_tokens": total_cached,
    }
    metadata["cost_usd"] = total_cost or None
    metadata["turns"] = turns
    metadata["tool_calls"] = tool_calls
    if tool_names:
        metadata["tool_names_by_count"] = tool_names
    if model_used:
        metadata["model"] = model_used
    if thinking_chunks:
        metadata["thinking"] = "\n\n".join(thinking_chunks)
    return response_text, metadata


def parse_opencode_output(raw_output: str) -> tuple[str, dict]:
    """Parse OpenCode JSONL output. Returns (response_text, metadata).

    Tracks turns (step_finish), tool_calls (tool_use), tools_by_count,
    and SUMS cost across all step_finish events (previously only the last
    step's cost was recorded).

    metadata['thinking'] captures any reasoning-type parts OpenCode surfaces
    (varies by provider; OpenRouter-backed models with reasoning enabled emit
    `type: reasoning` parts).
    """
    response_text = ""
    metadata: dict = {}
    thinking_chunks: list[str] = []
    total_cost = 0.0
    in_tok = out_tok = 0
    turns = 0
    tool_calls = 0
    tool_names: dict[str, int] = {}
    for line in raw_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        part = event.get("part", {}) if isinstance(event.get("part"), dict) else {}
        if etype == "text":
            response_text = part.get("text", "")
        elif etype == "reasoning":
            chunk = part.get("text") or part.get("content") or ""
            if chunk:
                thinking_chunks.append(chunk)
        elif etype == "step_finish":
            turns += 1
            total_cost += float(part.get("cost") or 0.0)
            tokens = part.get("tokens") or {}
            in_tok += int(tokens.get("input") or 0)
            out_tok += int(tokens.get("output") or 0)
        elif etype == "tool_use":
            tool_calls += 1
            name = event.get("tool") or part.get("tool") or "tool"
            tool_names[name] = tool_names.get(name, 0) + 1
    metadata["cost_usd"] = total_cost or None
    metadata["tokens"] = {"input": in_tok, "output": out_tok} if (in_tok or out_tok) else None
    metadata["usage"] = {"input_tokens": in_tok, "output_tokens": out_tok}
    metadata["turns"] = turns
    metadata["tool_calls"] = tool_calls
    if tool_names:
        metadata["tool_names_by_count"] = tool_names
    if thinking_chunks:
        metadata["thinking"] = "\n\n".join(thinking_chunks)
    return response_text, metadata


def parse_openhands_output(raw_output: str) -> tuple[str, dict]:
    """Parse OpenHands --json output. Returns (response_text, metadata).

    OpenHands CLI does not surface token counts or cost in its event stream,
    so `metadata["cost_usd"]` is left None and dashboard-side cost derivation
    uses a char-length heuristic. `turns` and `tool_calls` are counted from
    the event stream directly (agent messages with a `tool_call` field).
    """
    response_text = ""
    metadata: dict = {}
    turns = 0
    tool_calls = 0
    tool_names: dict[str, int] = {}
    # Estimate tokens by summing char-lengths of llm_message content, /4.
    char_in = 0
    char_out = 0
    # OpenHands outputs '--JSON Event--' markers followed by JSON blocks
    chunks = raw_output.split("--JSON Event--")
    for chunk in chunks:
        chunk = chunk.strip()
        # Find the JSON object in the chunk (may have TUI noise around it)
        brace_start = chunk.find("{")
        if brace_start == -1:
            continue
        # Find matching closing brace
        depth = 0
        json_str = ""
        for i in range(brace_start, len(chunk)):
            if chunk[i] == "{":
                depth += 1
            elif chunk[i] == "}":
                depth -= 1
            if depth == 0:
                json_str = chunk[brace_start : i + 1]
                break
        if not json_str:
            continue
        try:
            event = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        src = event.get("source")
        # Count turns and tool calls across the whole stream
        if src == "agent":
            turns += 1
            if event.get("tool_call") is not None or event.get("action"):
                tool_calls += 1
                name = event.get("tool_name") or "action"
                tool_names[name] = tool_names.get(name, 0) + 1
        # Estimate tokens by summing character counts of rendered content
        if src == "user" or src == "environment":
            char_in += len(json.dumps(event, default=str))
        if src == "agent":
            char_out += len(json.dumps(event, default=str))
        if src == "agent" and event.get("llm_message"):
            llm_msg = event["llm_message"]
            content = llm_msg.get("content", [])
            thinking_chunks: list[str] = []
            for part in content:
                if part.get("type") == "text":
                    response_text = part.get("text", "")
                elif part.get("type") in ("thinking", "reasoning"):
                    chunk = part.get("thinking") or part.get("text") or ""
                    if chunk:
                        thinking_chunks.append(chunk)
            # LiteLLM-style: some providers put reasoning in a top-level field
            rc = llm_msg.get("reasoning_content")
            if rc:
                thinking_chunks.append(rc)
            if thinking_chunks:
                metadata["thinking"] = "\n\n".join(thinking_chunks)
            metadata["model"] = event.get("llm_response_id")
        # ActionEvent with kind=FinishAction carries the model's final verdict
        # in action.message (the OpenHands agent's terminal action when it's
        # ready to stop). Prefer this over llm_message because streaming
        # agents often emit all their JSON verdict here, not as a text block.
        if src == "agent":
            action = event.get("action") or {}
            if isinstance(action, dict) and action.get("kind") == "FinishAction":
                msg = action.get("message") or ""
                if msg:
                    response_text = msg
            # Some OpenHands variants also surface reasoning_content on the
            # ActionEvent top level.
            rc_top = event.get("reasoning_content")
            if rc_top:
                metadata.setdefault("_thinking_extra", []).append(rc_top)
    metadata["turns"] = turns
    metadata["tool_calls"] = tool_calls
    if tool_names:
        metadata["tool_names_by_count"] = tool_names
    # Rough token estimate: ~4 chars/token. Used by the dashboard when the
    # harness doesn't surface native token counts.
    if char_in or char_out:
        metadata["usage"] = {
            "input_tokens": char_in // 4,
            "output_tokens": char_out // 4,
        }
    return response_text, metadata


async def evaluate_one(
    md_path: Path,
    output_dir: Path,
    prompt_text: str,
    model: str,
    judge: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
    dry_run: bool,
    max_retries: int = 3,
    effort: str | None = None,
    console_key: str | None = None,
    sandbox_dir: Path | None = None,
    platform: str | None = None,
) -> str:
    """Evaluate a single trajectory. Returns 'ok', 'skip', or 'error'."""
    cfg = HARNESS_CONFIG[judge]
    deliverable_id = md_path.stem
    json_path = output_dir / f"{deliverable_id}.json"

    # Bail out if the spend cap has been hit (watchdog sets this event).
    if _CAP_ABORT_EVENT is not None and _CAP_ABORT_EVENT.is_set():
        print(f"[{index}/{total}] [CAP-ABORT] {deliverable_id}")
        return "error"

    # Resumability: skip if output already exists
    if json_path.exists():
        print(f"[{index}/{total}] [SKIP] {deliverable_id}")
        return "skip"

    if dry_run:
        print(f"[{index}/{total}] [DRY-RUN] {deliverable_id}")
        return "skip"

    # Template substitution. If the prompt contains `{{TRAJECTORY_ID}}`, the
    # judge is expected to read the markdown itself via tools (e.g. P6-style).
    # Per-trajectory: substitute the placeholder, send no trajectory content on
    # stdin. If the placeholder is absent: legacy behaviour, send full markdown
    # on stdin alongside the prompt (P0/P1/P4/P5 style).
    if "{{TRAJECTORY_ID}}" in prompt_text:
        prompt_text = prompt_text.replace("{{TRAJECTORY_ID}}", deliverable_id)
        trajectory_content = ""
    else:
        trajectory_content = md_path.read_text()
    result = None
    cli_output = None
    codex_meta = {}
    raw_output = ""
    stderr_text = ""
    response_text = ""

    for attempt in range(max_retries):
        session_id = str(uuid.uuid4())
        rate_limit_retries = 0
        while True:  # inner loop for rate limit waits (doesn't consume attempts)
            rate_limit_wait = 0  # set >0 to sleep AFTER releasing semaphore
            async with semaphore:
                label = f"[{index}/{total}]"
                if attempt > 0:
                    label += f" [RETRY {attempt}]"
                print(f"{label} [RUN]  {deliverable_id} (session {session_id[:8]}...)")

                # CLI subprocess
                env = {k: v for k, v in os.environ.items() if k not in cfg["env_unset"]}
                # Redirect subprocess TMPDIR to user's scratch dir so heavy
                # tool-use (subagents, bash, tmux) doesn't fill /tmp and
                # cascade-kill Claude Code's own bash tool (which needs /tmp).
                scratch_tmp = Path.home() / "scratch" / "tmp"
                if scratch_tmp.exists():
                    env["TMPDIR"] = str(scratch_tmp)
                # Claude Code overrides OPENROUTER_API_KEY with its own token in
                # the process env. For every OpenRouter-using subprocess (codex,
                # opencode, openhands), replace it with the real key from the
                # dotenv file. When platform=openrouter_byok, use the BYOK key
                # (OPENROUTER_GEMINI_KEY) which routes to the upstream provider.
                if platform == "openrouter_byok":
                    _or_key = _load_openrouter_byok_key()
                else:
                    _or_key = _load_openrouter_key()
                if _or_key:
                    env["OPENROUTER_API_KEY"] = _or_key
                # OpenHands needs LLM env vars for model routing
                if judge == "openhands":
                    # Route to the right provider based on model prefix +
                    # platform override. A `vllm_yul201_*` platform (used for
                    # self-hosted vLLM on the superpod) means "respect the
                    # LLM_* env vars the caller already set" — don't touch
                    # them. Same for any model prefixed with `openai/` coming
                    # in via a vllm_yul201 platform.
                    if platform and platform.startswith("vllm_yul201_"):
                        # Caller (e.g. /tmp/vllm_val_launcher.py) is responsible
                        # for setting LLM_API_KEY / LLM_MODEL / LLM_BASE_URL
                        # with the tunnel URL and API key. Leave env alone.
                        pass
                    elif model.startswith("google/") or model.startswith("gemini/"):
                        # Gemini models: use Gemini API directly via LiteLLM.
                        # Read GEMINI_API_KEY from ~/.gemini/.env since it's not
                        # normally in os.environ in this setup.
                        env["LLM_API_KEY"] = _load_gemini_key() or ""
                        env["LLM_MODEL"] = f"gemini/{model.split('/')[-1]}" if not model.startswith("gemini/") else model
                        env.pop("LLM_BASE_URL", None)
                    else:
                        # Other models: use OpenRouter
                        env["LLM_API_KEY"] = _load_openrouter_key() or ""
                        env["LLM_MODEL"] = f"openrouter/{model}" if not model.startswith("openrouter/") else model
                        env["LLM_BASE_URL"] = "https://openrouter.ai/api/v1"
                # pi reads API keys directly from env vars (no LiteLLM indirection):
                # GEMINI_API_KEY / GOOGLE_API_KEY for google-provider models,
                # OPENROUTER_API_KEY for openrouter-provider models (already set above).
                if judge == "pi":
                    if model.startswith("gemini") or model.startswith("google/") or model.startswith("gemini/"):
                        gk = _load_gemini_key()
                        if gk:
                            env["GEMINI_API_KEY"] = gk
                            env["GOOGLE_API_KEY"] = gk
                # gemini-cli also reads GEMINI_API_KEY from env. It loads
                # ~/.gemini/.env on its own, but a stale or empty
                # GEMINI_API_KEY in the parent shell can take precedence and
                # cause expired-key errors after a key rotation. Always inject
                # the freshly-loaded key into the child env so rotations land.
                # Also set GEMINI_CLI_TRUST_WORKSPACE=true so 0.39+ does not
                # silently exit headless mode on untrusted folders.
                if judge == "gemini":
                    gk = _load_gemini_key()
                    if gk:
                        env["GEMINI_API_KEY"] = gk
                        env["GOOGLE_API_KEY"] = gk
                    env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
                # OpenCode caps per-request output at OUTPUT_TOKEN_MAX (default 32k);
                # raise to 2^16 so thinking models (Qwen 3.x, MiniMax) don't truncate
                # mid-reasoning. Must match `limit.output` set per-model in
                # ~/.config/opencode/opencode.json — opencode uses
                # min(model.limit.output, OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX).
                if judge == "opencode":
                    env["OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX"] = "65536"
                # Console API key for Claude: set ANTHROPIC_API_KEY, enable --bare
                use_console_key = bool(console_key) and judge == "claude"
                if use_console_key:
                    env["ANTHROPIC_API_KEY"] = console_key
                cmd = build_cmd(judge, prompt_text, model, session_id,
                                trajectory_content=trajectory_content,
                                effort=effort,
                                use_console_key=use_console_key,
                                platform=platform)
                # Gemini CLI needs prompt-then-trajectory on stdin (the CLI appends
                # -p AFTER stdin, which would reverse the order). Every other harness
                # receives trajectory-only on stdin (prompt goes via CLI arg).
                if judge == "gemini":
                    stdin_input = (prompt_text + "\n\n" + trajectory_content).encode()
                else:
                    stdin_input = trajectory_content.encode() if cfg["uses_stdin"] else None
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if cfg["uses_stdin"] else asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=str(sandbox_dir) if sandbox_dir else None,
                )
                # Per-trajectory timeout so a single hung subprocess can't
                # deadlock the whole eval (observed with qwen35-9B + opencode:
                # an opencode worker would silently hang on a misbehaved
                # vLLM stream and the parent would block forever in
                # `proc.communicate`). 30 min is generous for any normal
                # trajectory (long-context judges rarely take >5min).
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=stdin_input), timeout=1800
                    )
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=10
                        )
                    except asyncio.TimeoutError:
                        stdout, stderr = b"", b""
                    print(f"[{index}/{total}] [TIMEOUT] {deliverable_id}: subprocess exceeded 1800s, killed", flush=True)

                raw_output = stdout.decode(errors="replace").strip()
                stderr_text = stderr.decode(errors="replace").strip()

                # Check for rate limit errors in stderr only.
                #
                # The judge model's own reasoning text (which lands in stdout)
                # routinely contains the words "rate limit" or "quota" — for
                # example, the trajectory might describe an agent hitting a
                # rate limit during a task. Matching those in stdout produced
                # spurious 300s waits and 10x retries against trajectories
                # that were never actually rate-limited.
                #
                # Real rate-limit errors land in stderr (codex CLI prints them
                # there explicitly). vLLM platforms don't rate-limit at all,
                # and OpenRouter BYOK / Gemini direct have very generous quotas
                # at this scale, so for those we skip detection entirely.
                rate_limited = False
                wait_seconds = 300  # default 5 min wait
                NO_REAL_RATELIMIT = (
                    (platform or "").startswith("vllm_")
                    or platform == "openrouter_byok"
                    or platform == "google_gemini_direct"
                )
                if not NO_REAL_RATELIMIT:
                    # Only inspect stderr — the model's stdout reasoning text
                    # is full of legitimate "rate limit" / "quota" mentions.
                    for text in [stderr_text]:
                        if "usage limit" in text.lower() or "rate limit" in text.lower() or "quota" in text.lower():
                            rate_limited = True
                            time_match = re.search(r"try again at (\d{1,2}:\d{2}\s*[AP]M)", text, re.IGNORECASE)
                            if time_match:
                                try:
                                    from datetime import datetime
                                    reset_str = time_match.group(1)
                                    now = datetime.now()
                                    reset_time = datetime.strptime(reset_str, "%I:%M %p").replace(
                                        year=now.year, month=now.month, day=now.day
                                    )
                                    if reset_time < now:
                                        reset_time = reset_time.replace(day=now.day + 1)
                                    wait_seconds = max(int((reset_time - now).total_seconds()) + 30, 60)
                                except Exception:
                                    pass
                            # Cap wait time at 30 min to avoid sleeping through reset windows
                            wait_seconds = min(wait_seconds, 1800)
                            break

                if rate_limited:
                    rate_limit_retries += 1
                    if rate_limit_retries > 10:
                        print(f"[{index}/{total}] [ERROR] {deliverable_id}: rate limited too many times")
                        return "error"
                    print(f"[{index}/{total}] [RATE_LIMIT] {deliverable_id}: waiting {wait_seconds}s for reset (rate retry {rate_limit_retries})...")
                    rate_limit_wait = wait_seconds
                    # Fall through to release semaphore before sleeping
                else:
                    if proc.returncode != 0:
                        print(f"[{index}/{total}] [ERROR] {deliverable_id} (exit {proc.returncode}): {stderr_text[:200]}")
                        return "error"

                    # Parse output per judge type
                    if judge == "codex":
                        response_text, codex_meta = parse_codex_output(raw_output)
                    elif judge == "opencode":
                        response_text, codex_meta = parse_opencode_output(raw_output)
                    elif judge == "openhands":
                        response_text, codex_meta = parse_openhands_output(raw_output)
                    elif judge == "pi":
                        response_text, codex_meta = parse_pi_output(raw_output)
                    else:
                        try:
                            cli_output = json.loads(raw_output)
                        except json.JSONDecodeError:
                            print(f"[{index}/{total}] [ERROR] {deliverable_id}: failed to parse {judge} JSON output")
                            return "error"
                        response_text = cli_output.get(cfg["response_key"], "")
                    break  # exit inner rate-limit loop

            # Semaphore released. Sleep for rate limit OUTSIDE the semaphore
            # so other workers can proceed while this one waits.
            if rate_limit_wait > 0:
                await asyncio.sleep(rate_limit_wait)
                continue  # retry inner loop
            break  # non-rate-limit exit (success, etc.)

        # Parse the model's JSON evaluation. The judge sometimes wraps the JSON
        # in prose ("Based on my analysis...") + markdown fences; use the robust
        # extractor from scripts.rescue_raw_responses which handles:
        #   1. pure JSON
        #   2. JSON inside surrounding ```json fences
        #   3. JSON inside a ```json block embedded in prose
        #   4. balanced {...} substring (picks the last one with "success")
        parsed = _extract_verdict_from_response(response_text)
        if parsed is not None and "success" in parsed:
            result = parsed
            break  # success
        # Gemini-cli fallback: gemini-cli's --output-format json has a bug
        # where stdout `.response` captures the wrong text (a fragment of a
        # tool-call arg, an intermediate text, or degenerate decoder output).
        # The correct final answer sits in the session JSON at
        # ~/.gemini/tmp/sandbox-*/chats/session-*.json under
        # messages[-1].content for the last gemini message with text.
        # When primary parse fails on a gemini-harness run, look up the
        # session by the session_id we passed via --session-id and try
        # extracting the verdict from there before retrying or giving up.
        if judge == "gemini" and parsed is None:
            try:
                gemini_tmp = Path.home() / ".gemini" / "tmp"
                # gemini-cli reports its OWN session_id in stdout JSON; that
                # is what's stored in the session file's `sessionId` field
                # AND embedded as the filename suffix (first 8 chars).
                # Path/format depends on gemini-cli version:
                #   v0.34: ~/.gemini/tmp/sandbox-N/chats/session-*.json (single JSON)
                #   v0.40: ~/.gemini/tmp/<project>/chats/session-*.jsonl (JSONL stream)
                # Glob both. v0.40 sessions are a poor recovery source in
                # practice (gibberish stdout = gibberish session content) but
                # any v0.34 leftover with stdout-truncation gets recovered.
                gemini_sid = ""
                if cli_output:
                    gemini_sid = cli_output.get("session_id", "") or ""
                short = gemini_sid.split("-", 1)[0] if gemini_sid else ""
                candidates: list[Path] = []
                if short:
                    candidates += list(gemini_tmp.glob(f"*/chats/*-{short}.jsonl"))
                    candidates += list(gemini_tmp.glob(f"*/chats/*-{short}.json"))
                for f in candidates:
                    final_text = ""
                    try:
                        if f.suffix == ".jsonl":
                            sid_in_file = ""
                            for line in f.read_text().splitlines():
                                try:
                                    e = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if "sessionId" in e and not sid_in_file:
                                    sid_in_file = e.get("sessionId", "")
                                if e.get("type") == "gemini":
                                    c = e.get("content")
                                    if isinstance(c, str) and c.strip():
                                        final_text = c
                            if sid_in_file != gemini_sid:
                                continue
                        else:
                            d = json.loads(f.read_text())
                            if d.get("sessionId") != gemini_sid:
                                continue
                            for m in d.get("messages", []):
                                if m.get("type") == "gemini":
                                    c = m.get("content")
                                    if isinstance(c, str) and c.strip():
                                        final_text = c
                    except (OSError, json.JSONDecodeError):
                        continue
                    parsed_fb = _extract_verdict_from_response(final_text)
                    if parsed_fb is not None and "success" in parsed_fb:
                        print(f"[{index}/{total}] [SESSION_FALLBACK] {deliverable_id}: recovered verdict from session JSON")
                        result = parsed_fb
                        result["_recovered_from_session"] = True
                        break
                if result is not None:
                    break  # got it from fallback, exit retry loop
            except Exception:
                # Best-effort: if the fallback fails for any reason, fall through
                # to the existing short-truncation guard / retry path.
                pass
        # Short-truncation guard: if the model produced very little text and
        # nothing JSON-like, retrying with the full prompt + screenshots is
        # almost always wasted money (observed pattern: gemini-cli returns
        # < 30 chars of trailing-edge gibberish like "eb.json) PM" after
        # consuming 100k+ input tokens; retries hit the same failure mode).
        # Bail out of the retry loop and let the caller treat it as ERROR.
        text_len = len(response_text.strip())
        looks_truncated = (
            text_len < 200
            and "success" not in response_text.lower()
            and "{" not in response_text
        )
        # Allow disabling the short-truncation guard via env var, e.g. for
        # cleanup repushes where the user wants every retry to fire (the
        # short outputs are stochastic and additional samples sometimes
        # produce valid JSON).
        if looks_truncated and not os.environ.get("EVAL_DISABLE_SHORT_TRUNCATION_GUARD"):
            if attempt > 0:
                print(f"[{index}/{total}] [SHORT_TRUNCATION] {deliverable_id}: model output too short to retry ({text_len} chars), bailing")
            result = {"raw_response": response_text, "short_truncation": True}
            break
        if attempt < max_retries - 1:
            print(f"[{index}/{total}] [PARSE_FAIL] {deliverable_id}: retrying ({attempt + 1}/{max_retries})")
        else:
            # Preserve the raw for later rescue/debugging, but if the extractor
            # got a partial dict (e.g. missing "success"), keep that too.
            result = parsed if parsed is not None else {"raw_response": response_text}
            if "raw_response" not in result:
                result["raw_response"] = response_text

    result["deliverable_id"] = deliverable_id

    # Build metadata with a normalized common schema + harness-specific raw field.
    # Common keys: cost_usd, duration_ms, input_tokens, output_tokens, model_used
    # `raw` preserves the original harness-specific dict so nothing is lost.
    cost_usd: float | None = None
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_used: str | None = None
    thinking: str | None = None
    raw_meta: dict = {}

    if judge == "claude":
        raw_meta = dict(cli_output) if cli_output else {}
        cost_usd = raw_meta.get("total_cost_usd")
        duration_ms = raw_meta.get("duration_ms")
        usage = raw_meta.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        model_usage = raw_meta.get("modelUsage") or {}
        model_used = next(iter(model_usage.keys()), None) if isinstance(model_usage, dict) else None
        # Extended-thinking blocks live in the session transcript, not in
        # the CLI's top-level JSON. Pull them from ~/.claude/projects/...
        if cfg["supports_session_id"]:
            project_dir_name = Path.cwd().as_posix().replace("/", "-")
            session_src = Path.home() / ".claude" / "projects" / project_dir_name / f"{session_id}.jsonl"
            thinking = extract_claude_session_thinking(session_src)
    elif judge == "gemini":
        raw_meta = dict(cli_output) if cli_output else {}
        stats = raw_meta.get("stats") or {}
        duration_ms = stats.get("totalLatencyMs")
        models = stats.get("models") or {}
        # Sum tokens across all models that appear in the trace
        if isinstance(models, dict):
            for _name, m in models.items():
                if not isinstance(m, dict):
                    continue
                input_tokens = (input_tokens or 0) + (m.get("promptTokenCount") or m.get("totalPromptTokens") or 0)
                output_tokens = (output_tokens or 0) + (m.get("candidatesTokenCount") or m.get("totalCandidatesTokens") or 0)
            model_used = next(iter(models.keys()), None)
        # Gemini CLI does not surface thought parts in its JSON output today;
        # thinking remains None. Enable it via the direct Phase 2 runner
        # (llm_judges/evaluate.py --thinking-budget) when we need the thoughts.
    elif judge in ("codex", "opencode", "openhands", "pi"):
        raw_meta = dict(codex_meta) if codex_meta else {}
        cost_usd = raw_meta.get("cost_usd")
        usage = raw_meta.get("usage") or {}
        tokens = raw_meta.get("tokens") or {}
        input_tokens = usage.get("input_tokens") or tokens.get("input")
        output_tokens = usage.get("output_tokens") or tokens.get("output")
        model_used = raw_meta.get("model")
        thinking = raw_meta.get("thinking")

    result["_meta"] = {
        "cost_usd": cost_usd,
        "duration_ms": duration_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model_used": model_used,
        "thinking": thinking,
        "raw": raw_meta,  # preserve harness-specific original for debugging
    }

    # Log Console API spend so we can enforce the $1000 cap
    if judge == "claude" and use_console_key and cost_usd is not None:
        log_console_usage(deliverable_id, cost_usd, model)

    json_path.write_text(json.dumps(result, indent=2) + "\n")

    # Save stdout and stderr.
    # In sandbox mode, <exp_dir>/stdout and stderr are siblings of results/;
    # otherwise, they live under output_dir (back-compat).
    if output_dir.name == "results" and output_dir.parent.exists():
        stdout_dir = output_dir.parent / "stdout"
        stderr_dir = output_dir.parent / "stderr"
    else:
        stdout_dir = output_dir / "stdout"
        stderr_dir = output_dir / "stderr"
    stdout_dir.mkdir(exist_ok=True)

    stdout_path = stdout_dir / f"{deliverable_id}.jsonl"
    if judge == "claude" and cfg["supports_session_id"]:
        # Claude: copy the full session transcript from ~/.claude/projects/
        project_dir_name = Path.cwd().as_posix().replace("/", "-")
        session_src = Path.home() / ".claude" / "projects" / project_dir_name / f"{session_id}.jsonl"
        if session_src.exists():
            shutil.copy2(session_src, stdout_path)
            # Claude Code creates session files mode 600 in ~/.claude/projects/.
            # shutil.copy2 preserves that, which blocks rsync access for collaborators.
            # Force 644 so the file is world-readable like the rest of results/.
            stdout_path.chmod(0o644)
        elif raw_output:
            stdout_path.write_text(raw_output)
    elif judge in ("codex", "opencode", "openhands", "gemini", "pi") and raw_output:
        stdout_path.write_text(raw_output)

    # For Gemini CLI, also copy the full chat transcript from
    # ~/.gemini/tmp/<project>/chats/session-*<sessionId>*.json alongside.
    # The CLI's stdout envelope only carries the final response+stats; the
    # chat file has the full messages[] and is namespaced by CWD project.
    if judge == "gemini" and cli_output:
        gemini_session_id = cli_output.get("session_id")
        if gemini_session_id:
            short_id = gemini_session_id.split("-")[0]
            project_name = Path.cwd().name  # sandbox dir name
            chats_dir = Path.home() / ".gemini" / "tmp" / project_name / "chats"
            chat_dst = stdout_dir / f"{deliverable_id}.chat.json"
            # Match any session-*<short_id>*.json (CLI uses a timestamp+short id)
            if chats_dir.exists():
                for cand in chats_dir.glob(f"session-*{short_id}*.json"):
                    try:
                        shutil.copy2(cand, chat_dst)
                        chat_dst.chmod(0o644)
                        break
                    except OSError:
                        pass

    # Save stderr only when non-empty (keeps dir small).
    if stderr_text:
        stderr_dir.mkdir(exist_ok=True)
        (stderr_dir / f"{deliverable_id}.log").write_text(stderr_text)

    # Distinguish clean verdicts from parse-fallback results for clearer logs.
    # If the model produced a parseable JSON verdict, result has `success`.
    # If all parse attempts failed, result has `raw_response` (possibly empty).
    if "success" in result:
        print(f"[{index}/{total}] [OK]   {deliverable_id}")
    elif (result.get("raw_response") or "").strip():
        print(f"[{index}/{total}] [PARSE_EMPTY] {deliverable_id} — unparseable model output (raw stored)")
    else:
        print(f"[{index}/{total}] [PARSE_EMPTY] {deliverable_id} — empty model output (check stderr)")
    return "ok"


def _preflight_tmpdir_check() -> None:
    """Abort if /tmp is dangerously full or if scratch tmpdir is missing.

    Experiments should never fill /tmp: Claude Code's bash tool, tmux sessions,
    and many CLI tempfiles will cascade-fail. We require ~/scratch/tmp to exist
    and be writable; subprocesses are directed there via TMPDIR in build_cmd.
    """
    import shutil as _shutil
    tmp_usage = _shutil.disk_usage("/tmp")
    tmp_used_pct = (tmp_usage.used / tmp_usage.total) * 100 if tmp_usage.total else 0
    if tmp_used_pct > 85:
        print(
            f"WARNING: /tmp is {tmp_used_pct:.0f}% full "
            f"({tmp_usage.free/1024/1024/1024:.1f} GB free). "
            f"Clean before running heavy evaluations to avoid cascade failures.",
            flush=True,
        )
    scratch_tmp = Path.home() / "scratch" / "tmp"
    if not scratch_tmp.exists():
        print(
            f"ERROR: scratch tmpdir {scratch_tmp} missing. "
            f"Run `mkdir -p {scratch_tmp}` before launching.",
            flush=True,
        )
        raise SystemExit(1)
    # Ensure scratch is writable
    if not os.access(str(scratch_tmp), os.W_OK):
        print(f"ERROR: scratch tmpdir {scratch_tmp} is not writable.", flush=True)
        raise SystemExit(1)


async def run(args: argparse.Namespace) -> None:
    _preflight_tmpdir_check()
    prompt_text = Path(args.prompt).read_text().strip()

    # Sandbox mode: provision an experiment dir, use its sandbox/ as CWD, results dir.
    sandbox_dir: Path | None = None
    exp_dir: Path | None = None
    if args.sandbox:
        # Default platform per-harness when --platform not given. This keeps
        # config.json and the dashboard's provider column populated for routine
        # launches (e.g. gemini-CLI always hits google_gemini_direct unless
        # overridden).
        platform = args.platform
        if not platform:
            if args.harness == "gemini":
                platform = "google_gemini_direct"
            elif args.harness == "claude":
                platform = "anthropic_console" if args.use_console_key else "anthropic_subscription"
        # --continue: auto-resume the most recent matching exp_id.
        # Mutually exclusive with --resume-exp (manual id wins if both given).
        if getattr(args, "continue_exp", False):
            if args.resume_exp:
                print(f"[--continue] ignored because --resume-exp={args.resume_exp} "
                      "was also passed; honouring the explicit id.", flush=True)
            else:
                match = _find_matching_exp(
                    harness=args.harness,
                    model=args.model,
                    prompt_path=Path(args.prompt),
                    platform=platform,
                )
                if match:
                    print(f"[--continue] resuming exp {match}", flush=True)
                    args.resume_exp = match
                else:
                    print("[--continue] no match found, will create fresh exp", flush=True)
        cfg = _provision_experiment(
            harness=args.harness,
            model=args.model,
            prompt_path=Path(args.prompt),
            exp_id=args.resume_exp,
            notes=args.note or "",
            extra_config={
                "platform": platform,
                "effort": args.effort,
                "uses_console_key": bool(args.use_console_key),
                "workers": args.workers,
            },
        )
        exp_dir = Path(cfg["path"])
        sandbox_dir = exp_dir / "sandbox"
        output_dir = exp_dir / "results"
        # md_dir comes from the sandbox by default in sandbox mode
        md_dir = sandbox_dir / "agenthorizon_md" if args.md_dir is None else Path(args.md_dir)
        print(f"[sandbox] exp_id = {cfg['exp_id']}")
        print(f"[sandbox] path   = {exp_dir}")
        print(f"[sandbox] cwd    = {sandbox_dir}")
    else:
        if args.output_dir is None or args.md_dir is None:
            print("ERROR: with --no-sandbox, both --md-dir and --output-dir are required.", flush=True)
            return
        md_dir = Path(args.md_dir)
        output_dir = Path(args.output_dir)

    md_files = sorted(md_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {md_dir}")
        return

    # Split filter: restrict to val / test / public_test / private_test / full
    # (default: full = no filter). --splits (multi) takes precedence over
    # --split (single) so a single exp_id can cover multiple splits.
    if args.splits:
        split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
        valid = {"val", "test", "public_test", "private_test", "full"}
        bad = [s for s in split_names if s not in valid]
        if bad:
            print(f"ERROR: unknown split(s) in --splits: {bad}", flush=True)
            return
    else:
        split_names = [args.split]

    # 'full' means "no filter" — if any of the listed splits is 'full', skip
    # the filter entirely. Otherwise union all the listed splits' IDs.
    if "full" in split_names:
        split_label = "full"
    else:
        # Per-split label files were consolidated into a single full
        # `agenthorizon_labels.jsonl` on 2026-04-30. Source split membership
        # from the per-split DATA files (`agenthorizon_<split>.jsonl`),
        # which still ship in data/standard/.
        split_ids: set[str] = set()
        for s in split_names:
            split_data = REPO_ROOT / "data" / "standard" / f"agenthorizon_{s}.jsonl"
            if not split_data.exists():
                print(f"ERROR: split='{s}' but {split_data} not found", flush=True)
                return
            for line in split_data.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    split_ids.add(json.loads(line)["trajectory_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        before = len(md_files)
        md_files = [f for f in md_files if f.stem in split_ids]
        split_label = "+".join(split_names) if len(split_names) > 1 else split_names[0]
        print(f"[splits={split_label}] filtered {before} -> {len(md_files)} md files")

    if args.ids_file:
        ids_path = Path(args.ids_file)
        if not ids_path.exists():
            print(f"ERROR: --ids-file {ids_path} not found", flush=True)
            return
        allowed_ids = {line.strip() for line in ids_path.read_text().splitlines() if line.strip()}
        before = len(md_files)
        md_files = [f for f in md_files if f.stem in allowed_ids]
        print(f"[--ids-file={ids_path.name}] filtered {before} -> {len(md_files)} md files")

    if args.limit is not None:
        md_files = md_files[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(md_files)

    # Resolve Console API key for Claude (billed via Console credits, capped at $1000)
    console_key = None
    if args.harness == "claude" and args.use_console_key:
        console_key = _load_console_key()
        if not console_key:
            print("ERROR: --use-console-key set but CONSOLE_ANTHROPIC_API_KEY not found in ~/.claude/.env", flush=True)
            return
        # Resolve effective cap (CLI override or built-in default)
        global _CAP_USD_OVERRIDE
        _CAP_USD_OVERRIDE = args.cap_usd
        effective_cap = args.cap_usd if args.cap_usd is not None else CONSOLE_SPEND_CAP_USD
        current_spend = get_console_cumulative_spend()
        print(f"Console API spend so far: ${current_spend:.2f} / ${effective_cap:.2f}", flush=True)
        if current_spend >= effective_cap:
            print(f"ERROR: Console API spend cap (${effective_cap}) already reached. Refusing to launch.", flush=True)
            return
        if current_spend >= 0.8 * effective_cap:
            print(f"WARNING: Console API spend at {current_spend/effective_cap*100:.0f}% of cap. Proceed with caution.", flush=True)

    # Codex (ChatGPT subscription) weekly rate-limit pre-flight. Mirrors the
    # Console pre-flight above but for the weekly window reported by Codex CLI
    # in its session token_count events. Only enforced when the user explicitly
    # declared --platform chatgpt_subscription so OpenRouter-routed Codex runs
    # (Qwen, GLM, Gemma, Gemini) are unaffected.
    codex_weekly_watch = (args.harness == "codex" and args.platform == "chatgpt_subscription")
    if codex_weekly_watch:
        rl = get_codex_weekly_usage()
        if rl is None:
            print(
                "[codex-cap] No recent OpenAI-authed Codex session with weekly rate-limit data found. "
                "This is expected if the previous weekly window rolled over or if you haven't used "
                "`codex login` recently. Proceeding; the watchdog will pick up live rate limits once "
                "the first trajectory finishes.",
                flush=True,
            )
        else:
            pct = get_codex_weekly_used_percent()
            resets_at = (rl.get("secondary") or {}).get("resets_at")
            resets_str = (
                datetime.fromtimestamp(int(resets_at), tz=timezone.utc).isoformat()
                if resets_at else "unknown"
            )
            plan = rl.get("plan_type") or "?"
            print(
                f"Codex weekly usage: {pct:.1f}% / {CODEX_WEEKLY_CAP_PERCENT:.0f}% "
                f"(plan={plan}, window resets {resets_str})",
                flush=True,
            )
            if pct is not None and pct >= CODEX_WEEKLY_CAP_PERCENT:
                print(
                    f"ERROR: Codex weekly usage ({pct:.1f}%) already at or above "
                    f"cap ({CODEX_WEEKLY_CAP_PERCENT:.0f}%). Refusing to launch. "
                    f"Window resets {resets_str}.",
                    flush=True,
                )
                return
            if pct is not None and pct >= CODEX_WEEKLY_WARN_PERCENT:
                print(
                    f"WARNING: Codex weekly usage at {pct:.1f}% of cap. Proceed with caution; "
                    f"the watchdog will abort if it hits {CODEX_WEEKLY_CAP_PERCENT:.0f}%.",
                    flush=True,
                )

    print(f"Evaluating {total} trajectories from {md_dir}")
    print(f"  Harness: {args.harness}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Model: {args.model}")
    if args.effort:
        print(f"  Effort: {args.effort}")
    if console_key:
        print(f"  Auth: Console API (--bare, key from ~/.claude/.env)")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_dir}")
    if args.dry_run:
        print(f"  *** DRY RUN — no {args.harness} calls will be made ***")
    print()

    semaphore = asyncio.Semaphore(args.workers)
    start = time.time()

    # Start the spend-cap watchdog(s). Both share the same _CAP_ABORT_EVENT so
    # whichever limit fires first aborts the batch. Console watchdog runs for
    # any --use-console-key Claude run; Codex weekly watchdog runs for any
    # codex + chatgpt_subscription run.
    global _CAP_ABORT_EVENT
    watchdog_tasks: list[asyncio.Task] = []
    if console_key or codex_weekly_watch:
        _CAP_ABORT_EVENT = asyncio.Event()
    if console_key:
        watchdog_tasks.append(asyncio.create_task(_cap_watchdog()))
        eff_cap = _CAP_USD_OVERRIDE if _CAP_USD_OVERRIDE is not None else CONSOLE_SPEND_CAP_USD
        print(f"[watchdog] Spend cap checker running every {CONSOLE_SPEND_CHECK_INTERVAL_SEC}s (cap ${eff_cap})", flush=True)
    if codex_weekly_watch:
        watchdog_tasks.append(asyncio.create_task(_codex_weekly_watchdog()))
        print(f"[watchdog] Codex weekly-usage checker running every {CODEX_WEEKLY_CHECK_INTERVAL_SEC}s (cap {CODEX_WEEKLY_CAP_PERCENT:.0f}%)", flush=True)

    tasks = [
        evaluate_one(
            md_path=md_path,
            output_dir=output_dir,
            prompt_text=prompt_text,
            model=args.model,
            judge=args.harness,
            semaphore=semaphore,
            index=i + 1,
            total=total,
            dry_run=args.dry_run,
            max_retries=args.max_retries,
            effort=args.effort,
            console_key=console_key,
            sandbox_dir=sandbox_dir,
            platform=args.platform,
        )
        for i, md_path in enumerate(md_files)
    ]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        if watchdog_tasks:
            if _CAP_ABORT_EVENT is not None:
                _CAP_ABORT_EVENT.set()
            for wt in watchdog_tasks:
                wt.cancel()
            for wt in watchdog_tasks:
                try:
                    await wt
                except asyncio.CancelledError:
                    pass

    elapsed = time.time() - start
    counts = {"total": total, "ok": 0, "skip": 0, "error": 0}
    for r in results:
        counts[r] += 1

    print()
    print(f"Done in {elapsed:.1f}s — OK: {counts['ok']}, Skip: {counts['skip']}, Error: {counts['error']}")

    if not args.dry_run:
        # In sandbox mode, config.json is the single source of truth for run metadata.
        # For the legacy flow (no sandbox), keep a minimal summary.json next to results.
        if exp_dir is not None:
            _mark_experiment_complete(exp_dir, counts)
        else:
            summary_path = output_dir / "summary.json"
            summary = {
                "prompt_file": str(args.prompt),
                "md_dir": str(md_dir),
                "harness": args.harness,
                "model": args.model,
                "workers": args.workers,
                "elapsed_seconds": round(elapsed, 1),
                **counts,
            }
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"Summary written to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory markdown files using an LLM judge CLI."
    )
    parser.add_argument("--prompt", default="prompts/evaluate_trajectory.txt", help="Path to evaluation prompt file (default: prompts/evaluate_trajectory.txt, which is the P6 variant as of 2026-04-24; the pre-P6 variant is archived at prompts/archive/evaluate_trajectory_v0.txt).")
    parser.add_argument("--md-dir", default=None, help="Directory containing .md trajectory files (required with --no-sandbox)")
    parser.add_argument("--output-dir", default=None, help="Where to write per-trajectory result files (required with --no-sandbox)")
    parser.add_argument(
        "--sandbox", action=argparse.BooleanOptionalAction, default=True,
        help="Sandbox mode (DEFAULT): auto-provision experiments/<harness>/<model>/<exp_id>/ and run subprocess from its sandbox/ dir. "
             "Prevents leakage of project context (CLAUDE.md, etc.) and guarantees a reproducible CWD per experiment. "
             "Pass --no-sandbox to use the legacy flow requiring explicit --md-dir and --output-dir.",
    )
    parser.add_argument("--resume-exp", default=None, help="Resume an existing experiment by exp_id")
    parser.add_argument("--continue", dest="continue_exp", action="store_true",
                        help="Auto-resume the most recent matching exp_id for this "
                             "(harness, model, platform, prompt_hash, sandbox_version) "
                             "combination. Avoids fragmenting one logical experiment "
                             "across multiple exp_ids when adding splits in phases. "
                             "Refuses (with a helpful error) if the matching exp "
                             "already has 1700 result files on disk (= full coverage). "
                             "Prints '[--continue] resuming exp <id>' or "
                             "'[--continue] no match found, fresh exp' so the "
                             "behavior is visible. Mutually exclusive with --resume-exp.")
    parser.add_argument("--note", default="", help="Free-form note saved in config.json and exp_ids.jsonl")
    def _platform_validator(s: str) -> str:
        # Self-hosted vLLM on YUL201 uses dynamic platform strings of the form
        # `vllm_yul201_<preset>` which are mapped to codex provider overrides
        # at build_cmd time. Accept any such string outside the closed set.
        if s in VALID_PLATFORMS or s.startswith("vllm_yul201_"):
            return s
        raise argparse.ArgumentTypeError(
            f"platform must be one of {sorted(VALID_PLATFORMS)} "
            f"or start with 'vllm_yul201_'; got {s!r}"
        )
    parser.add_argument(
        "--platform",
        type=_platform_validator,
        default=None,
        help="Explicit API platform/provider (for reproducibility and paper tables). "
             "Examples: anthropic_console, anthropic_subscription, google_gemini_direct, "
             "openrouter, openrouter_byok, chatgpt_subscription, or vllm_yul201_<preset>. "
             "Left null when not specified; declare it explicitly rather than inferring.",
    )
    parser.add_argument("--harness", default="claude", choices=list(HARNESS_CONFIG.keys()),
                        help="Judge CLI to use: claude, gemini, or codex (default: claude)")
    parser.add_argument("--workers", type=int, default=1, help="Max concurrent judge subprocesses (default: 1)")
    parser.add_argument(
        "--model", default="sonnet",
        help="Model name passed to the judge CLI --model flag (default: sonnet)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N trajectories (for testing)")
    parser.add_argument("--cap-usd", type=float, default=None,
                        help="Override the Console API spend cap for this launch "
                             "(default: CONSOLE_SPEND_CAP_USD=$1000). Only effective "
                             "for --use-console-key runs. Watchdog aborts at this value.")
    parser.add_argument("--ids-file", default=None,
                        help="Path to a text file with one trajectory_id per line. "
                             "Restrict evaluation to only those IDs (applied after --split, before --limit).")
    parser.add_argument("--split", default="full",
                        choices=["val", "test", "public_test", "private_test", "full"],
                        help="Restrict to val (200), public_test (1000), private_test (500), "
                             "test (1500), or full (1700) trajectories. "
                             "Uses data/standard/agenthorizon_<split>.jsonl to get IDs.")
    parser.add_argument("--splits", default=None,
                        help="Multi-split mode (overrides --split). Comma-separated "
                             "list of splits whose trajectory IDs are unioned and "
                             "scored within ONE exp_id. e.g. "
                             "`--splits val,public_test,private_test` runs all 1700 "
                             "in one exp_dir. The UI/experiment-status skill still "
                             "groups results by trajectory_id → split for display, "
                             "so you see one row per split but only one underlying "
                             "experiment. Use this when you would otherwise launch "
                             "the same model+harness three times in succession; "
                             "lands all results in one place. "
                             "Resume across splits is also supported via "
                             "`--resume-exp <id>` with a different `--split` value.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries on JSON parse failure (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without calling the judge")
    parser.add_argument(
        "--effort",
        choices=["xhigh", "high", "medium", "low", "minimal"],
        default=None,
        help="Reasoning effort for claude judge (e.g. xhigh for Opus 4.7, high for Opus 4.6)",
    )
    parser.add_argument(
        "--use-console-key",
        action="store_true",
        help="Use Console API key from ~/.claude/.env (CONSOLE_ANTHROPIC_API_KEY) "
             "instead of subscription. Forces --bare to prevent OAuth fallback. "
             "Enforces $1000 cumulative spend cap.",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
