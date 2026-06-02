"""Download screenshots from standardized trajectory JSONL to local storage.

Reads a standard-format JSONL file and downloads all screenshot URLs to a local
directory, organized by trajectory UUID: <output_dir>/<uuid>/step_<N>.png

Usage:
    uv run python scripts/download_standard_screenshots.py \
        --input data/standard/agenthorizon.jsonl \
        --output-dir data/standard/human_images \
        --workers 32
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_one(url: str, dest: str) -> tuple[str, bool, str]:
    """Download a single URL to dest. Returns (url, success, error_msg)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return (url, True, "skipped")
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urlretrieve(url, dest)
        return (url, True, "")
    except Exception as e:
        return (url, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Download screenshots from standardized trajectory JSONL"
    )
    parser.add_argument(
        "--input", required=True, help="Path to standard JSONL file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save screenshots"
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Parallel download threads (default: 32)"
    )
    args = parser.parse_args()

    logger.info("Loading %s", args.input)
    tasks = []  # (url, dest_path)
    with open(args.input) as f:
        for line in f:
            traj = json.loads(line)
            tid = traj["trajectory_id"]
            for step in traj.get("steps", []):
                url = step.get("screenshot")
                if url:
                    step_id = step["step_id"]
                    ext = os.path.splitext(url)[-1] or ".png"
                    dest = os.path.join(args.output_dir, tid, f"step_{step_id}{ext}")
                    tasks.append((url, dest))

    logger.info("Found %d screenshots to download", len(tasks))

    ok = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, url, dest): (url, dest) for url, dest in tasks}
        for i, future in enumerate(as_completed(futures)):
            url, success, msg = future.result()
            if success:
                if msg == "skipped":
                    skipped += 1
                else:
                    ok += 1
            else:
                errors += 1
                if errors <= 10:
                    logger.warning("Failed: %s — %s", url[:80], msg)

            total = ok + skipped + errors
            if total % 5000 == 0:
                logger.info(
                    "Progress: %d/%d (ok=%d, skip=%d, err=%d)",
                    total, len(tasks), ok, skipped, errors,
                )

    logger.info(
        "Done: %d downloaded, %d skipped, %d errors (total %d)",
        ok, skipped, errors, len(tasks),
    )


if __name__ == "__main__":
    main()
