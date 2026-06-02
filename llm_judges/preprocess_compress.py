#!/usr/bin/env python3
"""Approach A: Naive Compression -- compress all screenshots to fit in context.

Resizes every screenshot to a small resolution so the entire trajectory
(text + all images) fits within the token budget. Every screenshot is
included but at reduced quality.

Input:
    - Trajectory JSONL (data/standard/agenthorizon.jsonl)
    - Screenshot directory (data/media/images/)

Output:
    - One JSON file per trajectory in the output directory, containing:
      {
        "trajectory_id": str,
        "messages": [
            {"role": "system", "content": str},
            {"role": "user", "content": [
                {"type": "text", "text": str},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                ...
            ]}
        ],
        "token_estimate": int,
        "num_screenshots": int,
        "image_resolution": [width, height]
      }

Usage:
    # Preprocess all trajectories
    uv run python llm_judges/preprocess_compress.py \
        --input data/standard/agenthorizon.jsonl \
        --output-dir data/preprocessed/compress \
        --max-tokens 60000

    # Test on a small sample
    uv run python llm_judges/preprocess_compress.py \
        --input data/standard/agenthorizon.jsonl \
        --output-dir data/preprocessed/compress \
        --max-tokens 60000 \
        --limit 5

    # Dry run (just estimate sizes)
    uv run python llm_judges/preprocess_compress.py \
        --input data/standard/agenthorizon.jsonl \
        --output-dir data/preprocessed/compress \
        --dry-run --limit 10
"""

import argparse
import functools
import json
import math
import sys
from pathlib import Path

# Ensure the project root is on the path so we can import from llm_judges
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_judges.utils import (
    build_instruction_text,
    estimate_image_tokens,
    estimate_text_tokens,
    format_step_text,
    get_screenshot_path,
    load_and_resize_image,
    image_to_base64,
    load_local_image_map,
    load_trajectories,
)

print = functools.partial(print, flush=True)

# Evaluation prompt -- AgentHorizon judge framework, 4-field output schema
EVAL_PROMPT = (
    "You are evaluating a trajectory of an agent performing a computer task. "
    "The trajectory includes:\n"
    "- A Goal describing what the agent was supposed to accomplish\n"
    "- A sequence of Steps showing the actions taken (clicks, typing, key presses, etc.)\n\n"
    "## Judging framework\n\n"
    "A task has three layers:\n"
    "- **Core Objective** (What + Who): Primary Action + Primary Entity. Changing it -> Critical Mistake.\n"
    "- **Parameters** (When, Which, Where): Scope and data points. Changing them -> Bad Side Effect or Misunderstanding.\n"
    "- **Constraints** (Format, Style): How it looks. Changing them -> Misunderstanding.\n\n"
    "## Mistake types\n"
    "- **Critical Mistake** -- core objective not achieved; user must restart.\n"
    "- **Bad Side Effect** -- goal achieved but with unwanted baggage to delete (>5 min cleanup or external cost).\n"
    "- **Misunderstanding of the Instruction** -- agent took a reasonable action but misinterpreted a detail (<1 min fix).\n\n"
    "## Verdict rule\n"
    "- success = true when all three layers are satisfied.\n"
    "- success = false when any mistake type applies. A Misunderstanding is still a failure.\n\n"
    "## Output\n\n"
    "Be decisive. Respond with a JSON object only:\n"
    "{\n"
    '  "success": true | false,\n'
    '  "reasoning": "1-3 sentences citing which layer (core objective, parameters, or constraints) was or was not satisfied.",\n'
    '  "confidence": "low" | "medium" | "high",\n'
    '  "mistake_type": null when success=true, else one of: "Critical Mistake", "Bad Side Effect", "Misunderstanding of the Instruction"\n'
    "}\n\n"
    "Notes: confidence is evidence strength, not verdict. Low confidence != failure. mistake_type is null on success."
)


def compute_target_resolution(
    trajectory: dict,
    base_dir: Path,
    max_tokens: int,
    min_width: int = 64,
    max_width: int = 512,
    local_image_id: str | None = None,
) -> tuple[int, int]:
    """Binary-search for the largest image resolution that fits the budget.

    We keep the aspect ratio fixed at 1710:1112 (the standard screenshot
    ratio in this dataset) and find the widest image such that
    text_tokens + N * image_tokens <= max_tokens.

    Returns:
        (target_width, target_height)
    """
    # Count screenshots and estimate text tokens
    steps = trajectory.get("steps", [])
    n_screenshots = 0
    for step in steps:
        screenshot_path = get_screenshot_path(step, base_dir, local_image_id=local_image_id)
        if screenshot_path is not None:
            n_screenshots += 1

    if n_screenshots == 0:
        return max_width, int(max_width * 1112 / 1710)

    # Estimate text tokens (instruction + all step text)
    instruction = build_instruction_text(trajectory)
    text_parts = [EVAL_PROMPT, f"## Goal\n\n{instruction}\n\n## Steps\n\nTotal steps: **{len(steps)}**\n"]
    for i, step in enumerate(steps):
        text_parts.append(format_step_text(step, i))
    total_text = "\n\n".join(text_parts)
    text_tokens = estimate_text_tokens(total_text)

    # Budget for images
    image_budget = max_tokens - text_tokens - 200  # 200 token buffer
    if image_budget <= 0:
        return min_width, int(min_width * 1112 / 1710)

    tokens_per_image = image_budget / n_screenshots

    # Binary search for resolution
    aspect_ratio = 1112 / 1710  # height/width
    lo, hi = min_width, max_width
    best_w = min_width

    while lo <= hi:
        mid = (lo + hi) // 2
        h = int(mid * aspect_ratio)
        img_tokens = estimate_image_tokens(mid, h)
        if img_tokens <= tokens_per_image:
            best_w = mid
            lo = mid + 1
        else:
            hi = mid - 1

    best_h = int(best_w * aspect_ratio)
    return best_w, best_h


def preprocess_one(
    trajectory: dict,
    base_dir: Path,
    max_tokens: int,
    target_width: int | None = None,
    target_height: int | None = None,
    local_image_id: str | None = None,
) -> dict:
    """Preprocess a single trajectory with compressed screenshots.

    Returns the preprocessed payload dict ready for API calls.
    """
    trajectory_id = trajectory["trajectory_id"]
    instruction = build_instruction_text(trajectory)
    steps = trajectory.get("steps", [])

    # Auto-compute resolution if not provided
    if target_width is None or target_height is None:
        target_width, target_height = compute_target_resolution(
            trajectory, base_dir, max_tokens, local_image_id=local_image_id,
        )

    # Build the multimodal content array
    content_parts = []

    # Header text
    header = (
        f"## Goal\n\n{instruction}\n\n"
        f"## Steps\n\nTotal steps: **{len(steps)}**\n"
    )
    content_parts.append({"type": "text", "text": header})

    total_tokens = estimate_text_tokens(EVAL_PROMPT) + estimate_text_tokens(header)
    n_screenshots = 0

    for i, step in enumerate(steps):
        step_text = format_step_text(step, i)
        content_parts.append({"type": "text", "text": step_text})
        total_tokens += estimate_text_tokens(step_text)

        # Add compressed screenshot if available
        screenshot_path = get_screenshot_path(step, base_dir, local_image_id=local_image_id)
        if screenshot_path is not None:
            try:
                img_bytes, w, h = load_and_resize_image(
                    screenshot_path, target_width, target_height, quality=60
                )
                b64 = image_to_base64(img_bytes)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                    },
                })
                total_tokens += estimate_image_tokens(w, h)
                n_screenshots += 1
            except Exception as e:
                content_parts.append({
                    "type": "text",
                    "text": f"[Screenshot unavailable: {e}]",
                })

    messages = [
        {"role": "system", "content": EVAL_PROMPT},
        {"role": "user", "content": content_parts},
    ]

    return {
        "trajectory_id": trajectory_id,
        "messages": messages,
        "token_estimate": total_tokens,
        "num_screenshots": n_screenshots,
        "image_resolution": [target_width, target_height],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Approach A: Compress all screenshots to fit in context window."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to trajectory JSONL (e.g., data/standard/agenthorizon.jsonl)"
    )
    parser.add_argument(
        "--labels", default="data/standard/agenthorizon_labels.jsonl",
        help="Labels JSONL for trajectory_id -> original_id mapping (used only for "
             "screenshot path resolution; verdict fields are never read). "
             "Default: data/standard/agenthorizon_labels.jsonl"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write preprocessed JSON files"
    )
    parser.add_argument(
        "--base-dir", default=".",
        help="Base directory for resolving screenshot paths (default: .)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=60000,
        help="Maximum token budget for input (default: 60000)"
    )
    parser.add_argument(
        "--target-width", type=int, default=None,
        help="Override: fixed target width for all screenshots (auto-computed if not set)"
    )
    parser.add_argument(
        "--target-height", type=int, default=None,
        help="Override: fixed target height for all screenshots (auto-computed if not set)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N trajectories (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just estimate token counts, don't write output files"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(args.input, limit=args.limit)
    print(f"Loaded {len(trajectories)} trajectories from {args.input}")
    image_map = load_local_image_map(args.labels)
    print(f"Loaded local_image_id mapping for {len(image_map)} trajectories from {args.labels}")
    print(f"Max tokens: {args.max_tokens}")
    if args.target_width and args.target_height:
        print(f"Fixed resolution: {args.target_width}x{args.target_height}")
    else:
        print(f"Resolution: auto-computed per trajectory")
    print()

    token_counts = []
    screenshot_counts = []
    resolutions = []

    for i, traj in enumerate(trajectories):
        tid = traj["trajectory_id"]
        n_steps = len(traj.get("steps", []))

        result = preprocess_one(
            traj,
            base_dir,
            args.max_tokens,
            target_width=args.target_width,
            target_height=args.target_height,
            local_image_id=image_map.get(tid),
        )

        token_counts.append(result["token_estimate"])
        screenshot_counts.append(result["num_screenshots"])
        resolutions.append(tuple(result["image_resolution"]))

        label = f"[{i+1}/{len(trajectories)}]"
        res = f"{result['image_resolution'][0]}x{result['image_resolution'][1]}"
        print(
            f"{label} {tid[:12]}... "
            f"steps={n_steps} screenshots={result['num_screenshots']} "
            f"res={res} tokens~{result['token_estimate']}"
        )

        if not args.dry_run:
            out_path = output_dir / f"{tid}.json"
            # Write without base64 data for the metadata (too large)
            # The full payload is used by evaluate.py which re-reads this
            out_path.write_text(json.dumps(result, indent=None) + "\n")

    print()
    print("=== Summary ===")
    if token_counts:
        import statistics
        print(f"Token estimates: mean={statistics.mean(token_counts):.0f}, "
              f"median={statistics.median(token_counts):.0f}, "
              f"max={max(token_counts):.0f}")
        print(f"Screenshots:    mean={statistics.mean(screenshot_counts):.0f}, "
              f"median={statistics.median(screenshot_counts):.0f}")
        over_budget = sum(1 for t in token_counts if t > args.max_tokens)
        print(f"Over budget:    {over_budget}/{len(token_counts)}")
        unique_res = set(resolutions)
        if len(unique_res) <= 5:
            print(f"Resolutions:    {', '.join(f'{w}x{h}' for w,h in sorted(unique_res))}")
        else:
            print(f"Resolutions:    {len(unique_res)} unique values")
    if not args.dry_run:
        print(f"\nOutput written to {output_dir}/")


if __name__ == "__main__":
    main()
