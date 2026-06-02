"""Render standardized trajectory JSONL to per-trajectory files.

Reads a standard-format JSONL file (one trajectory per line) and produces one
file per trajectory in the chosen format (Markdown, JSON, or both). Output
contains only the task instruction and step-by-step actions -- NO labels or
metadata that could reveal ground truth.

Usage:
    # Both formats (default)
    uv run python scripts/render_trajectories.py \
        data/standard/agenthorizon.jsonl \
        data/standard/agenthorizon

    # Markdown only
    uv run python scripts/render_trajectories.py \
        data/standard/agenthorizon.jsonl \
        data/standard/agenthorizon_md \
        --format md

    # JSON only
    uv run python scripts/render_trajectories.py \
        data/standard/agenthorizon.jsonl \
        data/standard/agenthorizon_json \
        --format json

Options:
    --format {md,json,both}  Output format(s) to generate (default: both).
    --index N                Convert only the trajectory at line index N.
    --id UUID                Convert only the trajectory with the given trajectory_id.
    --no-images              Omit screenshot image links from the Markdown output.

With --format both, two sibling directories are created:
    <output_dir>_md/   for Markdown files
    <output_dir>_json/ for JSON files
(If <output_dir> already ends in _md or _json, that suffix is stripped first.)
"""

import argparse
import json
from pathlib import Path


def _format_action(action: dict) -> str:
    """Format a standard-format action into a human-readable string."""
    action_type = action.get("type", "unknown")
    params = action.get("parameters", {})

    if action_type == "click":
        button = params.get("button", "left")
        x, y = params.get("x", "?"), params.get("y", "?")
        n = params.get("num_clicks", 1)
        click_str = f"{'right' if button == 'right' else 'left'}-click"
        suffix = f" x{n}" if n and n > 1 else ""
        return f"{click_str} ({x}, {y}){suffix}"

    if action_type == "type":
        text = params.get("text", "")
        if len(text) > 80:
            text = text[:77] + "..."
        return f'type "{text}"'

    if action_type == "press":
        keys = params.get("keys", [])
        return f"press {' '.join(keys)}"

    if action_type == "hotkey":
        keys = params.get("keys", [])
        return f"hotkey {'+'.join(keys)}"

    if action_type == "scroll":
        direction = params.get("direction", "?")
        amount = params.get("amount", "?")
        x, y = params.get("x", "?"), params.get("y", "?")
        return f"scroll {direction} {amount}px at ({x}, {y})"

    if action_type == "drag":
        sx = params.get("start_x", "?")
        sy = params.get("start_y", "?")
        ex = params.get("end_x", "?")
        ey = params.get("end_y", "?")
        return f"drag ({sx}, {sy}) -> ({ex}, {ey})"

    if action_type in ("key_down", "key_up"):
        key = params.get("key", "?")
        return f"{action_type.replace('_', '')} {key}"

    return action_type


def render_markdown(
    trajectory: dict,
    output_path: Path,
    *,
    include_images: bool = True,
    local_image_id: str | None = None,
    local_image_root: str = "./data/media/images",
) -> None:
    """Render a single trajectory to a Markdown file.

    If `local_image_id` is provided, screenshot links use the local path
    `<local_image_root>/<local_image_id>/step_<step_id>.png` instead of the
    URL stored in the trajectory's `screenshot` field.
    """
    task = trajectory.get("task", {})
    steps = trajectory.get("steps", [])
    lines: list[str] = []

    lines.append("# Trajectory Report")
    lines.append("")

    instruction = task.get("instruction", "")
    lines.append("## Goal")
    lines.append("")
    lines.append(instruction.strip())
    lines.append("")

    lines.append("## Steps")
    lines.append("")
    lines.append(f"Total steps: **{len(steps)}**")
    lines.append("")

    for step in steps:
        step_id = step.get("step_id", "?")
        action = step.get("action", {})
        action_str = _format_action(action)
        timestamp_us = step.get("timestamp_us")

        lines.append("---")
        lines.append("")
        lines.append(f"### Step {step_id}")
        lines.append("")

        if include_images:
            if local_image_id is not None:
                # Source data uses 1-indexed step files (step_1.png..step_N.png).
                # Standard JSONL uses 0-indexed step_id from the iteration loop.
                # Add 1 to match the local file naming.
                try:
                    file_index = int(step_id) + 1
                except (ValueError, TypeError):
                    file_index = step_id
                local_path = f"{local_image_root}/{local_image_id}/step_{file_index}.png"
                lines.append(f"![Step {step_id} screenshot]({local_path})")
                lines.append("")
            else:
                screenshot = step.get("screenshot")
                if screenshot:
                    lines.append(f"![Step {step_id} screenshot]({screenshot})")
                    lines.append("")

        lines.append(f"**Action:** `{action_str}`")
        lines.append("")

        if timestamp_us is not None:
            try:
                ts_ms = int(timestamp_us) / 1000
                lines.append(f"**Timestamp:** {ts_ms:.0f} ms")
            except (ValueError, TypeError):
                pass
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def render_json(
    trajectory: dict,
    output_path: Path,
    *,
    local_image_id: str | None = None,
    local_image_root: str = "./data/media/images",
) -> None:
    """Render a single trajectory to a JSON file (structured, no labels).

    If `local_image_id` is provided, each step's `screenshot` field is
    rewritten to the local path (matching the markdown renderer).
    """
    steps = trajectory.get("steps", [])
    if local_image_id is not None:
        rewritten = []
        for step in steps:
            new_step = dict(step)
            step_id = new_step.get("step_id")
            try:
                file_index = int(step_id) + 1
            except (ValueError, TypeError):
                file_index = step_id
            new_step["screenshot"] = (
                f"{local_image_root}/{local_image_id}/step_{file_index}.png"
            )
            rewritten.append(new_step)
        steps = rewritten

    # Blind-safe projection: only the fields a judge should see.
    payload = {
        "trajectory_id": trajectory.get("trajectory_id"),
        "task": trajectory.get("task", {}),
        "environment": trajectory.get("environment"),
        "steps": steps,
    }
    # Drop None values so the JSON is compact
    payload = {k: v for k, v in payload.items() if v is not None}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


def _resolve_output_dirs(
    output_dir: Path, formats: list[str]
) -> dict[str, Path]:
    """Pick output directories per format.

    - If only one format is requested, use output_dir as-is.
    - If both formats are requested, derive sibling `_md` / `_json` dirs:
        * Strip a trailing `_md` or `_json` from output_dir if present
        * Append `_md` and `_json` to the base name
    """
    if len(formats) == 1:
        fmt = formats[0]
        return {fmt: output_dir}

    base = output_dir
    name = base.name
    for suffix in ("_md", "_json"):
        if name.endswith(suffix):
            base = base.with_name(name[: -len(suffix)])
            break
    return {
        "md": base.with_name(base.name + "_md"),
        "json": base.with_name(base.name + "_json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render standardized trajectory JSONL to Markdown and/or JSON files."
    )
    parser.add_argument("input", help="Path to the standard JSONL file")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output directory. With --format both, the dir name is used as a base and _md/_json suffixes are appended.",
    )
    parser.add_argument(
        "--format",
        choices=["md", "json", "both"],
        default="both",
        help="Output format(s) to generate (default: both)",
    )
    parser.add_argument("--index", type=int, default=None, help="Convert only the trajectory at this line index")
    parser.add_argument("--id", dest="trajectory_id", default=None, help="Convert only the trajectory with this UUID")
    parser.add_argument("--no-images", action="store_true", help="Omit screenshot image links (Markdown only)")
    parser.add_argument(
        "--labels",
        default=None,
        help=(
            "Path to agenthorizon_labels.jsonl. When provided, screenshot links "
            "are rewritten to local paths using the label's `original_id` field."
        ),
    )
    parser.add_argument(
        "--local-image-root",
        default="./data/media/images",
        help="Prefix for local screenshot paths when --labels is set (default: ./data/media/images)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    include_images = not args.no_images
    formats = ["md", "json"] if args.format == "both" else [args.format]

    if args.output_dir is None:
        # Default: strip .jsonl, use that as the base
        output_dir = input_path.with_suffix("")
    else:
        output_dir = Path(args.output_dir)

    output_dirs = _resolve_output_dirs(output_dir, formats)

    print(f"Loading {input_path}...")
    with open(input_path) as f:
        trajectories = [json.loads(line) for line in f]
    print(f"Loaded {len(trajectories)} trajectories")
    print(f"Formats: {', '.join(formats)}")
    for fmt, d in output_dirs.items():
        print(f"  {fmt} -> {d}/")

    # Optional: build trajectory_id -> original_id map from labels for local paths.
    id_to_local: dict[str, str] = {}
    if args.labels:
        labels_path = Path(args.labels)
        with open(labels_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                tid = rec.get("trajectory_id")
                oid = rec.get("original_id")
                if tid and oid:
                    id_to_local[tid] = oid
        print(f"Loaded local_image map for {len(id_to_local)} trajectories from {labels_path}")

    # Narrow to a single trajectory if --index or --id provided
    if args.index is not None:
        selected = [trajectories[args.index]]
    elif args.trajectory_id is not None:
        selected = [t for t in trajectories if t.get("trajectory_id") == args.trajectory_id]
        if not selected:
            print(f"Error: trajectory_id '{args.trajectory_id}' not found")
            return
    else:
        selected = trajectories

    print(f"Rendering {len(selected)} trajectories...")
    for traj in selected:
        tid = traj.get("trajectory_id", "unknown")
        if "md" in formats:
            render_markdown(
                traj,
                output_dirs["md"] / f"{tid}.md",
                include_images=include_images,
                local_image_id=id_to_local.get(tid) if id_to_local else None,
                local_image_root=args.local_image_root,
            )
        if "json" in formats:
            render_json(
                traj,
                output_dirs["json"] / f"{tid}.json",
                local_image_id=id_to_local.get(tid) if id_to_local else None,
                local_image_root=args.local_image_root,
            )
    print(f"Done. Rendered {len(selected)} trajectories.")


if __name__ == "__main__":
    main()
