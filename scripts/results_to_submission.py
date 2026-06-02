#!/usr/bin/env python3
"""Convert per-trajectory judge result JSONs into the AgentHorizon submission
template format (one JSONL row per trajectory).

Input layout (what evaluate.py writes):
    <results-dir>/<trajectory_id>.json -> {"success":..., "reasoning":...,
        "confidence":..., "mistake_type":..., "_meta":{...}}

Output (default):
    results/analysis/submission_<run-name>.jsonl  -- one row per template
    trajectory_id, using data/standard/agenthorizon_full_submission_template
    .jsonl as the canonical row order. Trajectories the judge did not score
    are emitted with all-null fields so the file is always 1700 rows wide.
    `results/analysis/` is the only results/ subdir tracked by git, so this
    location lets the submission ride along with the existing analysis JSONs.

Usage:
    uv run python scripts/results_to_submission.py \\
        --results-dir results/eval_agenthorizon_llmjudge_compress_gemini_flash_lite \\
        --template data/standard/agenthorizon_full_submission_template.jsonl
"""

import argparse
import json
import sys
from pathlib import Path


def load_template(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_mistake_type(value) -> str | None:
    """Coerce common variants to the canonical singular form."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "null":
        return None
    # The plan's CLAUDE.md note: singular form, not plural
    if s == "Misunderstanding of the Instructions":
        return "Misunderstanding of the Instruction"
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True,
                        help="Directory of per-trajectory JSONs")
    parser.add_argument(
        "--template",
        default="data/standard/agenthorizon_full_submission_template.jsonl",
        help="Path to agenthorizon_full_submission_template.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL (default: results/analysis/submission_<run-name>.jsonl)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    template_path = Path(args.template)
    if args.output:
        output_path = Path(args.output)
    else:
        analysis_dir = Path("results/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_path = analysis_dir / f"submission_{results_dir.name}.jsonl"

    template_rows = load_template(template_path)
    print(f"Template: {template_path}  ({len(template_rows)} rows)")

    template_ids = {r["trajectory_id"] for r in template_rows}

    # Collect predictions
    predictions = {}
    parse_errors = 0
    skipped_unknown = 0
    for jpath in sorted(results_dir.glob("*.json")):
        if jpath.name.startswith("_"):  # _summary.json
            continue
        d = json.loads(jpath.read_text())
        tid = d.get("deliverable_id") or d.get("trajectory_id") or jpath.stem
        if tid not in template_ids:
            skipped_unknown += 1
            continue
        # If the judge response failed to parse JSON, success is missing
        if "success" not in d:
            parse_errors += 1
            continue
        predictions[tid] = {
            "trajectory_id": tid,
            "success": bool(d["success"]),
            "confidence": d.get("confidence"),
            "mistake_type": (
                None if d["success"]
                else normalize_mistake_type(d.get("mistake_type"))
            ),
            "reasoning": d.get("reasoning"),
        }

    print(f"Predictions found: {len(predictions)}/{len(template_rows)}")
    if parse_errors:
        print(f"  Skipped (parse error, no success field): {parse_errors}")
    if skipped_unknown:
        print(f"  Skipped (trajectory_id not in template): {skipped_unknown}")
    missing = len(template_rows) - len(predictions)
    if missing:
        print(f"  Missing (will write nulls): {missing}")

    # Write in template order, filling missing with nulls
    with open(output_path, "w") as f:
        for r in template_rows:
            tid = r["trajectory_id"]
            row = predictions.get(tid, {
                "trajectory_id": tid,
                "success": None,
                "confidence": None,
                "mistake_type": None,
                "reasoning": None,
            })
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    sys.exit(main())
