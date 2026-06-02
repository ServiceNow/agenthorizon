#!/usr/bin/env python3
"""Analyze evaluation results against ground-truth labels.

Loads per-trajectory judge results (JSON) and ground-truth labels (JSONL),
computes classification metrics (accuracy, precision, recall, F1), and
prints breakdowns by label, mistake type, and negative source.

Usage:
    uv run python scripts/analyze_eval_results.py \
        --results-dir results/eval_agenthorizon_opus \
        --labels data/standard/agenthorizon_labels.jsonl

    # When result UUIDs don't match label UUIDs (e.g., after JSONL regeneration),
    # provide the trajectories JSONL to enable content-based UUID remapping:
    uv run python scripts/analyze_eval_results.py \
        --results-dir results/eval_agenthorizon_opus \
        --labels data/standard/agenthorizon_labels.jsonl \
        --trajectories data/standard/agenthorizon.jsonl

    # Save JSON report
    uv run python scripts/analyze_eval_results.py \
        --results-dir results/eval_agenthorizon_opus \
        --labels data/standard/agenthorizon_labels.jsonl \
        --trajectories data/standard/agenthorizon.jsonl \
        --output results/eval_agenthorizon_opus/analysis.json

    # Save Markdown report
    uv run python scripts/analyze_eval_results.py \
        --results-dir results/eval_agenthorizon_opus \
        --labels data/standard/agenthorizon_labels.jsonl \
        --trajectories data/standard/agenthorizon.jsonl \
        --markdown results/eval_agenthorizon_opus/analysis.md
"""

import argparse
import json
import logging
import os
import re
import uuid as _uuid_mod
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def load_results(results_dir: Path) -> tuple[dict[str, bool], int, list[str]]:
    """Load judge results from per-trajectory JSON files.

    Returns:
        predictions: dict mapping trajectory_id -> predicted positive (True/False)
        parse_errors: count of files where 'success' field was missing/unparseable
        parse_error_ids: list of trajectory_ids that failed to parse
    """
    predictions: dict[str, bool] = {}
    parse_errors = 0
    parse_error_ids: list[str] = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue

        trajectory_id = fname[: -len(".json")]

        # Skip non-UUID filenames (e.g., analysis.json)
        try:
            _uuid_mod.UUID(trajectory_id)
        except ValueError:
            continue
        fpath = results_dir / fname

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("  Parse error: %s (%s)", fname, exc)
            parse_errors += 1
            parse_error_ids.append(trajectory_id)
            continue

        if "success" not in data:
            log.warning("  Missing 'success' field: %s", fname)
            parse_errors += 1
            parse_error_ids.append(trajectory_id)
            continue

        predictions[trajectory_id] = bool(data["success"])

    return predictions, parse_errors, parse_error_ids


def load_labels(labels_path: Path) -> dict[str, dict]:
    """Load ground-truth labels from JSONL, keyed by trajectory_id."""
    labels: dict[str, dict] = {}

    with open(labels_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("  Labels line %d: parse error (%s)", line_num, exc)
                continue

            tid = record.get("trajectory_id")
            if tid:
                labels[tid] = record

    return labels


def build_uuid_remap(
    result_ids: set[str],
    labels: dict[str, dict],
    trajectories_path: Path,
    md_dir: Path,
) -> dict[str, str]:
    """Build a mapping from result UUIDs (old) to label UUIDs (new).

    When the standardized JSONL was regenerated (e.g., format version update),
    new random UUIDs were assigned. The markdown files (and thus result files)
    still use the old UUIDs. This function bridges the gap by matching content
    fingerprints (instruction text + demo deliverable ID from screenshot paths).

    Returns:
        remap: dict mapping old_uuid (result filename) -> new_uuid (label trajectory_id)
    """
    # Build JSONL fingerprints: (instruction, demo_deliverable_id) -> new_uuid
    jsonl_fp: dict[str, str] = {}
    with open(trajectories_path) as f:
        for line in f:
            rec = json.loads(line)
            tid = rec["trajectory_id"]
            inst = rec["task"]["instruction"]
            steps = rec.get("steps", [])
            first_screenshot = steps[0].get("screenshot", "") if steps else ""
            m = re.search(r"images/([\w-]+)/step_", first_screenshot)
            demo_id = m.group(1) if m else ""
            key = f"{inst}::{demo_id}"
            jsonl_fp[key] = tid

    # Match each markdown file's content to a JSONL entry
    remap: dict[str, str] = {}
    for fname in os.listdir(md_dir):
        if not fname.endswith(".md"):
            continue
        old_uuid = fname[:-3]
        if old_uuid not in result_ids:
            continue

        with open(md_dir / fname) as f:
            content = f.read()

        goal_match = re.search(r"## Goal\n\n(.*?)\n\n## ", content, re.DOTALL)
        goal = goal_match.group(1) if goal_match else ""
        screenshots = re.findall(r"images/([\w-]+)/step_\d+", content)
        demo_id = screenshots[0] if screenshots else ""
        key = f"{goal}::{demo_id}"

        new_uuid = jsonl_fp.get(key)
        if new_uuid and new_uuid in labels:
            remap[old_uuid] = new_uuid

    return remap


def compute_metrics(
    predictions: dict[str, bool],
    labels: dict[str, dict],
    parse_error_ids: list[str] | None = None,
) -> dict:
    """Compute classification metrics and breakdowns.

    Mapping:
        success=True  -> predicted positive  (task completed correctly)
        success=False -> predicted negative   (task NOT completed correctly)
        label="positive" -> actual positive
        label="negative" -> actual negative

    Parse errors are counted as incorrect predictions (wrong for whichever
    class the trajectory actually belongs to). This penalizes judges that
    fail to produce valid JSON.
    """
    tp = fp = tn = fn = 0
    unmatched = []

    # Per-label accuracy
    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0

    # Negative breakdowns: key -> (correct, total)
    by_mistake_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_negative_source: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for tid, predicted_positive in predictions.items():
        label_record = labels.get(tid)
        if label_record is None:
            unmatched.append(tid)
            continue

        actual_positive = label_record["label"] == "positive"

        if actual_positive:
            positive_total += 1
            if predicted_positive:
                tp += 1
                positive_correct += 1
            else:
                fn += 1
        else:
            negative_total += 1
            if predicted_positive:
                fp += 1
            else:
                tn += 1
                negative_correct += 1

            # Breakdown tracking for negatives. Normalise the label's
            # mistake_type so plural ("Misunderstanding of the Instructions")
            # and singular forms collapse into one bucket. Canonical form is
            # singular, matching the judge schema.
            raw_mt = label_record.get("mistake_type") or "Unknown"
            if raw_mt == "Misunderstanding of the Instructions":
                raw_mt = "Misunderstanding of the Instruction"
            mistake_type = raw_mt
            neg_source = label_record.get("negative_source", "Unknown")

            by_mistake_type[mistake_type][1] += 1
            by_negative_source[neg_source][1] += 1

            if not predicted_positive:  # correctly identified as negative
                by_mistake_type[mistake_type][0] += 1
                by_negative_source[neg_source][0] += 1

    # Count parse errors (excluded from accuracy, reported separately)
    parse_error_count = 0
    if parse_error_ids:
        for tid in parse_error_ids:
            label_record = labels.get(tid)
            if label_record is None:
                continue
            parse_error_count += 1

    # Derived metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_correct": positive_correct,
        "positive_total": positive_total,
        "negative_correct": negative_correct,
        "negative_total": negative_total,
        "parse_errors_penalized": parse_error_count,
        "by_mistake_type": dict(by_mistake_type),
        "by_negative_source": dict(by_negative_source),
        "unmatched": unmatched,
    }


def print_report(
    metrics: dict,
    total_results: int,
    parse_errors: int,
) -> None:
    """Print a formatted analysis report to the terminal."""
    m = metrics
    matched = m["total"]
    unmatched = len(m["unmatched"])

    print()
    print("=== Evaluation Analysis ===")
    print(
        f"Results: {total_results}  |  Matched: {matched}"
        f"  |  Parse errors: {parse_errors}  |  Unmatched: {unmatched}"
    )

    print()
    print("=== Overall Metrics ===")
    print(
        f"Accuracy: {m['accuracy']:6.1%}   "
        f"Precision: {m['precision']:6.1%}   "
        f"Recall: {m['recall']:6.1%}   "
        f"F1: {m['f1']:6.1%}"
    )

    print()
    print("=== Confusion Matrix ===")
    print(f"{'':20s} Predicted Pos   Predicted Neg")
    print(f"{'Actual Positive':20s} {m['tp']:>11d}   {m['fn']:>13d}")
    print(f"{'Actual Negative':20s} {m['fp']:>11d}   {m['tn']:>13d}")

    print()
    print("=== Accuracy by Label ===")
    if m["positive_total"]:
        pct = m["positive_correct"] / m["positive_total"]
        print(f"Positive: {pct:5.1%}  ({m['positive_correct']}/{m['positive_total']})")
    else:
        print("Positive: N/A (no positive labels)")
    if m["negative_total"]:
        pct = m["negative_correct"] / m["negative_total"]
        print(f"Negative: {pct:5.1%}  ({m['negative_correct']}/{m['negative_total']})")
    else:
        print("Negative: N/A (no negative labels)")

    if m["by_mistake_type"]:
        print()
        print("=== Negative Breakdown by Mistake Type ===")
        for mtype in sorted(m["by_mistake_type"]):
            correct, total = m["by_mistake_type"][mtype]
            pct = correct / total if total else 0.0
            print(f"  {mtype:40s} {pct:5.1%}  ({correct}/{total})")

    if m["by_negative_source"]:
        print()
        print("=== Negative Breakdown by Source ===")
        for source in sorted(m["by_negative_source"]):
            correct, total = m["by_negative_source"][source]
            pct = correct / total if total else 0.0
            print(f"  {source:45s} {pct:5.1%}  ({correct}/{total})")

    if m["unmatched"]:
        print()
        print(f"=== Unmatched Results ({unmatched}) ===")
        for tid in m["unmatched"][:10]:
            print(f"  {tid}")
        if unmatched > 10:
            print(f"  ... and {unmatched - 10} more")

    print()


def build_json_report(
    metrics: dict,
    total_results: int,
    parse_errors: int,
) -> dict:
    """Build a JSON-serializable report dict."""
    m = metrics
    return {
        "summary": {
            "total_results": total_results,
            "matched": m["total"],
            "parse_errors": parse_errors,
            "unmatched": len(m["unmatched"]),
        },
        "metrics": {
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
        },
        "confusion_matrix": {
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
        },
        "accuracy_by_label": {
            "positive": {
                "correct": m["positive_correct"],
                "total": m["positive_total"],
                "accuracy": round(
                    m["positive_correct"] / m["positive_total"], 4
                )
                if m["positive_total"]
                else None,
            },
            "negative": {
                "correct": m["negative_correct"],
                "total": m["negative_total"],
                "accuracy": round(
                    m["negative_correct"] / m["negative_total"], 4
                )
                if m["negative_total"]
                else None,
            },
        },
        "negative_by_mistake_type": {
            mtype: {
                "correct": counts[0],
                "total": counts[1],
                "accuracy": round(counts[0] / counts[1], 4) if counts[1] else None,
            }
            for mtype, counts in sorted(m["by_mistake_type"].items())
        },
        "negative_by_source": {
            source: {
                "correct": counts[0],
                "total": counts[1],
                "accuracy": round(counts[0] / counts[1], 4) if counts[1] else None,
            }
            for source, counts in sorted(m["by_negative_source"].items())
        },
    }


def build_markdown_report(
    metrics: dict,
    total_results: int,
    parse_errors: int,
    results_dir: str = "",
    labels_path: str = "",
) -> str:
    """Build a Markdown-formatted analysis report."""
    m = metrics
    matched = m["total"]
    unmatched = len(m["unmatched"])

    lines = []
    lines.append("# Evaluation Analysis Report")
    lines.append("")
    if results_dir:
        lines.append(f"**Results directory:** `{results_dir}`")
    if labels_path:
        lines.append(f"**Labels file:** `{labels_path}`")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total results | {total_results} |")
    lines.append(f"| Matched | {matched} |")
    lines.append(f"| Parse errors | {parse_errors} |")
    lines.append(f"| Unmatched | {unmatched} |")
    lines.append("")

    # Overall metrics
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Accuracy | {m['accuracy']:.1%} |")
    lines.append(f"| Precision | {m['precision']:.1%} |")
    lines.append(f"| Recall | {m['recall']:.1%} |")
    lines.append(f"| F1 | {m['f1']:.1%} |")
    lines.append("")

    # Confusion matrix
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append(f"| | Predicted Positive | Predicted Negative |")
    lines.append(f"|---|---|---|")
    lines.append(f"| **Actual Positive** | {m['tp']} (TP) | {m['fn']} (FN) |")
    lines.append(f"| **Actual Negative** | {m['fp']} (FP) | {m['tn']} (TN) |")
    lines.append("")

    # Accuracy by label
    lines.append("## Accuracy by Label")
    lines.append("")
    lines.append(f"| Label | Accuracy | Correct | Total |")
    lines.append(f"|-------|----------|---------|-------|")
    if m["positive_total"]:
        pct = m["positive_correct"] / m["positive_total"]
        lines.append(
            f"| Positive | {pct:.1%} | {m['positive_correct']} | {m['positive_total']} |"
        )
    if m["negative_total"]:
        pct = m["negative_correct"] / m["negative_total"]
        lines.append(
            f"| Negative | {pct:.1%} | {m['negative_correct']} | {m['negative_total']} |"
        )
    lines.append("")

    # Negative breakdown by mistake type
    if m["by_mistake_type"]:
        lines.append("## Negative Breakdown by Mistake Type")
        lines.append("")
        lines.append(f"| Mistake Type | Accuracy | Correct | Total |")
        lines.append(f"|-------------|----------|---------|-------|")
        for mtype in sorted(m["by_mistake_type"]):
            correct, total = m["by_mistake_type"][mtype]
            pct = correct / total if total else 0.0
            lines.append(f"| {mtype} | {pct:.1%} | {correct} | {total} |")
        lines.append("")

    # Negative breakdown by source
    if m["by_negative_source"]:
        lines.append("## Negative Breakdown by Source")
        lines.append("")
        lines.append(f"| Source | Accuracy | Correct | Total |")
        lines.append(f"|--------|----------|---------|-------|")
        for source in sorted(m["by_negative_source"]):
            correct, total = m["by_negative_source"][source]
            pct = correct / total if total else 0.0
            label = source.replace("_", " ").title()
            lines.append(f"| {label} | {pct:.1%} | {correct} | {total} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze evaluation results against ground-truth labels."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing per-trajectory result JSON files",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to ground-truth labels JSONL file",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=None,
        help="Path to trajectories JSONL file (for UUID remapping when labels "
        "were regenerated with new UUIDs after evaluation)",
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=None,
        help="Path to markdown directory used for evaluation. Auto-detected "
        "from _summary.json in results-dir if not specified.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON report (optional)",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Path to write Markdown report (optional)",
    )
    args = parser.parse_args()

    if not args.results_dir.is_dir():
        parser.error(f"Results directory not found: {args.results_dir}")
    if not args.labels.is_file():
        parser.error(f"Labels file not found: {args.labels}")

    log.info("Loading results from %s ...", args.results_dir)
    predictions, parse_errors, parse_error_ids = load_results(args.results_dir)
    total_results = len(predictions) + parse_errors
    log.info("  Loaded %d predictions (%d parse errors)", len(predictions), parse_errors)

    log.info("Loading labels from %s ...", args.labels)
    labels = load_labels(args.labels)
    log.info("  Loaded %d labels", len(labels))

    # Check if direct matching works
    direct_matches = sum(1 for tid in predictions if tid in labels)
    need_remap = direct_matches < len(predictions) * 0.5

    if need_remap:
        if args.trajectories is None:
            log.warning(
                "  Only %d/%d results match labels directly. The JSONL may have "
                "been regenerated with new UUIDs. Use --trajectories to enable "
                "content-based UUID remapping.",
                direct_matches,
                len(predictions),
            )
        else:
            # Auto-detect md-dir from _summary.json if not specified
            md_dir = args.md_dir
            if md_dir is None:
                summary_path = args.results_dir / "_summary.json"
                if summary_path.is_file():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    md_dir = Path(summary.get("md_dir", ""))
                    if not md_dir.is_dir():
                        parser.error(
                            f"Auto-detected md_dir '{md_dir}' from _summary.json "
                            f"is not a valid directory. Use --md-dir to specify."
                        )
                    log.info("  Auto-detected md-dir: %s", md_dir)
                else:
                    parser.error(
                        "Cannot auto-detect md-dir (_summary.json not found). "
                        "Use --md-dir to specify the markdown directory."
                    )

            if not args.trajectories.is_file():
                parser.error(f"Trajectories file not found: {args.trajectories}")

            log.info("Building UUID remap from %s + %s ...", args.trajectories, md_dir)
            remap = build_uuid_remap(
                set(predictions.keys()), labels, args.trajectories, md_dir
            )
            log.info("  Remapped %d/%d result UUIDs", len(remap), len(predictions))

            # Apply remap: translate prediction keys from old UUIDs to new UUIDs
            predictions = {
                remap.get(tid, tid): pred for tid, pred in predictions.items()
            }
            parse_error_ids = [remap.get(tid, tid) for tid in parse_error_ids]

    metrics = compute_metrics(predictions, labels, parse_error_ids=parse_error_ids)
    print_report(metrics, total_results, parse_errors)

    if args.output:
        report = build_json_report(metrics, total_results, parse_errors)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("JSON report written to %s", args.output)

    if args.markdown:
        md_report = build_markdown_report(
            metrics,
            total_results,
            parse_errors,
            results_dir=str(args.results_dir),
            labels_path=str(args.labels),
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write(md_report)
        log.info("Markdown report written to %s", args.markdown)


if __name__ == "__main__":
    main()
