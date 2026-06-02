#!/usr/bin/env python3
"""Aggregate splitter attempts into difficulty labels (splitter v2: pass^k OpenCode).

Reads OpenCode pass^k results from `experiments/.meta/runs_splitter_v2/node_{A,B}/`
(8 canonical attempt dirs, each containing `results/<traj_id>.json` for all 1700
trajs), joins against `data/standard/agenthorizon_labels.jsonl` (post-hoc analysis
only; NOT used during evaluation per CLAUDE.md), and emits:

  - agenthorizon_difficulty.jsonl : per-trajectory annotations (1700 rows)
  - splitter_pairs.jsonl          : per 2-way pair (~850 rows)
  - agenthorizon_easy_ids.json    : trajectory ids classified Easy (JSON array)
  - agenthorizon_hard_ids.json    : trajectory ids classified Hard (JSON array)

OpenCode result schema: top-level `success` boolean from the splitter (Qwen3.5-122B-A10B
via vLLM + OpenCode harness). An attempt agrees with ground truth iff
`success=True` matches `label=positive` (or `success=False` matches `label=negative`).

Trajectory difficulty is per-trajectory: a trajectory is Easy iff its own
splitter_success_count >= EASY_THRESHOLD (currently 7 of 8). Methodology
matches Polaris-style continuous-then-binned difficulty (raw pass-rate
thresholded), not tau-bench's strict pass^k indicator at k=n. The 7-of-8
cutoff allows for 1 stochastic miss at temp=0.6 sampling without forfeiting
the Easy label, while keeping the bar high enough that the Easy bucket
genuinely captures trivially-handled trajectories.

Pairs are NOT enforced to share a bucket; the positive half of a pair can
land in Easy while its negative partner lands in Hard (and vice versa).

splitter_pairs.jsonl additionally reports a per-pair difficulty derived from
the AND rule applied to the per-trajectory verdict (pair Easy iff both items
land in Easy, i.e. both >= EASY_THRESHOLD/8) for downstream analysis, but
this is not propagated back to the trajectory rows.

Edge cases:
  - size-7 triangle component: instructions with multiple negatives can occur
    (specifically, 1 of the 3 instructions has 2 swap negatives). For those
    instructions, we currently skip emitting a 2-way pair (warning logged).
  - size-1 lone-positive: no instruction-paired negative exists → no 2-way pair.

Usage:
  uv run python scripts/aggregate_difficulty.py [--output-dir DIR]
  uv run python scripts/aggregate_difficulty.py --quiet  # tick-mode summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# v2: pass^k OpenCode splitter runs (Qwen3.5-122B-A10B via vLLM, OpenCode harness).
# Each exp_dir under node_{A,B}/ contributes one attempt per trajectory via its
# results/<traj_id>.json file. v1 (thin-client splitter) is archived under
# experiments/.archive/splitter_v1/ and is no longer aggregated here.
RUNS_BASE = REPO_ROOT / "experiments" / ".meta" / "runs_splitter_v2"
LABELS_PATH = REPO_ROOT / "data" / "standard" / "agenthorizon_labels.jsonl"
SPLITTER_MODEL = "Qwen/Qwen3.5-122B-A10B"
STRATIFICATION_RUN_ID = "splitter_v2_passk_opencode_2026-04-29"

K_EXPECTED = 8
# A trajectory is Easy iff its splitter_success_count is at least this many
# of K_EXPECTED. Set to 7 (allow 1 stochastic miss) rather than 8 (strict
# tau-bench pass^k) so val's Easy bucket is large enough (91 trajs) for
# stable bal-acc ranking; 8/8 left only 69 in val. Documented as Polaris-
# style continuous-then-binned difficulty rather than tau-bench pass^k strict.
EASY_THRESHOLD = 7


def load_labels() -> dict[str, dict]:
    """Load `agenthorizon_labels.jsonl` indexed by trajectory_id."""
    out: dict[str, dict] = {}
    with LABELS_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out[d["trajectory_id"]] = d
    return out


# Per-split scoring (val/public_test/private_test) was retired 2026-04-30
# in favour of operating over the full 1700-trajectory dataset only.
# The split JSONLs were moved to data/.archive/agenthorizon_splits/.
# `find_split_for` / `split_index` are gone; the `split` field in
# `agenthorizon_difficulty.jsonl` is no longer emitted.


def instruction_id_for_item(item: dict) -> str | None:
    """Per §6.4: positives use original_id; negatives use paired_id (the source
    of the instruction)."""
    if item["label"] == "positive":
        return item.get("original_id")
    elif item["label"] == "negative":
        return item.get("paired_id")
    return None


def pair_group_id_for_item(item: dict) -> str:
    """Canonical pair-group identifier: hash of {original_id, paired_id} (sorted),
    stable across both items of the same swap pair. For positives without a
    paired_id, use original_id alone (size-1 component)."""
    parts = []
    if item.get("original_id"):
        parts.append(item["original_id"])
    if item.get("paired_id"):
        parts.append(item["paired_id"])
    if not parts:
        return ""
    canonical = "::".join(sorted(set(parts)))
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def discover_attempt_dirs() -> list[Path]:
    """Find the K=8 canonical attempt dirs under runs_splitter_v2/node_{A,B}/.

    Each "attempt" is one OpenCode evaluate_trajectories run that processed all
    1700 trajectories. The 8 canonical attempts are the exp_dirs whose results/
    folder contains exactly 1700 result JSONs. OOM-cycle partials (smaller
    intermediate dirs that got --continue'd into the canonical dir) are skipped.
    """
    candidates: list[Path] = []
    for node in ("node_A", "node_B"):
        nd = RUNS_BASE / node
        if not nd.exists():
            continue
        for exp_dir in sorted(nd.iterdir()):
            if not exp_dir.is_dir():
                continue
            results = exp_dir / "results"
            if not results.exists():
                continue
            n = sum(1 for _ in results.glob("*.json"))
            if n == 1700:
                candidates.append(exp_dir)
    return candidates


def aggregate_attempts(attempt_dirs: list[Path]) -> tuple[dict[str, dict], list[str]]:
    """Walk K attempt dirs and compute per-trajectory success/error counts.

    Per-trajectory stats:
        success_count : int — attempts where splitter agreed with ground-truth label
        failure_count : int — attempts where splitter disagreed (or unusable verdict)
        error_count   : int — attempts that errored out (no result file at all,
                              or unreadable JSON)
        attempts_seen : int — number of distinct attempt dirs that yielded a
                              parseable result for this trajectory

    OpenCode result schema: top-level `success` boolean, plus `confidence`,
    `reasoning`, `mistake_type`, `_meta`. Errors do not produce a result file
    (they are logged to <exp_dir>/errors.jsonl); we count those as error_count
    by checking for missing result files in dirs that should have processed
    every traj.
    """
    labels = load_labels()
    warnings: list[str] = []

    # Collect all known trajectories from labels (ground-truth universe)
    all_trajs = set(labels.keys())

    stats: dict[str, dict] = {}
    for traj_id in all_trajs:
        stats[traj_id] = {
            "success_count": 0, "failure_count": 0, "error_count": 0, "attempts_seen": 0,
        }

    for ad in attempt_dirs:
        results_dir = ad / "results"
        for traj_id, label_record in labels.items():
            expected_positive = (label_record["label"] == "positive")
            rf = results_dir / f"{traj_id}.json"
            s = stats[traj_id]
            if not rf.exists():
                # Attempt dir was marked complete (1700 files) yet this traj
                # has no result — should not happen given our coverage check.
                s["error_count"] += 1
                continue
            try:
                rec = json.loads(rf.read_text())
            except json.JSONDecodeError:
                warnings.append(f"unreadable result file {rf}; counting as error")
                s["error_count"] += 1
                continue
            s["attempts_seen"] += 1
            sval = rec.get("success")
            if sval is True and expected_positive:
                s["success_count"] += 1
            elif sval is False and not expected_positive:
                s["success_count"] += 1
            else:
                s["failure_count"] += 1
    return stats, warnings


def derive_trajectory_difficulty(s: dict) -> str:
    """Easy iff success_count >= EASY_THRESHOLD (default 7 of 8)."""
    if s["error_count"] > 0:
        return "error"
    if s["attempts_seen"] < K_EXPECTED:
        return "incomplete"
    if s["success_count"] >= EASY_THRESHOLD:
        return "easy"
    return "hard"


def build_per_trajectory_rows(
    stats: dict[str, dict],
    labels: dict[str, dict],
) -> list[dict]:
    rows = []
    for traj_id, s in stats.items():
        lbl = labels[traj_id]
        rows.append({
            "trajectory_id": traj_id,
            "instruction_id": instruction_id_for_item(lbl),
            "pair_group_id": pair_group_id_for_item(lbl),
            "label": lbl["label"],
            "negative_source": lbl.get("negative_source"),
            "splitter_model": SPLITTER_MODEL,
            "k": K_EXPECTED,
            "splitter_success_count": s["success_count"],
            "splitter_error_count": s["error_count"],
            "splitter_attempts_seen": s["attempts_seen"],
            "trajectory_difficulty": derive_trajectory_difficulty(s),
            "stratification_run_id": STRATIFICATION_RUN_ID,
        })
    rows.sort(key=lambda r: (r["pair_group_id"], r["instruction_id"] or "", r["label"]))
    return rows


def build_pair_rows(per_traj_rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Group per-trajectory rows by instruction_id; emit one row per instruction."""
    warnings: list[str] = []
    by_instruction: dict[str, list[dict]] = defaultdict(list)
    for r in per_traj_rows:
        if r["instruction_id"] is None:
            continue  # size-1 lone positive: no pair
        by_instruction[r["instruction_id"]].append(r)

    pair_rows = []
    for instruction_id, items in by_instruction.items():
        positives = [r for r in items if r["label"] == "positive"]
        negatives = [r for r in items if r["label"] == "negative"]
        if len(positives) != 1 or len(negatives) != 1:
            warnings.append(
                f"instruction_id={instruction_id} has {len(positives)}p+{len(negatives)}n "
                f"(expected 1p+1n); skipping 2-way pair"
            )
            continue
        p = positives[0]
        n = negatives[0]
        any_error = (p["trajectory_difficulty"] == "error" or n["trajectory_difficulty"] == "error")
        any_incomplete = (p["trajectory_difficulty"] == "incomplete" or n["trajectory_difficulty"] == "incomplete")
        if any_error:
            pair_diff = "error"
        elif any_incomplete:
            pair_diff = "incomplete"
        elif p["trajectory_difficulty"] == "easy" and n["trajectory_difficulty"] == "easy":
            # Pair AND derived from per-traj verdict at EASY_THRESHOLD.
            # At 7/8 this yields 180 Easy pairs (360 trajs); informational
            # only, not propagated back to traj rows.
            pair_diff = "easy"
        else:
            pair_diff = "hard"
        pair_rows.append({
            "instruction_id": instruction_id,
            "pair_group_id": p["pair_group_id"],
            "positive_trajectory_id": p["trajectory_id"],
            "negative_trajectory_id": n["trajectory_id"],
            "positive_success_count": p["splitter_success_count"],
            "negative_success_count": n["splitter_success_count"],
            "negative_source": n.get("negative_source"),
            "pair_difficulty": pair_diff,
            "stratification_run_id": p["stratification_run_id"],
        })
    pair_rows.sort(key=lambda r: (r["pair_group_id"], r["instruction_id"]))
    return pair_rows, warnings


def write_outputs(out_dir: Path, traj_rows: list[dict], pair_rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "agenthorizon_difficulty.jsonl").write_text(
        "\n".join(json.dumps(r) for r in traj_rows) + "\n"
    )
    (out_dir / "splitter_pairs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in pair_rows) + "\n"
    )
    easy_ids = [r["trajectory_id"] for r in traj_rows if r["trajectory_difficulty"] == "easy"]
    hard_ids = [r["trajectory_id"] for r in traj_rows if r["trajectory_difficulty"] == "hard"]
    (out_dir / "agenthorizon_easy_ids.json").write_text(
        json.dumps(sorted(easy_ids), indent=2) + "\n"
    )
    (out_dir / "agenthorizon_hard_ids.json").write_text(
        json.dumps(sorted(hard_ids), indent=2) + "\n"
    )


def summarise(traj_rows: list[dict], pair_rows: list[dict]) -> dict:
    n_attempts = sum(r["splitter_attempts_seen"] for r in traj_rows)
    n_success = sum(r["splitter_success_count"] for r in traj_rows)
    n_error = sum(r["splitter_error_count"] for r in traj_rows)
    n_parsed = n_attempts - n_error
    parse_rate = (n_parsed / n_attempts * 100.0) if n_attempts else 0.0
    err_rate = (n_error / n_attempts * 100.0) if n_attempts else 0.0
    traj_diff = Counter(r["trajectory_difficulty"] for r in traj_rows)
    pair_diff = Counter(r["pair_difficulty"] for r in pair_rows)
    return {
        "n_trajectories": len(traj_rows),
        "n_pairs": len(pair_rows),
        "n_attempts": n_attempts,
        "parse_rate_pct": round(parse_rate, 2),
        "error_rate_pct": round(err_rate, 2),
        "n_success_total": n_success,
        "trajectory_difficulty_counts": dict(traj_diff),
        "pair_difficulty_counts": dict(pair_diff),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "data" / "standard",
                        help="Where to write the 4 output files. Default: data/standard/")
    parser.add_argument("--quiet", action="store_true", help="One-line summary; no file writes")
    args = parser.parse_args()

    attempt_dirs = discover_attempt_dirs()
    if len(attempt_dirs) != K_EXPECTED:
        sys.stderr.write(
            f"WARNING: discovered {len(attempt_dirs)} canonical attempt dirs, "
            f"expected {K_EXPECTED}. Listing:\n"
        )
        for ad in attempt_dirs:
            sys.stderr.write(f"  {ad}\n")

    stats, warnings = aggregate_attempts(attempt_dirs)
    labels = load_labels()
    traj_rows = build_per_trajectory_rows(stats, labels)
    pair_rows, pair_warnings = build_pair_rows(traj_rows)
    warnings.extend(pair_warnings)

    summary = summarise(traj_rows, pair_rows)
    summary["n_attempt_dirs"] = len(attempt_dirs)

    if args.quiet:
        # one-line summary (used by stratify_by_splitter.py for live ticks)
        td = summary["trajectory_difficulty_counts"]
        pd = summary["pair_difficulty_counts"]
        sys.stdout.write(
            f"[agg {STRATIFICATION_RUN_ID}] trajs={summary['n_trajectories']} "
            f"pairs={summary['n_pairs']} attempts={summary['n_attempts']} "
            f"parse={summary['parse_rate_pct']}% err={summary['error_rate_pct']}% "
            f"easy={td.get('easy',0)} hard={td.get('hard',0)} inc={td.get('incomplete',0)} | "
            f"pair_easy={pd.get('easy',0)} pair_hard={pd.get('hard',0)} pair_inc={pd.get('incomplete',0)}\n"
        )
        return 0

    write_outputs(args.output_dir, traj_rows, pair_rows)
    print(f"Wrote 4 files to {args.output_dir}", file=sys.stderr)
    print(f"\nSummary:\n  {json.dumps(summary, indent=2)}", file=sys.stderr)
    if warnings:
        print(f"\n{len(warnings)} warnings:", file=sys.stderr)
        for w in warnings[:20]:
            print(f"  - {w}", file=sys.stderr)
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
