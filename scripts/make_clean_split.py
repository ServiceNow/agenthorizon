#!/usr/bin/env python3
"""Create AgentHorizon `clean*` splits.

A `clean*` variant = AgentHorizon with a positive trajectory removed iff the
meta-judge classified that positive at or above the variant's severity
threshold. All original negatives are kept regardless of meta-judge severity.

Variants
--------
    clean     -> drop sev 1, 2, 3; keep sev 4 + null as positives.
                 sev-1/2 are confirmed task failures and the positive
                 label is wrong; sev-3 is a real-but-non-blocking issue
                 where neither label is cleanly correct. All three are
                 dropped. Original negatives are kept intact.

Severity scale (from `prompts/meta_judge/AGENTS.md.template`):
    1 = Critical (fail)              -> positive label is wrong, drop
    2 = High     (fail)              -> positive label is wrong, drop
    3 = Medium   (pass, real issue)  -> positive label is shaky, drop
    4 = Low      (pass, cosmetic)    -> dropped only in clean_v0
    null         (no issue)          -> keep

A positive trajectory is `(own_instruction, own_trajectory)` for one
deliverable; if the meta-judge says that trajectory does not correctly execute
its own instruction, the positive label is incorrect and the item should
leave the set. Negatives are left untouched: their label is a statement about
instruction-trajectory mismatch and is unaffected by the meta-judge's verdict
on the underlying recording.

Sources
-------
The script supports two meta-judge sources:

  - `meta_judge_exp` (default, current): a per-trajectory meta-judge run
    under `experiments/meta_judge/.../results/<trajectory_id>.json`. Pass
    `--meta-judge-exp <path>` to point at the run; the default is the
    GPT-5.4-judge / GPT-5.4 base / 2026-05-01 run.

  - `legacy_v2`: the original per-deliverable Re-QA final-batch meta-judge
    at `results/archive/meta_judge_reqa_final_v2/`. Files keyed by
    `deliverable_id`; trajectories are dropped if their `original_id` is
    in the dirty deliverable set.

Outputs (under `data/standard/`):
    agenthorizon_<variant>.jsonl         -- trajectories (no labels)
    agenthorizon_<variant>_labels.jsonl  -- labels for the same set,
                                            sorted by trajectory_id

Usage:
    uv run python scripts/make_clean_split.py                       # dry-run, variant=clean, default exp
    uv run python scripts/make_clean_split.py --write               # write clean
    uv run python scripts/make_clean_split.py --source legacy_v2 --write
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "standard"

LABELS_PATH = DATA / "agenthorizon_labels.jsonl"
TRAJS_PATH = DATA / "agenthorizon.jsonl"

# Default meta-judge exp dir (used when --source meta_judge_exp).
DEFAULT_META_JUDGE_EXP = (
    REPO / "experiments" / "meta_judge" / "gpt-5.4-judge" / "gpt-5.4"
    / "2026-05-01_2055_bc5547"
)
LEGACY_META_JUDGE_DIR = REPO / "results" / "archive" / "meta_judge_reqa_final_v2"

VARIANT_DIRTY_SEVERITIES: dict[str, set[int]] = {
    "clean":    {1, 2, 3},
}

# Variants in this set keep sev=1 trajectories in the dataset but flip
# them from positive to negative. Empty by default; can be re-enabled per
# variant if we want a "natural failure" augmentation pass.
VARIANT_RELABEL_SEV1_AS_NEGATIVE: set[str] = set()

# Normalise mistake_type from the base judge's vocabulary (singular
# "Instruction") to AgentHorizon's labels-file vocabulary (plural
# "Instructions"). Mirrors the normaliser in scripts/analyze_eval_results.py.
_MT_NORMALISE = {
    "Misunderstanding of the Instruction": "Misunderstanding of the Instructions",
}


def _normalise_mistake_type(mt: str | None) -> str | None:
    if not mt:
        return mt
    return _MT_NORMALISE.get(mt, mt)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def load_dirty_from_meta_judge_exp(
    exp_dir: Path,
    dirty_severities: set[int],
) -> tuple[set[str], dict]:
    """Return (set of dirty TRAJECTORY_IDs, severity histogram).

    Reads from a per-trajectory meta-judge run produced by
    `scripts/run_meta_judge_eval.py`. Each results/<tid>.json has fields
    {meta_verdict, severity, judge_verdict_correct, ...}.
    """
    dirty: set[str] = set()
    sev_hist: Counter = Counter()
    results_dir = exp_dir / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"meta-judge results dir not found: {results_dir}")
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith(".json"):
            continue
        try:
            d = json.loads((results_dir / f).read_text())
        except json.JSONDecodeError:
            continue
        sev = d.get("severity")
        sev_hist[sev] += 1
        if sev in dirty_severities:
            dirty.add(f.replace(".json", ""))
    return dirty, dict(sev_hist)


def load_dirty_from_legacy_v2(
    dirty_severities: set[int],
) -> tuple[set[str], dict]:
    """Return (set of dirty DELIVERABLE_IDs, severity histogram) from the
    legacy Re-QA final-batch v2 meta-judge run."""
    dirty: set[str] = set()
    sev_hist: Counter = Counter()
    for f in sorted(os.listdir(LEGACY_META_JUDGE_DIR)):
        if not f.endswith(".json"):
            continue
        d = json.loads((LEGACY_META_JUDGE_DIR / f).read_text())
        sev = d.get("severity")
        sev_hist[sev] += 1
        if sev in dirty_severities:
            dirty.add(d.get("deliverable_id") or f.replace(".json", ""))
    return dirty, dict(sev_hist)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variant", choices=sorted(VARIANT_DIRTY_SEVERITIES.keys()),
                    default="clean")
    ap.add_argument("--source", choices=["meta_judge_exp", "legacy_v2"],
                    default="meta_judge_exp",
                    help="Which meta-judge source to read severity from.")
    ap.add_argument("--meta-judge-exp", default=str(DEFAULT_META_JUDGE_EXP),
                    help="Path to the meta-judge experiment dir "
                         "(only used when --source meta_judge_exp).")
    ap.add_argument("--write", action="store_true",
                    help="Write the output files (default: dry-run).")
    args = ap.parse_args()

    dirty_severities = VARIANT_DIRTY_SEVERITIES[args.variant]
    out_trajs = DATA / f"agenthorizon_{args.variant}.jsonl"
    out_labels = DATA / f"agenthorizon_{args.variant}_labels.jsonl"

    labels = load_jsonl(LABELS_PATH)
    trajs = load_jsonl(TRAJS_PATH)
    print(f"Variant: {args.variant} -- dropping severities {sorted(dirty_severities)}")
    print(f"Source : {args.source}")
    if args.source == "meta_judge_exp":
        print(f"  meta-judge exp: {args.meta_judge_exp}")
    print(f"Loaded {len(labels)} labels, {len(trajs)} trajectories")

    # Optionally pull the *full* meta-judge results (for the relabel step we
    # need to know which trajectories are sev-1 specifically, and the base
    # judge's predicted mistake_type for them).
    meta_results: dict[str, dict] = {}
    base_results: dict[str, dict] = {}
    relabel_sev1 = (args.source == "meta_judge_exp"
                    and args.variant in VARIANT_RELABEL_SEV1_AS_NEGATIVE)
    if relabel_sev1:
        mj_exp = Path(args.meta_judge_exp).resolve()
        for f in (mj_exp / "results").iterdir():
            if not f.name.endswith(".json"):
                continue
            try:
                meta_results[f.stem] = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
        # Base judge results: same trajectory_ids, look in
        # <base_exp>/results/<tid>.json. The base exp dir is recorded in
        # the meta-judge config.
        try:
            mj_cfg = json.loads((mj_exp / "config.json").read_text())
            base_exp_dir = REPO / (mj_cfg.get("base_exp_dir") or "")
        except (json.JSONDecodeError, OSError):
            base_exp_dir = None
        if base_exp_dir and (base_exp_dir / "results").exists():
            for f in (base_exp_dir / "results").iterdir():
                if not f.name.endswith(".json"):
                    continue
                try:
                    base_results[f.stem] = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue

    if args.source == "meta_judge_exp":
        dirty, sev_hist = load_dirty_from_meta_judge_exp(
            Path(args.meta_judge_exp).resolve(), dirty_severities
        )
        print(f"Meta-judge severity histogram: {sev_hist}")
        print(f"Dirty trajectories (sev "
              f"{'/'.join(str(s) for s in sorted(dirty_severities))}): {len(dirty)}")
        all_tids = {l["trajectory_id"] for l in labels}
        dirty_in_ah = dirty & all_tids
        print(f"  of which present in AgentHorizon: {len(dirty_in_ah)}")

        # Decide: which of the dirty trajectories get RELABELED vs DROPPED.
        # Variant `clean`: sev=1 -> relabel as negative; sev=2/3 -> drop.
        # (relabel + drop are sev-conditional because the variant's "dirty"
        # set spans 1/2/3 but the relabel rule only fires on sev=1.)
        relabel_ids: set[str] = set()
        if relabel_sev1:
            for tid in dirty_in_ah:
                m = meta_results.get(tid) or {}
                if m.get("severity") == 1:
                    relabel_ids.add(tid)
        dropped_ids: set[str] = set()
        for d in labels:
            if d["label"] == "positive" and d["trajectory_id"] in dirty_in_ah \
                    and d["trajectory_id"] not in relabel_ids:
                dropped_ids.add(d["trajectory_id"])
        if relabel_sev1:
            print(f"  -> sev=1 to RELABEL as negative:    {len(relabel_ids)}")
            print(f"  -> remaining dirty (sev 2/3) DROP:  {len(dropped_ids)}")
    else:
        dirty, sev_hist = load_dirty_from_legacy_v2(dirty_severities)
        print(f"Meta-judge severity histogram: {sev_hist}")
        print(f"Dirty deliverables (sev "
              f"{'/'.join(str(s) for s in sorted(dirty_severities))}): {len(dirty)}")
        ah_dids: set[str] = set()
        for d in labels:
            if d.get("original_id"): ah_dids.add(d["original_id"])
            if d.get("paired_id"):   ah_dids.add(d["paired_id"])
        dirty_in_ah = dirty & ah_dids
        print(f"  of which present in AgentHorizon: {len(dirty_in_ah)}")
        relabel_ids = set()
        dropped_ids = set()
        for d in labels:
            if d["label"] == "positive" and d.get("original_id") in dirty_in_ah:
                dropped_ids.add(d["trajectory_id"])

    # Build the surviving label list, applying relabel transforms in place.
    surviving_labels = []
    for d in labels:
        if d["trajectory_id"] in dropped_ids:
            continue
        if d["trajectory_id"] in relabel_ids:
            base = base_results.get(d["trajectory_id"]) or {}
            base_mt = _normalise_mistake_type(base.get("mistake_type"))
            new_label = dict(d)
            new_label["label"] = "negative"
            new_label["mistake_type"] = base_mt
            sev = (meta_results.get(d["trajectory_id"]) or {}).get("severity")
            new_label["negative_source"] = f"meta_judge_natural_fail_sev{sev}"
            new_label["meta_judge_severity"] = sev
            # New negatives don't have a paired_id (they aren't constructed
            # from an instruction-trajectory swap).
            new_label.pop("paired_id", None)
            surviving_labels.append(new_label)
        else:
            surviving_labels.append(d)
    surviving_trajs = [t for t in trajs if t["trajectory_id"] not in dropped_ids]

    pos = sum(1 for d in surviving_labels if d["label"] == "positive")
    neg = sum(1 for d in surviving_labels if d["label"] == "negative")
    relabeled = sum(1 for d in surviving_labels
                    if (d.get("negative_source") or "").startswith("meta_judge_natural_fail"))
    pos_dropped = sum(1 for d in labels if d["label"] == "positive" and d["trajectory_id"] in dropped_ids)
    neg_dropped = sum(1 for d in labels if d["label"] == "negative" and d["trajectory_id"] in dropped_ids)

    print(f"\n{args.variant}:")
    print(f"  trajectories kept: {len(surviving_trajs)} / {len(trajs)} "
          f"({100*len(surviving_trajs)/len(trajs):.1f}%)")
    print(f"  positives:         {pos}  (dropped {pos_dropped} as bad-positive)")
    print(f"  negatives:         {neg}  (of which {relabeled} relabeled from positive)")
    mt = Counter(d.get("mistake_type", "") for d in surviving_labels if d["label"] == "negative")
    print(f"  mistake types (negatives only): {dict(mt)}")

    # Sanity: every dropped tid was a positive that was meta-judge-dirty.
    for tid in dropped_ids:
        rec = next(d for d in labels if d["trajectory_id"] == tid)
        assert rec["label"] == "positive", f"non-positive dropped: {tid}"
    # Sanity: every relabel was originally a positive.
    for tid in relabel_ids:
        rec = next(d for d in labels if d["trajectory_id"] == tid)
        assert rec["label"] == "positive", f"non-positive relabeled: {tid}"

    if not args.write:
        print("\nDry-run only. Pass --write to emit:")
        print(f"  {out_trajs}")
        print(f"  {out_labels}")
        return

    surviving_trajs.sort(key=lambda x: x["trajectory_id"])
    surviving_labels.sort(key=lambda x: x["trajectory_id"])

    with out_trajs.open("w") as f:
        for t in surviving_trajs:
            f.write(json.dumps(t) + "\n")
    with out_labels.open("w") as f:
        for d in surviving_labels:
            f.write(json.dumps(d) + "\n")

    print(f"\nWrote {len(surviving_trajs)} -> {out_trajs}")
    print(f"Wrote {len(surviving_labels)} -> {out_labels}")


if __name__ == "__main__":
    main()
