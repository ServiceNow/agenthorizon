#!/usr/bin/env python3
"""Pair-aware val / public-test / private-test split for AgentHorizon.

The dataset contains 1700 labelled items derived from 425 (parent, child)
trajectory pairs. Each pair yields 4 labelled items:
    positive(parent_instr, parent_traj)
    positive(child_instr,  child_traj)
    negative(parent_instr, child_traj)   -- negative_source=parent_instruction_child_trajectory
    negative(child_instr,  parent_traj)  -- negative_source=child_instruction_parent_trajectory

A leakage-free split MUST keep all 4 items from a pair in the same side,
because a judge that has been tuned on the positive counterpart has
implicitly seen the trajectory file of the paired negative (same on-disk
assets, just with a different instruction).

Targets (seed=42):
    val           :  200 items  (~50 pairs)
    public_test   : 1000 items  (~250 pairs, labels released)
    private_test  :  500 items  (~125 pairs, labels WITHHELD — leaderboard)

The real dataset has 2 edge groups (sizes 3 and 1) on top of 424 clean
size-4 pairs; both edges are routed to public_test so val and private_test
land exactly on target.

Outputs (written if --write is passed):
    data/.archive/agenthorizon_private_test_ids.txt          (ID list for leaderboard intake)
    data/standard/agenthorizon_full_submission_template.jsonl (1700 rows; one prediction
                                                              slot per trajectory, sorted
                                                              by trajectory_id)

The full labels file (`data/standard/agenthorizon_labels.jsonl`) is NOT
re-emitted by this script; it is treated as the immutable source of truth
for ground-truth labels. Per-split label files were consolidated into the
full file on 2026-04-30 and the now-redundant `agenthorizon_<split>_labels.jsonl`
copies live under `data/archive/standard/`.

Per-split DATA JSONLs (val/public_test/private_test) were retired 2026-04-30:
scoring is over the full 1700-trajectory dataset only. The previous split
JSONLs were moved to `data/.archive/agenthorizon_splits/` and the script
no longer emits them. Only the integrity-checking + private_test ID list
+ full submission template are produced now.

Without --write, the script only prints the split plan + integrity checks.

Usage:
    uv run python scripts/split_agenthorizon.py --dry-run
    uv run python scripts/split_agenthorizon.py --write
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "standard"

LABELS_PATH = DATA / "agenthorizon_labels.jsonl"
TRAJS_PATH = DATA / "agenthorizon.jsonl"

SEED = 42
VAL_PAIRS = 50
PRIVATE_TEST_PAIRS = 125


def group_items(items: list[dict]) -> dict[frozenset, list[dict]]:
    """Group 1700 items into pair-components via union-find on the pairing graph.

    Nodes = `original_id`. Edge (a, b) for every negative item whose
    `paired_id` = b and `original_id` = a. Connected components of this graph
    are the atomic units that MUST stay on the same split side.

    Expected component shapes on the current 1700:
        - 423 clean size-4 components (standard parent/child pair)
        - 1 size-7 "triangle" component where one trajectory is paired with two
          others, producing 3 positives + 4 negatives
        - 1 size-1 component (a lone positive with no negative counterpart)
    """
    nodes: set[str] = set()
    for d in items:
        if d.get("original_id"): nodes.add(d["original_id"])
        if d.get("paired_id"):   nodes.add(d["paired_id"])

    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for d in items:
        if d["label"] == "negative" and d.get("paired_id"):
            union(d["original_id"], d["paired_id"])

    def component_key(d: dict) -> frozenset:
        """Canonical key: the frozenset of all nodes in this component."""
        seed = d.get("original_id") or d.get("paired_id")
        if seed is None: return frozenset()
        root = find(seed)
        return frozenset(n for n in nodes if find(n) == root)

    merged: dict[frozenset, list[dict]] = defaultdict(list)
    for d in items:
        merged[component_key(d)].append(d)
    return merged


def stable_key(k: frozenset) -> str:
    """Deterministic sort key for a pair (sorted tuple of ids as a string)."""
    return "::".join(sorted(str(x) for x in k))


def assign_splits(groups: dict[frozenset, list[dict]], seed: int = SEED):
    """Return (val_keys, public_keys, private_keys).

    Clean size-4 components are the sampling pool. Non-4 edge components (the
    size-7 triangle and the solo positive in the current data) are routed to
    public_test so val and private_test land exactly on their item targets.
    """
    size4 = [k for k, vs in groups.items() if len(vs) == 4]
    edges = [k for k, vs in groups.items() if len(vs) != 4]
    size4.sort(key=stable_key)
    edges.sort(key=stable_key)

    rng = random.Random(seed)
    shuffled = list(size4)
    rng.shuffle(shuffled)
    val = shuffled[:VAL_PAIRS]
    private = shuffled[VAL_PAIRS : VAL_PAIRS + PRIVATE_TEST_PAIRS]
    public = shuffled[VAL_PAIRS + PRIVATE_TEST_PAIRS :] + edges
    return val, public, private


def fingerprint(ids: list[str]) -> str:
    """Stable hash of an ID list — used in the integrity block."""
    h = hashlib.sha256()
    for x in sorted(ids):
        h.update(x.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="Write split files. Default is dry-run.")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only (default).")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    labels = [json.loads(l) for l in LABELS_PATH.read_text().splitlines() if l.strip()]
    trajs = {}
    for l in TRAJS_PATH.read_text().splitlines():
        if l.strip():
            d = json.loads(l)
            trajs[d["trajectory_id"]] = d

    print(f"Loaded {len(labels)} labelled items, {len(trajs)} trajectories")

    groups = group_items(labels)
    sizes = Counter(len(v) for v in groups.values())
    print(f"Pair groups: {len(groups)}   size distribution: {dict(sizes)}")

    val_keys, public_keys, private_keys = assign_splits(groups, seed=args.seed)

    val_items = [d for k in val_keys for d in groups[k]]
    public_items = [d for k in public_keys for d in groups[k]]
    private_items = [d for k in private_keys for d in groups[k]]

    # Integrity checks
    all_tids = {d["trajectory_id"] for d in labels}
    split_tids = {d["trajectory_id"] for d in val_items + public_items + private_items}
    assert all_tids == split_tids, f"Coverage mismatch: {len(all_tids - split_tids)} missing, {len(split_tids - all_tids)} extra"

    # Pair disjointness
    def pair_ids(items):
        out = set()
        for d in items:
            if d.get("original_id"): out.add(d["original_id"])
            if d.get("paired_id"): out.add(d["paired_id"])
        return out

    v_ids, pub_ids, priv_ids = pair_ids(val_items), pair_ids(public_items), pair_ids(private_items)
    overlap_v_pub = v_ids & pub_ids
    overlap_v_priv = v_ids & priv_ids
    overlap_pub_priv = pub_ids & priv_ids
    print("\nPair disjointness (unique original / paired ids):")
    print(f"  val ∩ public  : {len(overlap_v_pub)}")
    print(f"  val ∩ private : {len(overlap_v_priv)}")
    print(f"  public ∩ private: {len(overlap_pub_priv)}")

    def summarise(name, items):
        c = Counter(d["label"] for d in items)
        print(f"  {name:<14} n={len(items):<5} pos={c.get('positive',0):<4} neg={c.get('negative',0):<4} "
              f"fingerprint={fingerprint([d['trajectory_id'] for d in items])}")

    print("\nSplits:")
    summarise("val", val_items)
    summarise("public_test", public_items)
    summarise("private_test", private_items)

    if not args.write:
        print("\n(dry-run — no files written; pass --write to persist)")
        return

    # Per-split data JSONLs (val/public_test/private_test) were retired
    # 2026-04-30. Scoring is now over the full 1700-trajectory dataset only.
    # The private_test ID list (still needed by the leaderboard intake server
    # to know which IDs to score) is written to data/.archive/ — it's a
    # release artifact, not a live corpus file. The full submission template
    # is the only data/standard/ output.
    archive_dir = REPO / "data" / ".archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting split metadata...")
    (archive_dir / "agenthorizon_private_test_ids.txt").write_text(
        "\n".join(sorted(d["trajectory_id"] for d in private_items)) + "\n"
    )
    print(f"  wrote private_test ID list to {archive_dir}/agenthorizon_private_test_ids.txt")

    # Single full submission template: one line per trajectory with placeholder
    # fields scorers fill in. Sorted by trajectory_id so diffing + intake
    # validation stays trivial. Replaces the old per-split templates (now
    # archived under data/archive/standard/) — one file covers val, public_test,
    # and private_test, and the scoring server filters by intended split.
    template_path = DATA / "agenthorizon_full_submission_template.jsonl"
    all_items = val_items + public_items + private_items
    with template_path.open("w") as f:
        for tid in sorted(d["trajectory_id"] for d in all_items):
            row = {
                "trajectory_id": tid,
                "success": None,             # bool, required
                "confidence": None,          # "low" | "medium" | "high", optional
                "mistake_type": None,        # null on success=true; else one of
                                             #   "Critical Mistake",
                                             #   "Bad Side Effect",
                                             #   "Misunderstanding of the Instruction"
                "reasoning": None,           # free text, optional
            }
            f.write(json.dumps(row) + "\n")
    print(f"  wrote full submission template ({len(all_items)} rows)")


if __name__ == "__main__":
    main()
