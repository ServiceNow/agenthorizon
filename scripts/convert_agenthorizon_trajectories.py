"""
Convert human-annotated trajectories (delivery_*.json) to the standard trajectory format.

Outputs two files:
1. A JSONL file with trajectories (UUIDs, shuffled, NO labels) for blind evaluation.
2. A labels JSONL file mapping trajectory UUIDs to ground-truth labels and metadata.

Each deliverable produces a positive trajectory (own instruction + own execution).
Paired deliverables also produce two negative trajectories by cross-assigning
instructions and demonstrations.

Usage:
    uv run python scripts/convert_agenthorizon_trajectories.py \
        --input data/new/final/final_delivery_batch.json \
        --output data/standard/agenthorizon.jsonl \
        --labels-output data/standard/agenthorizon_labels.jsonl \
        --data-dir data/media
"""

import argparse
import json
import logging
import os
import random
import uuid
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OS_NORMALIZE = {
    "windows": "windows",
    "linux": "linux",
    "macos": "macos",
    "ubuntu": "linux",
    "mac": "macos",
}


def normalize_os(os_str: str) -> str | None:
    """Normalize OS string to standard format. Returns None if invalid."""
    key = os_str.strip().lower()
    return OS_NORMALIZE.get(key)


def normalize_trajectory_type(ttype: str) -> str:
    """Normalize trajectory type strings to a consistent format."""
    return ttype.strip().lower().replace(" ", "")


def convert_action(tool_call: dict) -> dict:
    """Convert a human trajectory tool_call to the standard action format."""
    action_type = tool_call["type"]
    args = tool_call["function"]["arguments"]

    if action_type in ("click", "doubleClick"):
        button_raw = args.get("text", "left-click")
        button = "right" if "right" in button_raw else "left"
        num_clicks = args.get("numClicks", 2 if action_type == "doubleClick" else 1)
        return {
            "type": "click",
            "parameters": {
                "x": args["x"],
                "y": args["y"],
                "button": button,
                "num_clicks": num_clicks,
            },
        }

    if action_type == "typing":
        return {
            "type": "type",
            "parameters": {"text": args["text"]},
        }

    if action_type == "press":
        return {
            "type": "press",
            "parameters": {"keys": [args["text"]]},
        }

    if action_type == "hotkey":
        keys = [k.strip() for k in args["text"].split("+")]
        return {
            "type": "hotkey",
            "parameters": {"keys": keys},
        }

    if action_type == "scroll":
        return {
            "type": "scroll",
            "parameters": {
                "x": args.get("scrollX", 0),
                "y": args.get("scrollY", 0),
                "direction": args.get("scrollDirection", "down"),
                "amount": args.get("totalScrollDistance", 0),
            },
        }

    if action_type == "dragFromTo":
        return {
            "type": "drag",
            "parameters": {
                "start_x": args["x"],
                "start_y": args["y"],
                "end_x": args["xEnd"],
                "end_y": args["yEnd"],
            },
        }

    if action_type == "drag":
        return {
            "type": "drag",
            "parameters": {
                "start_x": 0,
                "start_y": 0,
                "end_x": args.get("x", 0),
                "end_y": args.get("y", 0),
            },
        }

    if action_type == "keyDown":
        return {
            "type": "key_down",
            "parameters": {"key": args["text"]},
        }

    if action_type == "keyUp":
        return {
            "type": "key_up",
            "parameters": {"key": args["text"]},
        }

    logger.warning("Unknown action type: %s", action_type)
    return {
        "type": action_type,
        "parameters": args,
    }


def extract_steps(deliverable: dict, data_dir: str | None = None) -> list[dict]:
    """Extract steps from a deliverable's assistant reasoning.

    Args:
        deliverable: The deliverable dict from the delivery JSON.
        data_dir: If provided, screenshot paths are local relative paths
            matching the layout of download_delivery.py:
            <data_dir>/images/<deliverable_id>/step_<step_id>.<ext>
            If None, stores the original GCS URL.
    """
    did = deliverable["deliverable_id"]
    tool_calls = []
    for msg in deliverable["messages"]:
        if msg["role"] != "assistant":
            continue
        reasoning = msg.get("reasoning", {})
        for process_item in reasoning.get("process", []):
            for event in process_item.get("events", []):
                tool_calls.extend(event.get("tool_calls", []))

    steps = []
    for i, tc in enumerate(tool_calls):
        screenshot = None
        if tc.get("image") and tc["image"].get("url"):
            if data_dir:
                # Local path matching download_delivery.py naming convention
                img_url = tc["image"]["url"]
                ext = Path(urlparse(img_url).path).suffix or ".png"
                step_id = tc.get("step_id", "unknown")
                screenshot = os.path.join(data_dir, "images", did, f"step_{step_id}{ext}")
            else:
                screenshot = tc["image"]["url"]

        timestamp_us = None
        if tc.get("timestamp_microsecs"):
            try:
                timestamp_us = int(tc["timestamp_microsecs"])
            except (ValueError, TypeError):
                pass

        steps.append({
            "step_id": i,
            "screenshot": screenshot,
            "action": convert_action(tc),
            "thought": None,
            "action_description": None,
            "timestamp_us": timestamp_us,
        })

    return steps


def get_instruction(deliverable: dict) -> str:
    """Extract the task instruction from the user message."""
    for msg in deliverable["messages"]:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def build_trajectory(
    deliverable: dict,
    source_file: str,
    trajectory_id: str,
    instruction: str,
    steps: list[dict],
) -> dict:
    """Build a standard trajectory dict (no label — labels go to separate file)."""
    notes = deliverable.get("notes", {})

    # Normalize OS
    raw_os = notes.get("os", "")
    env_os = normalize_os(raw_os)

    # Extract category/subcategory
    categories = notes.get("task_category_list", [])
    category = categories[0].get("category") if categories else None
    subcategory = categories[0].get("subcategory") if categories else None

    # Build milestones
    milestones = []
    for m in notes.get("milestones", []):
        milestones.append({
            "step_id": m["step_id"],
            "description": m["task_subgoal"],
        })

    task = {
        "instruction": instruction,
        "category": category,
        "subcategory": subcategory,
        "task_type": notes.get("task_type"),
        "applications": notes.get("application_names", []),
    }

    return {
        "version": "1.1",
        "trajectory_id": trajectory_id,
        "source": {
            "name": "human",
            "original_file": source_file,
        },
        "task": task,
        "environment": {
            "os": env_os,
            "screen_resolution": None,
        },
        "steps": steps,
        "milestones": milestones if milestones else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert human trajectories to standard format"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the delivery JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path (trajectories, no labels)",
    )
    parser.add_argument(
        "--labels-output",
        required=True,
        help="Output JSONL file path (ground-truth labels)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to downloaded media directory (for local screenshot paths). "
        "If omitted, stores original GCS URLs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_file = os.path.relpath(args.input, repo_root)

    logger.info("Loading %s", args.input)
    with open(args.input) as f:
        deliverables = json.load(f)
    logger.info("Loaded %d deliverables", len(deliverables))

    # Index by deliverable_id for parent-child lookups
    by_id = {d["deliverable_id"]: d for d in deliverables}

    trajectories = []  # (trajectory_dict, label_dict) pairs
    positive_count = 0
    negative_count = 0
    skipped = 0
    # Deliverables whose positives must be dropped at the end because their
    # cross-instruction negatives were skipped (identical-instruction case).
    # Both sides of such a pair get dropped so the output contains no
    # orphan positives (pair-group size-1 components).
    drop_positives_for: set[str] = set()

    for deliverable in deliverables:
        did = deliverable["deliverable_id"]

        try:
            instruction = get_instruction(deliverable)
            steps = extract_steps(deliverable, data_dir=args.data_dir)

            # Positive trajectory: own instruction + own execution
            pos_uuid = str(uuid.uuid4())
            pos_traj = build_trajectory(
                deliverable=deliverable,
                source_file=source_file,
                trajectory_id=pos_uuid,
                instruction=instruction,
                steps=steps,
            )
            pos_label = {
                "trajectory_id": pos_uuid,
                "label": "positive",
                "original_id": did,
                "trajectory_scenario": deliverable.get("notes", {}).get("trajectory_scenario"),
                "trajectory_type": deliverable.get("notes", {}).get("trajectory_type"),
            }
            trajectories.append((pos_traj, pos_label))
            positive_count += 1

            # If this deliverable has a paired_task_id, create two negatives
            # by cross-assigning instructions and demonstrations.
            #
            # Terminology (matching the delivery JSON fields):
            #   "inst1" = this deliverable's instruction
            #   "inst2" = paired (parent) deliverable's instruction
            #   "demo_1" = this deliverable's demonstration (steps)
            #   "demo_2" = paired (parent) deliverable's demonstration (steps)
            #
            # Negative 1: inst2 (parent instruction) + demo_1 (child steps)
            #   → mistake_type_inst2_demo_1
            # Negative 2: inst1 (child instruction) + demo_2 (parent steps)
            #   → mistake_type_inst1_demo_2
            paired_id = deliverable["notes"].get("paired_task_id")
            if paired_id and paired_id in by_id:
                parent = by_id[paired_id]
                parent_instruction = get_instruction(parent)
                parent_steps = extract_steps(parent, data_dir=args.data_dir)
                notes = deliverable.get("notes", {})

                # Skip pairs where parent and child have identical instructions.
                # These produce negatives indistinguishable from positives.
                # Known case: 000233394/000232629 (identical after Re-QA rework).
                # We also drop BOTH positives from the final output so the
                # dataset contains no size-1 orphan components.
                if instruction == parent_instruction:
                    logger.warning(
                        "Skipping negative pair %s / %s: identical instructions "
                        "(dropping both positives too)",
                        did, paired_id,
                    )
                    drop_positives_for.add(did)
                    drop_positives_for.add(paired_id)
                    continue

                # Negative 1: parent instruction + child demo
                neg1_uuid = str(uuid.uuid4())
                neg1_traj = build_trajectory(
                    deliverable=deliverable,
                    source_file=source_file,
                    trajectory_id=neg1_uuid,
                    instruction=parent_instruction,
                    steps=steps,
                )
                neg1_label = {
                    "trajectory_id": neg1_uuid,
                    "label": "negative",
                    "original_id": did,
                    "paired_id": paired_id,
                    "negative_source": "parent_instruction_child_trajectory",
                    "mistake_type": notes.get("mistake_type_inst2_demo_1"),
                }
                trajectories.append((neg1_traj, neg1_label))
                negative_count += 1

                # Negative 2: child instruction + parent demo
                neg2_uuid = str(uuid.uuid4())
                neg2_traj = build_trajectory(
                    deliverable=parent,
                    source_file=source_file,
                    trajectory_id=neg2_uuid,
                    instruction=instruction,
                    steps=parent_steps,
                )
                neg2_label = {
                    "trajectory_id": neg2_uuid,
                    "label": "negative",
                    "original_id": paired_id,
                    "paired_id": did,
                    "negative_source": "child_instruction_parent_trajectory",
                    "mistake_type": notes.get("mistake_type_inst1_demo_2"),
                }
                trajectories.append((neg2_traj, neg2_label))
                negative_count += 1

        except Exception:
            logger.exception("Failed to convert deliverable %s", did)
            skipped += 1

    # Drop positives flagged by the identical-instruction skip path.
    if drop_positives_for:
        before = len(trajectories)
        trajectories = [
            (traj, lbl) for (traj, lbl) in trajectories
            if not (lbl["label"] == "positive" and lbl["original_id"] in drop_positives_for)
        ]
        dropped = before - len(trajectories)
        positive_count -= dropped
        logger.warning(
            "Dropped %d orphan positives for identical-instruction deliverables: %s",
            dropped, sorted(drop_positives_for),
        )

    # Shuffle trajectories so positive/negative are intermixed
    rng = random.Random(args.seed)
    rng.shuffle(trajectories)

    # Write outputs
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.labels_output)), exist_ok=True)

    with open(args.output, "w") as traj_f, open(args.labels_output, "w") as label_f:
        for traj, label in trajectories:
            traj_f.write(json.dumps(traj) + "\n")
            label_f.write(json.dumps(label) + "\n")

    logger.info(
        "Done: %d positive, %d negative, %d skipped → %d total trajectories",
        positive_count,
        negative_count,
        skipped,
        positive_count + negative_count,
    )
    logger.info("Trajectories: %s", args.output)
    logger.info("Labels: %s", args.labels_output)


if __name__ == "__main__":
    main()
