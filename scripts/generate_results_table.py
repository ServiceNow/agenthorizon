"""Generate results tables from experiment analysis JSONs.

Reads all analysis JSON files from the experiments directory and produces
a Markdown and/or LaTeX results table.

Usage:
    # Print Markdown table to stdout
    uv run python scripts/generate_results_table.py

    # Write Markdown README
    uv run python scripts/generate_results_table.py --markdown results/analysis/README.md

    # Write LaTeX table
    uv run python scripts/generate_results_table.py --latex results/analysis/results.tex

    # Both
    uv run python scripts/generate_results_table.py \
        --markdown results/analysis/README.md \
        --latex results/analysis/results.tex
"""

import argparse
import json
import re
from pathlib import Path

EXPERIMENTS_DIR = Path("results/analysis")

# Map analysis JSON filenames to judge/model metadata.
# Pattern: eval_<dataset>_<judge_name>.json
# We extract what we can from the filename and augment from the JSON.
JUDGE_LABELS = {
    "opus": ("Claude Code", "Opus 4.6"),
    "sonnet": ("Claude Code", "Sonnet 4.6"),
    "haiku": ("Claude Code", "Haiku 4.5"),
    "gemini_pro": ("Gemini CLI", "3.1 Pro"),
    "gemini_flash_lite": ("Gemini CLI", "Flash Lite"),
    "gemini": ("Gemini CLI", "3.1 Pro"),
    "codex_qwen397b": ("Codex", "Qwen 3.5 397B"),
    "codex_qwen27b": ("Codex", "Qwen 3.5 27B"),
    "codex_qwen9b": ("Codex", "Qwen 3.5 9B"),
    "codex_gemma31b": ("Codex", "Gemma 4 31B"),
    "codex_gpt54": ("Codex", "GPT-5.4"),
    "codex_gpt54mini": ("Codex", "GPT-5.4 Mini"),
    "codex_sonnet_or": ("Codex+OR", "Sonnet 4.6"),
    "api_qwen": ("Direct API", "Qwen 3.5 397B"),
    "openhands_qwen397b": ("OpenHands", "Qwen 3.5 397B"),
    "opencode_qwen397b": ("OpenCode", "Qwen 3.5 397B"),
    "codex_gemma26b": ("Codex", "Gemma 4 26B"),
    "codex_gemini_pro_or": ("Codex", "Gemini 3.1 Pro"),
    "codex_gemini_flash_or": ("Codex", "Gemini 3.1 Flash Lite"),
    "opencode_gemini_pro": ("OpenCode", "Gemini 3.1 Pro"),
    "opencode_gemini_flash": ("OpenCode", "Gemini 3.1 Flash Lite"),
    "openhands_gemini_pro": ("OpenHands", "Gemini 3.1 Pro"),
    "openhands_gemini_flash": ("OpenHands", "Gemini 3.1 Flash Lite"),
}


def parse_run_info(filepath: Path, data: dict) -> dict:
    """Extract run metadata from filename and analysis JSON."""
    stem = filepath.stem  # e.g., eval_agenthorizon_gemini
    # Try matching known multi-word judge keys first (e.g., codex_qwen)
    judge_key = None
    dataset = None
    for key in sorted(JUDGE_LABELS.keys(), key=len, reverse=True):
        suffix = f"_{key}"
        if stem.endswith(suffix):
            judge_key = key
            dataset = stem.removeprefix("eval_").removesuffix(suffix)
            break
    if judge_key is None:
        parts = stem.split("_")
        judge_key = parts[-1]
        dataset = "_".join(parts[1:-1])
    provider, model = JUDGE_LABELS.get(judge_key, ("Unknown", judge_key))

    return {
        "file": filepath.name,
        "provider": provider,
        "model": model,
        "dataset": dataset,
        "judge_key": judge_key,
        "metrics": data.get("metrics", {}),
        "confusion_matrix": data.get("confusion_matrix", {}),
        "summary": data.get("summary", {}),
        "accuracy_by_label": data.get("accuracy_by_label", {}),
        "negative_by_mistake_type": data.get("negative_by_mistake_type", {}),
        "negative_by_source": data.get("negative_by_source", {}),
    }


def load_experiments(experiments_dir: Path) -> list[dict]:
    """Load all analysis JSONs and return sorted run info."""
    runs = []
    for f in sorted(experiments_dir.glob("eval_*.json")):
        data = json.loads(f.read_text())
        runs.append(parse_run_info(f, data))
    return runs


def pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def generate_markdown(runs: list[dict]) -> str:
    lines = [
        "# AgentHorizon: Experimental Results",
        "",
        "## Results Table",
        "",
        "| Judge | Model | Dataset | N | Parse Err | Accuracy | Precision | Recall | F1 | Pos Acc | Neg Acc |",
        "|-------|-------|---------|---|-----------|----------|-----------|--------|-----|---------|---------|",
    ]
    for r in runs:
        m = r["metrics"]
        al = r["accuracy_by_label"]
        n = r["summary"].get("matched", r["summary"].get("total_results", "?"))
        parse_err = r["summary"].get("parse_errors", 0)
        pos_acc = pct(al.get("positive", {}).get("accuracy", 0))
        neg_acc = pct(al.get("negative", {}).get("accuracy", 0))
        lines.append(
            f"| {r['provider']} | {r['model']} | {r['dataset']} | {n:,} | {parse_err} "
            f"| {pct(m.get('accuracy', 0))} | {pct(m.get('precision', 0))} "
            f"| {pct(m.get('recall', 0))} | {pct(m.get('f1', 0))} "
            f"| {pos_acc} | {neg_acc} |"
        )

    # Mistake type breakdown
    mistake_types = ["Critical Mistake", "Bad Side Effect", "Misunderstanding of the Instructions"]
    lines += [
        "",
        "## Negative Breakdown by Mistake Type",
        "",
        "| Judge | Model | Critical Mistake | Bad Side Effect | Misunderstanding |",
        "|-------|-------|-----------------|-----------------|------------------|",
    ]
    for r in runs:
        mt = r["negative_by_mistake_type"]
        cells = []
        for t in mistake_types:
            d = mt.get(t, {})
            if d:
                cells.append(f"{pct(d['accuracy'])} ({d['correct']}/{d['total']})")
            else:
                cells.append("-")
        lines.append(f"| {r['provider']} | {r['model']} | {' | '.join(cells)} |")

    # Source breakdown
    lines += [
        "",
        "## Negative Breakdown by Source",
        "",
        "| Judge | Model | Child Instr + Parent Traj | Parent Instr + Child Traj |",
        "|-------|-------|--------------------------|--------------------------|",
    ]
    for r in runs:
        ns = r["negative_by_source"]
        cells = []
        for src in ["child_instruction_parent_trajectory", "parent_instruction_child_trajectory"]:
            d = ns.get(src, {})
            if d:
                cells.append(f"{pct(d['accuracy'])} ({d['correct']}/{d['total']})")
            else:
                cells.append("-")
        lines.append(f"| {r['provider']} | {r['model']} | {' | '.join(cells)} |")

    # Notes
    lines += [
        "",
        "## Methodology",
        "",
        "- **Parse error penalty:** If a judge fails to return valid JSON after retries, the trajectory counts as an incorrect prediction for whichever class it belongs to. This penalizes judges that cannot follow the output format.",
        "- **Pos Acc** = accuracy on positive trajectories (correctly identifying task success).",
        "- **Neg Acc** = accuracy on negative trajectories (correctly identifying task failure).",
        "- **Precision** = TP / (TP + FP). **Recall** = TP / (TP + FN). Positive = task completed successfully.",
        "- **N** = number of trajectories matched to labels (including penalized parse errors). **Parse Err** = count of trajectories that failed to produce valid JSON.",
        "",
        "*Auto-generated by `scripts/generate_results_table.py`.*",
    ]

    return "\n".join(lines) + "\n"


def generate_latex(runs: list[dict]) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{AgentHorizon judge evaluation results.}",
        r"\label{tab:agenthorizon-results}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Judge & Model & N & Accuracy & Precision & Recall & F1 & Pos Acc & Neg Acc \\",
        r"\midrule",
    ]
    for r in runs:
        m = r["metrics"]
        al = r["accuracy_by_label"]
        n = r["summary"].get("matched", r["summary"].get("total_results", "?"))
        pos_acc = pct(al.get("positive", {}).get("accuracy", 0))
        neg_acc = pct(al.get("negative", {}).get("accuracy", 0))
        lines.append(
            f"{r['provider']} & {r['model']} & {n:,} & {pct(m.get('accuracy', 0))} "
            f"& {pct(m.get('precision', 0))} & {pct(m.get('recall', 0))} "
            f"& {pct(m.get('f1', 0))} & {pos_acc} & {neg_acc} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Negative trajectory detection accuracy by mistake type.}",
        r"\label{tab:agenthorizon-mistake-type}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Judge & Model & Critical Mistake & Bad Side Effect & Misunderstanding \\",
        r"\midrule",
    ]
    mistake_types = ["Critical Mistake", "Bad Side Effect", "Misunderstanding of the Instructions"]
    for r in runs:
        mt = r["negative_by_mistake_type"]
        cells = []
        for t in mistake_types:
            d = mt.get(t, {})
            cells.append(pct(d["accuracy"]) if d else "-")
        lines.append(
            f"{r['provider']} & {r['model']} & {' & '.join(cells)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate results tables from experiment JSONs.")
    parser.add_argument("--dir", default=str(EXPERIMENTS_DIR),
                        help=f"Experiments results directory (default: {EXPERIMENTS_DIR})")
    parser.add_argument("--markdown", help="Write Markdown table to this file")
    parser.add_argument("--latex", help="Write LaTeX table to this file")
    args = parser.parse_args()

    experiments_dir = Path(args.dir)
    runs = load_experiments(experiments_dir)

    if not runs:
        print(f"No eval_*.json files found in {experiments_dir}")
        return

    md = generate_markdown(runs)

    if args.markdown:
        md_path = Path(args.markdown)
        # Preserve manually-written content after the auto-generated section
        SEPARATOR = "---\n\n## Next Steps"
        if md_path.exists():
            existing = md_path.read_text()
            sep_idx = existing.find(SEPARATOR)
            if sep_idx != -1:
                manual_section = existing[sep_idx:]
                md = md.rstrip() + "\n\n" + manual_section
        md_path.write_text(md)
        print(f"Wrote Markdown to {args.markdown}")

    if args.latex:
        tex = generate_latex(runs)
        Path(args.latex).write_text(tex)
        print(f"Wrote LaTeX to {args.latex}")

    if not args.markdown and not args.latex:
        print(md)


if __name__ == "__main__":
    main()
