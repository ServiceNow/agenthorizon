#!/usr/bin/env python3
"""Compare AgentHorizon judge runs across models × harnesses × prompts.

Auto-discovers runs under experiments/<harness>/<model>/<exp_id>/ (and
.archive/), detects which prompt variant was used by sha256-matching the
snapshotted prompt.md against prompts/evaluate_trajectory_*.txt, computes
per-trajectory stats, and emits five markdown tables: accuracy, median per
trajectory, mean per trajectory, max per trajectory, total cost per run.

Usage:
    uv run python scripts/compare_runs.py
    uv run python scripts/compare_runs.py --min-n 50 --sort prompt
    uv run python scripts/compare_runs.py --harness gemini,codex
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "data" / "standard" / "agenthorizon_labels.jsonl"

# Rough USD pricing per 1M input / 1M output tokens. Used only for cost fallback
# when no per-call cost is captured (non-Anthropic harnesses).
PRICE = {
    "gemini-3.1-flash-lite-preview": (0.10, 0.40),
    "gemini-3.1-pro-preview": (1.25, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-opus-4-7": (15.00, 75.00),
}


def _price_for(model_slug: str) -> tuple[float, float]:
    """Look up pricing tolerant to slug/slash forms (e.g. google_gemini-... vs google/gemini-...)."""
    m = model_slug.replace("google_", "").replace("google/", "")
    return PRICE.get(m, PRICE.get(model_slug, (0.0, 0.0)))


def prompt_variant_map() -> dict[str, str]:
    """Map sha256(prompt_content) -> variant label."""
    m = {}
    for f in (REPO / "prompts").iterdir():
        if not f.is_file() or not f.name.startswith("evaluate_trajectory"):
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        variant = f.stem.replace("evaluate_trajectory", "").lstrip("_") or "v3"
        m[h] = variant.upper() if variant != "v3" else "P0(v3)"
    return m


def short_model(m: str) -> str:
    return (m.replace("google/", "")
             .replace("gemini-3.1-", "")
             .replace("-preview", "")
             .replace("claude-", ""))


def short_harness(h: str) -> str:
    return {"gemini": "gCLI", "codex": "Codex", "claude": "Claude",
            "openhands": "OH", "opencode": "OC"}.get(h, h)


def extract_per_trajectory(d: dict, harness: str) -> dict:
    """Return turns, tool_calls, tokens_in, tokens_out, cost_billed."""
    meta = d.get("_meta") or {}
    raw = meta.get("raw") or {}
    billed = meta.get("cost_usd")  # Claude Console fills this in
    # Gemini CLI path
    stats = raw.get("stats") if isinstance(raw, dict) else None
    if isinstance(stats, dict):
        models = stats.get("models") or {}
        turns = sum(int(((m or {}).get("api") or {}).get("totalRequests") or 0)
                    for m in models.values())
        tools = int((stats.get("tools") or {}).get("totalCalls") or 0)
        ti = to = 0
        for mv in models.values():
            tks = (mv or {}).get("tokens") or {}
            ti += int(tks.get("prompt") or 0)
            to += int(tks.get("candidates") or 0) + int(tks.get("thoughts") or 0)
        return {"turns": turns, "tools": tools, "ti": ti, "to": to, "billed": billed}
    # Codex path. Turn/tool counts backfilled from stdout/<tid>.jsonl events.
    usage = raw.get("usage") or {} if isinstance(raw, dict) else {}
    ti = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    to = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    # Backfilled fields (set by scripts/backfill_codex_stats.py or the live
    # parser in parse_codex_output). Absent → fall back to None.
    turns = raw.get("turns") if isinstance(raw, dict) else None
    tools = raw.get("tool_calls") if isinstance(raw, dict) else None
    return {"turns": turns, "tools": tools, "ti": ti, "to": to, "billed": billed}


def scan_runs(harnesses: list[str] | None, include_archive: bool) -> list[dict]:
    variant_map = prompt_variant_map()
    runs = []
    roots = [REPO / "experiments"]
    if include_archive:
        archive_root = REPO / "experiments" / ".archive"
        if archive_root.exists():
            # Each subdir of .archive/ (v3, v4, ...) is its own scanning root.
            roots.extend(sorted(p for p in archive_root.iterdir() if p.is_dir()))
    for root in roots:
        for harness_dir in sorted(root.iterdir()):
            if not harness_dir.is_dir() or harness_dir.name.startswith("."):
                continue
            if harnesses and harness_dir.name not in harnesses:
                continue
            for model_dir in sorted(harness_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for exp_dir in sorted(model_dir.iterdir()):
                    if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                        continue
                    prompt_file = exp_dir / "prompt.md"
                    sandbox_prompt = exp_dir / "sandbox" / "prompt.md"
                    prompt_path = prompt_file if prompt_file.exists() else sandbox_prompt
                    if not prompt_path.exists():
                        continue
                    ph = hashlib.sha256(prompt_path.read_bytes()).hexdigest()[:16]
                    variant = variant_map.get(ph, "?")
                    # If the prompt contains {{TRAJECTORY_ID}}, it's a template;
                    # its stored hash will differ from prompts/evaluate_trajectory_pN.txt
                    # because the snapshot keeps the placeholder while the runtime
                    # hash is per-trajectory. Re-match with placeholder-stripped hash.
                    if variant == "?":
                        pt_text = prompt_path.read_text()
                        import re
                        # Try stripping the placeholder marker tolerance
                        for cand_file in (REPO / "prompts").iterdir():
                            if not cand_file.name.startswith("evaluate_trajectory"):
                                continue
                            if cand_file.read_text() == pt_text:
                                variant = cand_file.stem.replace("evaluate_trajectory", "").lstrip("_").upper() or "P0(v3)"
                                break
                    runs.append({
                        "harness": harness_dir.name,
                        "model": model_dir.name,
                        "exp_id": exp_dir.name,
                        "prompt_variant": variant,
                        "path": exp_dir,
                        "archived": ".archive" in root.parts,
                    })
    return runs


def compute_stats(run: dict, labels: dict) -> dict | None:
    results = run["path"] / "results"
    if not results.exists():
        return None
    turns_l, tools_l, ti_l, to_l, cost_l = [], [], [], [], []
    pos_c = pos_t = neg_c = neg_t = 0
    turns_tracked = True
    for f in results.glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "success" not in d:
            continue
        pt = extract_per_trajectory(d, run["harness"])
        if pt["turns"] is None:
            turns_tracked = False
        else:
            turns_l.append(pt["turns"])
            tools_l.append(pt["tools"])
        ti_l.append(pt["ti"])
        to_l.append(pt["to"])
        pin, pout = _price_for(run["model"])
        if pt["billed"] is not None:
            cost_l.append(float(pt["billed"]))
        elif pin or pout:
            cost_l.append(pt["ti"] * pin / 1e6 + pt["to"] * pout / 1e6)
        else:
            cost_l.append(0.0)
        lb = labels.get(f.stem) or {}
        lbl = lb.get("label")
        s = bool(d.get("success"))
        if lbl == "positive":
            pos_t += 1; pos_c += int(s)
        elif lbl == "negative":
            neg_t += 1; neg_c += int(not s)
    n = len(ti_l)
    if n == 0:
        return None

    def md(x): return statistics.median(x) if x else 0
    def mn(x): return sum(x) / len(x) if x else 0

    return {
        "n": n,
        "acc_avg": ((pos_c / pos_t) + (neg_c / neg_t)) / 2 * 100 if pos_t and neg_t else None,
        "acc_pos": pos_c / pos_t * 100 if pos_t else None,
        "acc_neg": neg_c / neg_t * 100 if neg_t else None,
        "turns_med": md(turns_l) if turns_tracked else None,
        "turns_mean": mn(turns_l) if turns_tracked else None,
        "turns_max": max(turns_l) if (turns_tracked and turns_l) else None,
        "tools_med": md(tools_l) if turns_tracked else None,
        "tools_mean": mn(tools_l) if turns_tracked else None,
        "tools_max": max(tools_l) if (turns_tracked and tools_l) else None,
        "ti_med": md(ti_l), "ti_mean": mn(ti_l),
        "to_med": md(to_l), "to_mean": mn(to_l),
        "cost_med": md(cost_l), "cost_mean": mn(cost_l), "cost_total": sum(cost_l),
    }


def fmt_row(label: str, run: dict, s: dict, cols: list[tuple[str, str, str]]) -> str:
    parts = [label, f"{s['n']}"]
    for _, key, fmt in cols:
        v = s.get(key)
        if v is None:
            parts.append("—")
        elif fmt == "pct":
            parts.append(f"{v:.1f}")
        elif fmt == "int":
            parts.append(f"{int(v):,}")
        elif fmt == "fint":  # int but from float
            parts.append(f"{v:,.0f}")
        elif fmt == "float1":
            parts.append(f"{v:.1f}")
        elif fmt == "cost3":
            parts.append(f"${v:.3f}")
        elif fmt == "cost4":
            parts.append(f"${v:.4f}")
        elif fmt == "cost2":
            parts.append(f"${v:.2f}")
        else:
            parts.append(str(v))
    return "| " + " | ".join(parts) + " |"


def render_table(title: str, cols: list[tuple[str, str, str]], runs_with_stats: list[tuple[str, dict, dict]]):
    print(f"\n### {title}\n")
    header = ["Run", "N"] + [c[0] for c in cols]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for label, run, s in runs_with_stats:
        print(fmt_row(label, run, s, cols))


PROMPT_ORDER = ["P0", "P0(V3)", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def _prompt_sort_key(p: str) -> int:
    # strip suffix like "[arch]"; pick first known prompt label
    base = p.split()[0].upper() if p else "?"
    if base in PROMPT_ORDER:
        return PROMPT_ORDER.index(base)
    return 99


def render_pivot(enriched: list[tuple[str, dict, dict]]):
    """Rows = model × harness. Cols = prompt variant. Two tables: accuracy, total cost."""
    # group by (model, harness)
    by_row: dict[tuple[str, str], dict[str, dict]] = {}
    prompts_seen: set[str] = set()
    for label, run, s in enriched:
        key = (run["model"], run["harness"])
        variant = run["prompt_variant"] or "?"
        prompts_seen.add(variant)
        # If multiple runs per (model, harness, prompt), keep the biggest N
        existing = by_row.setdefault(key, {})
        if variant not in existing or existing[variant]["n"] < s["n"]:
            existing[variant] = s
    prompts = sorted(prompts_seen, key=_prompt_sort_key)

    def row_label(m, h):
        return f"{short_model(m)} @ {short_harness(h)}"

    sorted_rows = sorted(by_row.keys(),
                        key=lambda k: (_price_for(k[0])[1], k[1], k[0]))  # by output price (cheap first), then harness, model

    # --- Accuracy pivot ---
    print("\n### Accuracy by prompt variant (Avg %, pos%/neg% in parens)\n")
    header = ["Model × Harness"] + prompts
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for k in sorted_rows:
        row = [row_label(*k)]
        for p in prompts:
            s = by_row[k].get(p)
            if s and s["acc_avg"] is not None:
                tag = f"<sub>n={s['n']}</sub>" if s["n"] < 150 else ""
                row.append(f"**{s['acc_avg']:.1f}** ({s['acc_pos']:.0f}/{s['acc_neg']:.0f}){tag}")
            else:
                row.append("—")
        print("| " + " | ".join(row) + " |")

    # --- Total cost pivot ---
    print("\n### Total cost by prompt variant (USD)\n")
    header = ["Model × Harness"] + prompts
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for k in sorted_rows:
        row = [row_label(*k)]
        for p in prompts:
            s = by_row[k].get(p)
            if s:
                tag = f"<sub>n={s['n']}</sub>" if s["n"] < 150 else ""
                row.append(f"${s['cost_total']:.2f}{tag}")
            else:
                row.append("—")
        print("| " + " | ".join(row) + " |")

    # --- Tool calls pivot (mean) ---
    # Turns are harness-specific (Codex ≡ 1 per invocation, Gemini CLI increments per round-trip).
    # Tool calls are the comparable cross-harness behavior metric.
    any_tracked = any(s.get("tools_mean") is not None for _, _, s in enriched)
    if any_tracked:
        print("\n### Mean tool calls per trajectory by prompt variant\n")
        header = ["Model × Harness"] + prompts
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for k in sorted_rows:
            row = [row_label(*k)]
            for p in prompts:
                s = by_row[k].get(p)
                if s and s.get("tools_mean") is not None:
                    row.append(f"{s['tools_mean']:.1f}")
                elif s:
                    row.append("n/t")
                else:
                    row.append("—")
            print("| " + " | ".join(row) + " |")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-n", type=int, default=20, help="Skip runs with fewer results than this (default 20)")
    p.add_argument("--harness", default=None, help="Comma-separated harness names (e.g. gemini,codex,claude)")
    p.add_argument("--no-archive", action="store_true", help="Exclude .archive/ runs")
    p.add_argument("--sort", choices=["model", "prompt"], default="model",
                   help="Sort order (default: group by model × harness, then prompt)")
    p.add_argument("--pivot", action="store_true",
                   help="Pivot: rows = model×harness, cols = prompt. Shows accuracy + cost pivots.")
    p.add_argument("--json", action="store_true", help="Dump raw stats as JSON instead of markdown")
    p.add_argument("--labels", default=None,
                   help="Override labels file. Use for re-scoring against an archived split "
                        "(e.g. data/.archive/_legacy_leaky_split_20260418/agenthorizon_<val|test>_labels.jsonl).")
    args = p.parse_args()

    harnesses = args.harness.split(",") if args.harness else None
    labels_path = Path(args.labels) if args.labels else LABELS
    labels = {json.loads(l)["trajectory_id"]: json.loads(l)
              for l in labels_path.read_text().splitlines() if l.strip()}

    runs = scan_runs(harnesses, include_archive=not args.no_archive)
    enriched = []
    for r in runs:
        s = compute_stats(r, labels)
        if s is None or s["n"] < args.min_n:
            continue
        label = f"{short_model(r['model'])} {short_harness(r['harness'])} {r['prompt_variant']}"
        if r["archived"]:
            label += " [arch]"
        enriched.append((label, r, s))

    if args.sort == "prompt":
        enriched.sort(key=lambda x: (x[1]["prompt_variant"], x[1]["model"], x[1]["harness"]))
    else:
        enriched.sort(key=lambda x: (x[1]["model"], x[1]["harness"], x[1]["prompt_variant"]))

    if args.json:
        out = {label: {**s, "prompt_variant": r["prompt_variant"], "harness": r["harness"], "model": r["model"]}
               for label, r, s in enriched}
        print(json.dumps(out, indent=2, default=str))
        return

    print(f"# AgentHorizon run comparison\n")
    print(f"{len(enriched)} runs, min_n={args.min_n}" + (
        f", harness={args.harness}" if args.harness else ""))

    if args.pivot:
        render_pivot(enriched)
        return

    # Table 1: accuracy + tool behavior (what the judge did). Turns dropped —
    # they're harness-specific (Codex ≡ 1, Gemini CLI increments per round-trip)
    # and not comparable across harnesses. Tool calls are the comparable metric.
    render_table("Accuracy and tool use", [
        ("Avg %", "acc_avg", "pct"), ("Pos %", "acc_pos", "pct"), ("Neg %", "acc_neg", "pct"),
        ("Tool med", "tools_med", "fint"), ("Tool mean", "tools_mean", "float1"), ("Tool max", "tools_max", "fint"),
    ], enriched)

    # Table 2: tokens + cost (budget story).
    render_table("Tokens and cost", [
        ("Tin med", "ti_med", "fint"), ("Tin mean", "ti_mean", "fint"),
        ("Tout med", "to_med", "fint"), ("Tout mean", "to_mean", "fint"),
        ("$/traj med", "cost_med", "cost4"), ("$/traj mean", "cost_mean", "cost4"),
        ("Total $", "cost_total", "cost2"),
    ], enriched)


if __name__ == "__main__":
    main()
