# AgentHorizon

A long-horizon benchmark for evaluating judges of computer-use agents. AgentHorizon tests whether an LLM judge can accurately determine if a GUI agent completed a complex, multi-step desktop task, given a trajectory of screenshots and actions spanning 100-300+ steps across real-world applications.

**Key features:**
- **1,373 trajectories** (523 positive + 850 negative), blind-evaluated with UUID IDs
- **Long-horizon:** ~136 steps on average, with many exceeding 200+ steps
- **6 domains:** Data Science, Creativity, Education, General, Productivity, Development
- **3 operating systems:** macOS, Windows, Linux
- **323 unique application combinations** (desktop apps, browsers, IDEs, creative tools)
- **Adversarial negatives:** mismatched instruction-demonstration pairs that test judge discernment

## Repository layout

This repository is the **evaluation harness**: code, prompts, and judge implementations. The **benchmark data** (trajectories, screenshots, labels) is distributed separately as a dataset, see [Getting the data](#getting-the-data).

| Path | Description |
|---|---|
| `scripts/` | CLI utilities: convert trajectories, render Markdown, run evaluations, analyze results, build splits |
| `llm_judges/` | Judge implementation and trajectory preprocessing (compress / filter / summarize) |
| `prompts/evaluate_trajectory.txt` | The judge evaluation prompt |
| `experiments/.meta/template/` | Per-run layout template (output dirs); restore the trajectory inputs via the data download below |
| `paper/croissant.json` | Croissant dataset metadata |
| `STANDARD.md` | Trajectory schema, action types, label format, and how positive/negative pairs are constructed |

## Getting the data

The benchmark data is hosted as a dataset on the Hugging Face Hub:
**https://huggingface.co/datasets/ServiceNow/AgentHorizon**

```bash
pip install -U "huggingface_hub[cli]"

hf download ServiceNow/AgentHorizon --repo-type dataset --local-dir data/
```

This downloads:

| Path | What it is |
|---|---|
| `data/AgentHorizon.jsonl` | Labels for the main split (605 items). |
| `data/AgentHorizon-Simple.jsonl` | Labels for the simpler split (768 items). |
| `data/sandbox/data/markdowns/<trajectory_id>.md` | Each trajectory rendered as Markdown (the judge input). |
| `data/sandbox/data/jsons/<trajectory_id>.json` | The same trajectory in structured form. |
| `data/sandbox/data/media/images/<steps_id>/step_N.png` | Screenshots referenced by the Markdown and JSON. |

The two splits differ in difficulty. `AgentHorizon-Simple.jsonl` holds items that a strong open-weight agent already solves consistently; `AgentHorizon.jsonl` holds the harder items where judges disagree. The Markdown trajectories ship pre-rendered, so you can run a judge over them directly. `scripts/render_trajectories.py` is provided for regenerating Markdown from a raw standardized JSONL if you build your own trajectories.

### Restoring the experiment sandbox

The run template (`experiments/.meta/template/`) ships with the output-dir skeleton but not the trajectory inputs, which are part of the dataset. Restore them with:

```bash
hf download ServiceNow/AgentHorizon \
    --repo-type dataset --include "sandbox/*" \
    --local-dir experiments/.meta/template/
```

This recreates `experiments/.meta/template/sandbox/` with `data/markdowns/`, `data/jsons/`, and `data/media/images/`.

## Evaluating a judge

End-to-end: point a judge at the rendered Markdown trajectories, then score the verdicts.

### Step 1 (optional): Render trajectories to Markdown

The dataset already ships trajectories rendered to Markdown at `data/sandbox/data/markdowns/`, so you can skip to Step 2. If you are building your own trajectories from a standardized JSONL, render them with:

```bash
uv run python scripts/render_trajectories.py \
    your_trajectories.jsonl \
    your_md_dir
```

Each `.md` file has a `## Goal` section and a `## Steps` section. **Important:** do not pass `--no-images`. A multimodal judge needs the screenshots to assess task completion accurately.

### Step 2: Run the judge

The evaluation script pipes each Markdown trajectory into a judge CLI (`claude` or `gemini`) and collects a structured JSON verdict. It is resumable: re-running the same command skips completed trajectories.

```bash
uv run python scripts/evaluate_trajectories.py \
    --no-sandbox \
    --prompt prompts/evaluate_trajectory.txt \
    --md-dir data/sandbox/data/markdowns \
    --output-dir results/eval_agenthorizon_opus \
    --harness claude --model opus --workers 4
```

| Option | Description |
|---|---|
| `--no-sandbox` | Use the explicit `--md-dir` / `--output-dir` flow. By default the script runs in sandbox mode, provisioning an isolated working directory per experiment under `experiments/`. |
| `--prompt` | Path to the judge prompt (`prompts/evaluate_trajectory.txt`). |
| `--md-dir` | Directory of trajectory Markdown files (the rendered trajectories). |
| `--output-dir` | Where to write per-trajectory result JSONs. |
| `--harness` | Judge CLI: `claude` or `gemini`. |
| `--model` | Model name passed to the judge CLI. |
| `--workers` | Concurrent judge processes. |
| `--limit N` | Evaluate only the first N trajectories (for a pilot). |
| `--dry-run` | Preview without making API calls. |

The judge returns `{"success": true/false, "reasoning": "..."}` per trajectory, saved as `<trajectory_id>.json`.

**Using a different judge:** point your own judge at the Markdown files in `data/sandbox/data/markdowns/` plus the prompt, and save one JSON per trajectory named `<trajectory_id>.json` containing at least `{"success": true/false}`.

### Step 3: Score

Compare the verdicts against the published labels:

```bash
uv run python scripts/analyze_eval_results.py \
    --results-dir results/eval_agenthorizon_opus \
    --labels data/AgentHorizon.jsonl \
    --markdown results/eval_agenthorizon_opus/analysis.md
```

Scoring reports overall accuracy, the confusion matrix, and breakdowns by label and mistake type (Critical Mistake, Bad Side Effect, Misunderstanding). `success=true` means the judge predicts the trajectory is correct. Parse errors are penalized as incorrect predictions. Score the simpler split the same way with `--labels data/AgentHorizon-Simple.jsonl`.

### Cost and time estimates

Based on Claude Opus runs:

| Metric | Value |
|---|---|
| Cost per trajectory | ~$0.10 |
| Time per trajectory | ~15s |
| Total (1,373 trajectories) | ~$140, ~6 hours |

Use `--limit 50` for a quick pilot before the full run.

## Further reading

- [STANDARD.md](STANDARD.md) — trajectory schema, action types, labels, pairing.
- [`scripts/README.md`](scripts/README.md) — full script reference.
