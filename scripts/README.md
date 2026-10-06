# Scripts

Utility scripts for the AgentHorizon benchmark: downloading media, converting trajectories to the standardized format, generating Markdown for evaluation, and running LLM-based judge evaluations.

All scripts should be run from the **project root** using `uv run python`.

## Quick Start: Blind Evaluation Pipeline

The recommended end-to-end workflow for running a blind judge evaluation on AgentHorizon data:

```bash
cd research-agenthorizon

# 1. Download all videos and images from the delivery JSON
uv run python scripts/download_delivery.py --json data/new/final/final_delivery_batch.json

# 2. Verify downloads are complete
uv run python scripts/verify_delivery.py --json data/new/final/final_delivery_batch.json

# 3. Convert a raw delivery JSON to the construction-stage standard format
#    --data-dir maps media URLs to local screenshot paths.
#    Raw conversion does not reproduce the reviewed 1,373-item release by itself;
#    use the published Hugging Face files for benchmark evaluation.
uv run python scripts/convert_agenthorizon_trajectories.py \
    --input data/new/final/final_delivery_batch.json \
    --output data/standard/agenthorizon.jsonl \
    --labels-output data/standard/agenthorizon_labels.jsonl \
    --data-dir data/media

# 4. Convert standardized trajectories to Markdown for evaluation
#    Images are included by default (claude -p is multimodal and views screenshots)
uv run python scripts/render_trajectories.py \
    data/standard/agenthorizon.jsonl \
    data/standard/agenthorizon_md

# 5. Run blind evaluation using Claude as a judge
uv run python scripts/evaluate_trajectories.py \
    --prompt prompts/evaluate_trajectory.txt \
    --md-dir data/standard/agenthorizon_md \
    --output-dir results/eval_agenthorizon_opus \
    --workers 1 --model opus

# 6. Analyze judge results against ground truth
uv run python scripts/analyze_eval_results.py \
    --results-dir results/eval_agenthorizon_opus \
    --labels data/standard/agenthorizon_labels.jsonl \
    --trajectories data/standard/agenthorizon.jsonl \
    --output results/eval_agenthorizon_opus/analysis.json
```

### Important Notes

- **Labels must NEVER be exposed to the judge.** The labels file (`agenthorizon_labels_*.jsonl`) is only for post-hoc analysis after evaluation completes.
- Trajectory IDs are UUIDs — the judge cannot infer positive/negative from the ID.
- The JSONL is shuffled — positive and negative trajectories are intermixed.
- The evaluation is resumable — if interrupted, re-run the same command and it skips completed trajectories.
- Raw delivery conversion precedes evidence review and release curation. Do not use its row count as the benchmark denominator. The reviewed release contains **1,373 tasks (523 positive and 850 negative)**; see [`docs/benchmark-construction.md`](../docs/benchmark-construction.md).

### Legacy Quick Start (raw delivery JSON, no blind evaluation)

```bash
# Convert raw delivery JSON directly to Markdown (no standardization)
uv run python scripts/delivery_to_markdown.py data/new/final/final_delivery_batch.json \
    data/standard/agenthorizon_md --data-dir data/media

# Generate a CSV of paired tasks (for the comparer app)
uv run python scripts/generate_task_pairs_csv.py final_delivery_batch

# Evaluate raw Markdown (labels visible in filenames — not recommended)
uv run python scripts/evaluate_trajectories.py \
    --prompt prompts/evaluate_trajectory.txt \
    --md-dir data/standard/agenthorizon_md \
    --output-dir results/eval_agenthorizon \
    --workers 4 --model sonnet
```

---

## Scripts Reference

### download_delivery.py

Download all videos and images from a delivery JSON file using a parallel thread pool.

**Features:** resumable (skips existing files), retries with exponential backoff, error log for re-runs.

```bash
# Download everything (32 workers by default)
uv run python scripts/download_delivery.py --json data/new/final/final_delivery_batch.json

# Download only videos with 8 workers
uv run python scripts/download_delivery.py --type videos --workers 8

# Retry previously failed downloads
uv run python scripts/download_delivery.py --retry-errors
```

| Option | Description |
|---|---|
| `--json PATH` | Path to delivery JSON (default: `data/new/final/final_delivery_batch.json`) |
| `--type {videos,images,all}` | What to download (default: `all`) |
| `--workers N` | Number of parallel download threads (default: `32`) |
| `--retry-errors` | Only retry downloads from the error log |

**Output:** `data/media/{videos,images}/`

---

### verify_delivery.py

Verify that all media files referenced in the delivery JSON were properly downloaded.

```bash
# Basic check (existence + non-zero size)
uv run python scripts/verify_delivery.py

# Also validate file headers (magic bytes)
uv run python scripts/verify_delivery.py --check-content
```

| Option | Description |
|---|---|
| `--json PATH` | Path to delivery JSON (default: `data/new/final/final_delivery_batch.json`) |
| `--check-content` | Validate file content via magic bytes (MP4, PNG, JPEG, WebM) |

Issues are saved to `download_errors.json` for use with `download_delivery.py --retry-errors`.

---

### delivery_to_markdown.py

Convert delivery trajectories to human-readable Markdown documents with goals, step-by-step actions, and optional screenshot links.

```bash
# Convert all trajectories
uv run python scripts/delivery_to_markdown.py data/new/final/final_delivery_batch.json \
    data/standard/agenthorizon_md --data-dir data/media

# Convert a single trajectory by index
uv run python scripts/delivery_to_markdown.py data/new/final/final_delivery_batch.json \
    data/standard/agenthorizon_md --index 0

# Convert by deliverable ID, with metadata
uv run python scripts/delivery_to_markdown.py data/new/final/final_delivery_batch.json \
    data/standard/agenthorizon_md --id 000231601-231601-0000-000000231601 --metadata
```

| Option | Description |
|---|---|
| `input` | Path to the delivery JSON file (required) |
| `output_dir` | Output directory for .md files (default: `<input_stem>/`) |
| `--index N` | Convert only the trajectory at index N |
| `--id ID` | Convert only the trajectory with this deliverable_id |
| `--no-images` | Omit screenshot image links |
| `--metadata` | Include OS, task type, trajectory type, etc. |
| `--milestones` | Include milestone subgoals |
| `--data-dir DIR` | Path to downloaded media (for relative image/video paths) |

**Output:** One `<deliverable_id>.md` file per trajectory in the output directory.

---

### generate_task_pairs_csv.py

Generate a CSV pairing tasks based on their `paired_task_id` field in delivery JSON files.

```bash
# Process all JSON files in data/
uv run python scripts/generate_task_pairs_csv.py

# List available files
uv run python scripts/generate_task_pairs_csv.py --list

# Process a specific file (partial match, case-insensitive)
uv run python scripts/generate_task_pairs_csv.py delivery20260128

# Custom output path
uv run python scripts/generate_task_pairs_csv.py -o my_output.csv
```

| Option | Description |
|---|---|
| `FILE` | File name pattern to match (case-insensitive partial match) |
| `-l, --list` | List available delivery files and exit |
| `-o, --output` | Output CSV path (default: `data/pairs_<filename>.csv`) |

**Output CSV columns:** `deliverable_id_a`, `deliverable_id_b`, `task_a`, `task_b`, `mistake_type`, `trajectory_type`

The output CSV can be imported into Google Sheets or Excel for pair-level review.

---

### convert_human_trajectories.py

Convert a delivery JSON file into the standardized trajectory format. Produces two files: a blind JSONL (UUID IDs, shuffled, no labels) and a ground-truth labels JSONL.

The raw converter emits a construction-stage pool before evidence review. The released benchmark uses 425 complete pairs (850 candidate positives and 850 swapped negatives); review removes 327 unresolved candidate positives, yielding 1,373 released tasks. Use the published dataset files rather than raw-converter output when reproducing benchmark scores.

```bash
# Standard conversion with local screenshot paths
uv run python scripts/convert_agenthorizon_trajectories.py \
    --input data/new/final/final_delivery_batch.json \
    --output data/standard/agenthorizon.jsonl \
    --labels-output data/standard/agenthorizon_labels.jsonl \
    --data-dir data/media

# Without --data-dir (stores GCS URLs instead of local paths)
uv run python scripts/convert_agenthorizon_trajectories.py \
    --input data/new/final/final_delivery_batch.json \
    --output data/standard/agenthorizon.jsonl \
    --labels-output data/standard/agenthorizon_labels.jsonl
```

| Option | Description |
|---|---|
| `--input PATH` | Path to delivery JSON file (required) |
| `--output PATH` | Output JSONL file for trajectories (required) |
| `--labels-output PATH` | Output JSONL file for ground-truth labels (required) |
| `--data-dir DIR` | Path to downloaded media directory (for local screenshot paths). Maps GCS URLs to `<dir>/images/<did>/step_<N>.png`. If omitted, stores original GCS URLs. |
| `--seed N` | Random seed for shuffling (default: `42`) |

**Trajectory JSONL fields:** `version`, `trajectory_id` (UUID), `source`, `task` (instruction, category, applications), `environment`, `steps`, `milestones`

**Labels JSONL fields:** `trajectory_id` (UUID), `label` (positive/negative), `original_id`, `mistake_type` (for negatives), `paired_id`, `negative_source`

---

### render_trajectories.py

Convert standardized trajectory JSONL to readable Markdown documents. No labels or identifying metadata are included — safe for blind evaluation.

```bash
# Convert all trajectories
uv run python scripts/render_trajectories.py \
    data/standard/agenthorizon.jsonl \
    data/standard/agenthorizon_md

# Without screenshot links
uv run python scripts/render_trajectories.py \
    data/standard/agenthorizon.jsonl \
    data/standard/agenthorizon_md \
    --no-images

# Convert a single trajectory by index
uv run python scripts/render_trajectories.py \
    data/standard/agenthorizon.jsonl \
    data/standard/agenthorizon_md \
    --index 0
```

| Option | Description |
|---|---|
| `input` | Path to the standard JSONL file (required) |
| `output_dir` | Output directory for .md files (default: `<input_stem>_md/`) |
| `--index N` | Convert only the trajectory at line index N |
| `--id UUID` | Convert only the trajectory with this UUID |
| `--no-images` | Omit screenshot image links |

**Output:** One `<uuid>.md` file per trajectory in the output directory.

---

### evaluate_trajectories.py

Evaluate trajectory Markdown files by piping each one into `claude -p` and collecting structured JSON results. Supports concurrent evaluation, dry-run mode, and automatic resumption.

```bash
# Evaluate all trajectories
uv run python scripts/evaluate_trajectories.py \
    --prompt prompts/evaluate_trajectory.txt \
    --md-dir data/standard/agenthorizon_md \
    --output-dir results/eval_agenthorizon \
    --workers 4 --model sonnet

# Dry run (see what would be evaluated without calling claude)
uv run python scripts/evaluate_trajectories.py \
    --prompt prompts/evaluate_trajectory.txt \
    --md-dir data/standard/agenthorizon_md \
    --output-dir results/eval_agenthorizon \
    --dry-run

# Evaluate first 5 only (for testing)
uv run python scripts/evaluate_trajectories.py \
    --prompt prompts/evaluate_trajectory.txt \
    --md-dir data/standard/agenthorizon_md \
    --output-dir results/eval_agenthorizon \
    --limit 5
```

| Option | Description |
|---|---|
| `--prompt PATH` | Path to evaluation prompt file (required) |
| `--md-dir DIR` | Directory containing .md trajectory files (required) |
| `--output-dir DIR` | Where to write per-trajectory result JSONs (required) |
| `--workers N` | Max concurrent `claude -p` subprocesses (default: `1`) |
| `--model MODEL` | Model alias: `sonnet`, `opus`, `haiku` (default: `sonnet`) |
| `--limit N` | Only evaluate first N trajectories |
| `--dry-run` | Print what would be done without calling claude |

**Output:** Per-trajectory `.json` results, session transcripts in `sessions/`, a `_summary.json`, and success/failure CSVs in `results/{successes,failures}/`.

---

### jsonl_to_md.py

Convert a Claude Code session JSONL transcript into a readable Markdown document.

```bash
# Print to stdout
uv run python scripts/jsonl_to_md.py session.jsonl

# Write to file
uv run python scripts/jsonl_to_md.py session.jsonl output.md
```

| Argument | Description |
|---|---|
| `input.jsonl` | Path to the JSONL session file (required) |
| `output.md` | Output Markdown path (optional; prints to stdout if omitted) |

The output includes user messages, assistant responses, tool calls with parameters, tool results, and collapsible reasoning blocks.

---

### analyze_eval_results.py

Compare judge evaluation results against ground-truth labels. Computes classification metrics (accuracy, precision, recall, F1), a confusion matrix, and breakdowns by label, mistake type, and negative source.

```bash
# Print analysis to terminal
uv run python scripts/analyze_eval_results.py \
    --results-dir results/eval_agenthorizon_opus \
    --labels data/standard/agenthorizon_labels.jsonl \
    --trajectories data/standard/agenthorizon.jsonl

# Also save a JSON report
uv run python scripts/analyze_eval_results.py \
    --results-dir results/eval_agenthorizon_opus \
    --labels data/standard/agenthorizon_labels.jsonl \
    --trajectories data/standard/agenthorizon.jsonl \
    --output results/eval_agenthorizon_opus/analysis.json
```

| Option | Description |
|---|---|
| `--results-dir DIR` | Directory containing per-trajectory result JSON files (required) |
| `--labels PATH` | Path to ground-truth labels JSONL file (required) |
| `--trajectories PATH` | Path to trajectories JSONL file (for UUID remapping when labels were regenerated with new UUIDs after evaluation) |
| `--md-dir DIR` | Markdown directory for UUID remapping (auto-detected from `_summary.json` if omitted) |
| `--output PATH` | Path to write JSON report (optional; prints to stdout if omitted) |

**Mapping:** `success=true` → predicted positive (task completed), `success=false` → predicted negative. `label="positive"` → actual correct execution, `label="negative"` → mismatched instruction/demo.

**Output sections:**
- Overall metrics (accuracy, precision, recall, F1)
- Confusion matrix (TP, FP, TN, FN)
- Accuracy by label (positive vs negative)
- Negative breakdown by mistake type (Critical Mistake, Bad Side Effect, etc.)
- Negative breakdown by source (parent\_instruction\_child\_trajectory, etc.)
