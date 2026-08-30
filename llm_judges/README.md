# Phase 2: LLM-as-Judge Evaluation

Direct LLM API calls (no agent harness scaffolding) for judging computer-use
agent trajectories. The key question: does agent scaffolding help, or can a
raw API call judge just as well?

## Architecture

```
data/standard/agenthorizon.jsonl  -->  Preprocessing  -->  data/preprocessed/<approach>/
                                                          |
                                                    evaluate.py
                                                          |
                                                    results/eval_agenthorizon_llmjudge_<approach>_<model>/
                                                          |
                                                    analyze_eval_results.py  (Phase 1 compatible)
                                                          |
                                                    scripts/results_to_submission.py
                                                          |
                                                    results/analysis/submission_<run>.jsonl
```

## Approach A: Naive Compression (`preprocess_compress.py`)

Resize every screenshot to a small resolution and ship the whole trajectory
(text + base64 JPEGs) to the judge in one API call.

- Auto-resolution: binary-searches per-trajectory for the largest size that
  fits `--max-tokens`. Use `--target-width` / `--target-height` to override
  with a fixed resolution (recommended for cross-model comparability, since
  some vision encoders are resolution-invariant and others aren't).
- Token estimator is calibrated against Gemini-style tokenization. The same
  preprocessed file produces ~25K Qwen / ~28K Gemma / ~100K Gemini tokens
  for a typical 100-image trajectory.

## Quick Start

### Preprocess once (1× — auto-resolution, fits 100K Gemini tokens)

Preprocessed payloads are generated locally and are not committed. Run:

```bash
uv run python llm_judges/preprocess_compress.py \
    --input data/standard/agenthorizon.jsonl \
    --output-dir data/preprocessed/compress \
    --max-tokens 100000
```

### Preprocess once (2.2× — fixed 1126×730, more visual detail for open VL models)

Generate the higher-resolution variant with:

```bash
uv run python llm_judges/preprocess_compress.py \
    --input data/standard/agenthorizon.jsonl \
    --output-dir data/preprocessed/compress_2_2x \
    --target-width 1126 --target-height 730 \
    --max-tokens 10000000
```

### Evaluate with Gemini

```bash
# Gemini 3.1 Flash Lite (cheap, fast)
uv run python llm_judges/evaluate.py \
    --preprocessed-dir data/preprocessed/compress_2_2x \
    --output-dir results/eval_agenthorizon_llmjudge_compress_2_2x_gemini_flash_lite \
    --provider gemini \
    --model gemini-3.1-flash-lite-preview \
    --workers 8

# Gemini 3.1 Pro (higher quality, ~3-4× cost)
uv run python llm_judges/evaluate.py \
    --preprocessed-dir data/preprocessed/compress_2_2x \
    --output-dir results/eval_agenthorizon_llmjudge_compress_2_2x_gemini_pro \
    --provider gemini \
    --model gemini-3.1-pro-preview \
    --workers 8
```

> Gemini's vision tokenizer is resolution-invariant: 1× and 2.2× cost the
> same number of prompt tokens for the same trajectory. So the 2.2× preprocess
> is "free" on Gemini and gives strictly better detail to the open VL models.

### Evaluate with self-hosted vLLM (YUL201 cluster)

For open models, pass `--provider vllm` + a `--base-url`. The bearer token
is read from `~/.vllm/.env` (`VLLM_BEARER_TOKEN=...`).

```bash
# Gemma 4 31B (no thinking — Gemma is not reasoning-tuned)
uv run python llm_judges/evaluate.py \
    --preprocessed-dir data/preprocessed/compress_2_2x \
    --output-dir results/eval_agenthorizon_llmjudge_compress_2_2x_gemma4_31b \
    --provider vllm \
    --base-url https://vllm-gemma4-31b-sn.mcgill-nlp.org/v1 \
    --model google/gemma-4-31B-it \
    --workers 8

# Qwen 3.6 27B without thinking
uv run python llm_judges/evaluate.py \
    --preprocessed-dir data/preprocessed/compress_2_2x \
    --output-dir results/eval_agenthorizon_llmjudge_compress_2_2x_qwen36_27b \
    --provider vllm \
    --base-url https://vllm-qwen36-27b-sn.mcgill-nlp.org/v1 \
    --model Qwen/Qwen3.6-27B \
    --workers 8

# Qwen 3.6 27B with thinking enabled (captures reasoning trace into _meta.thinking;
# auto-bumps max_tokens 2048 -> 16384)
uv run python llm_judges/evaluate.py \
    --preprocessed-dir data/preprocessed/compress_2_2x \
    --output-dir results/eval_agenthorizon_llmjudge_compress_2_2x_qwen36_27b_thinking \
    --provider vllm \
    --base-url https://vllm-qwen36-27b-sn.mcgill-nlp.org/v1 \
    --model Qwen/Qwen3.6-27B \
    --enable-thinking \
    --workers 8
```

Other endpoints on the cluster (same `--provider vllm` pattern, swap base URL):

| Slug | Base URL | Model id |
|---|---|---|
| qwen35-9b | `https://vllm-qwen35-9b-sn.mcgill-nlp.org/v1` | `Qwen/Qwen3.5-9B` |
| qwen36-27b | `https://vllm-qwen36-27b-sn.mcgill-nlp.org/v1` | `Qwen/Qwen3.6-27B` |
| qwen36-35b-a3b | `https://vllm-qwen36-35b-a3b-sn.mcgill-nlp.org/v1` | `Qwen/Qwen3.6-35B-A3B` |
| gemma4-26b-a4b | `https://vllm-gemma4-26b-a4b-sn.mcgill-nlp.org/v1` | `google/gemma-4-26B-A4B-it` |
| gemma4-31b | `https://vllm-gemma4-31b-sn.mcgill-nlp.org/v1` | `google/gemma-4-31B-it` |

### Analyze and convert to submission format

```bash
# Compute accuracy / confusion matrix / breakdowns
uv run python scripts/analyze_eval_results.py \
    --results-dir results/eval_agenthorizon_llmjudge_compress_2_2x_qwen36_27b \
    --labels data/standard/agenthorizon_labels.jsonl \
    --trajectories data/standard/agenthorizon.jsonl \
    --output results/analysis/eval_agenthorizon_llmjudge_compress_2_2x_qwen36_27b.json

# Convert to AgentHorizon submission template
uv run python scripts/results_to_submission.py \
    --results-dir results/eval_agenthorizon_llmjudge_compress_2_2x_qwen36_27b
# -> results/analysis/submission_eval_agenthorizon_llmjudge_compress_2_2x_qwen36_27b.jsonl
```

## Local data layout

The screenshots are distributed with the dataset rather than checked into this
repository. Preprocessing reads them from
`data/media/images/<original_id>/step_<N>.png` through the labels mapping and
writes derived payloads under `data/preprocessed/`:

| Path | What |
|---|---|
| `data/preprocessed/compress` | Approach A at **1×** (auto-resolution, 512×332 max) — `--max-tokens 100000` |
| `data/preprocessed/compress_2_2x` | Approach A at **2.2×** (1126×730 fixed) |
| `data/media/images` | Raw screenshots, mapped through `original_id` in the labels |

## Committed results provenance

| File | Preprocessing | Model | Notes |
|---|---|---|---|
| `results/analysis/eval_agenthorizon_llmjudge_compress_gemini_flash_lite.json` | Approach A, **1×** (512×332 max, 100K-token budget) | Gemini 3.1 Flash Lite | Historical construction-pool run; not a released 1,373-item benchmark result |
| `results/analysis/submission_eval_agenthorizon_llmjudge_compress_gemini_flash_lite.jsonl` | (same as above) | (same) | Same predictions in submission template format |

## Testing

```bash
# Dry run (no output, just token estimates)
uv run python llm_judges/preprocess_compress.py \
    --input data/standard/agenthorizon.jsonl \
    --output-dir /tmp/test_compress \
    --limit 5 --dry-run

# Small sample
uv run python llm_judges/preprocess_compress.py \
    --input data/standard/agenthorizon.jsonl \
    --output-dir /tmp/test_compress \
    --limit 3

# Evaluate with dry run (no API key needed)
uv run python llm_judges/evaluate.py \
    --preprocessed-dir /tmp/test_compress \
    --output-dir /tmp/test_eval \
    --provider openrouter \
    --model qwen/qwen3.5-9b \
    --dry-run
```

## File Structure

```
llm_judges/
    __init__.py              # Package init
    utils.py                 # Shared utilities (loading, encoding, token estimation)
    preprocess_compress.py   # Approach A: naive compression
    preprocess_filter.py     # Approach B: two-stage filtering (see below)
    preprocess_summarize.py  # Approach C: two-stage summarization (see below)
    evaluate.py              # Evaluation runner (direct API calls)
    README.md                # This file
```

## Dependencies

All dependencies are already in the project environment:
- `PIL/Pillow` -- image loading and resizing
- `aiohttp` -- async HTTP for OpenAI-compatible providers
- `google-genai` -- Gemini API client
- Standard library: `asyncio`, `json`, `base64`, `pathlib`

## API Keys

- OpenRouter: `OPENROUTER_API_KEY` in `~/.claude/.env`
- Gemini: `GEMINI_API_KEY` in `~/.gemini/.env`
- Self-hosted vLLM: `VLLM_BEARER_TOKEN` in `~/.vllm/.env`

---

## Other approaches (B and C)

These exist in the codebase but are not the active path right now. Documented
for completeness; will become more relevant if Approach A doesn't fit a model.

### Approach B: Two-Stage Filtering (`preprocess_filter.py`)

Stage 1: Cheap model selects top-K most important steps (text-only).
Stage 2: Judge sees selected steps with full-resolution screenshots.

- Stage 1 can use LLM (`--stage1-provider`/`--stage1-model`) or heuristic (`--heuristic`)
- Default top-K: 15 steps (adjustable)
- Full-resolution screenshots for selected steps only

```bash
uv run python llm_judges/preprocess_filter.py \
    --input data/standard/agenthorizon.jsonl \
    --output-dir data/preprocessed/filter \
    --heuristic --top-k 15

uv run python llm_judges/evaluate.py \
    --preprocessed-dir data/preprocessed/filter \
    --output-dir results/eval_agenthorizon_llmjudge_filter_qwen397b \
    --provider openrouter \
    --model qwen/qwen3.5-397b-a17b \
    --workers 4
```

### Approach C: Two-Stage Summarization (`preprocess_summarize.py`)

Stage 1: Vision model describes each screenshot in text.
Stage 2: Judge gets text-only trajectory with screenshot descriptions.

- Most expensive preprocessing but smallest judge input
- Supports batched screenshot descriptions to reduce API calls
- Judge input is text-only (works with any model, any context window — useful
  for text-only models like Qwen 3.5 9B that can't ingest 100+ images)

```bash
uv run python llm_judges/preprocess_summarize.py \
    --input data/standard/agenthorizon.jsonl \
    --output-dir data/preprocessed/summarize \
    --stage1-provider gemini \
    --stage1-model gemini-2.0-flash-lite \
    --workers 16

uv run python llm_judges/evaluate.py \
    --preprocessed-dir data/preprocessed/summarize \
    --output-dir results/eval_agenthorizon_llmjudge_summarize_qwen397b \
    --provider openrouter \
    --model qwen/qwen3.5-397b-a17b \
    --workers 4
```
