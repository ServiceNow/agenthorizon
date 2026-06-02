# `experiments/`

Per-run experiment directory. Every call to `scripts/evaluate_trajectories.py` (sandbox mode, the default) provisions a fresh `experiments/<harness>/<model_slug>/<exp_id>/` folder by rsync'ing `.meta/template/` into it and appends a row to `.meta/exp_ids.jsonl`.

## Layout

```
experiments/
├── .gitignore                       # committed
├── README.md                        # committed (you are here)
│
├── .meta/                           # committed: template + registry
│   ├── exp_ids.jsonl                # append-only registry, one row per run
│   └── template/                    # skeleton copied into each new run dir
│       ├── errors.jsonl             # empty
│       ├── analysis/.gitkeep
│       ├── results/.gitkeep
│       ├── stdout/.gitkeep
│       ├── stderr/.gitkeep
│       └── sandbox/                 # CWD for the judge process
│           ├── AGENTS.md            # judge framework doc
│           ├── CLAUDE.md            # -> AGENTS.md (symlink)
│           ├── GEMINI.md            # -> AGENTS.md (symlink)
│           ├── prompt.md            # placeholder, overwritten per run
│           ├── .gitkeep
│           ├── data/media/images    # symlink -> ../../../../../data/media/images
│           ├── agenthorizon_json/   # 1,700 <traj_id>.json (committed; ~97 MB raw, ~10 MB packed)
│           └── agenthorizon_md/     # 1,700 <traj_id>.md   (committed; ~53 MB raw, ~10 MB packed)
│
├── .archive/                        # NOT committed (dotdir, local/EAI-only)
│   └── v{N}/                        # one subdir per archived cohort
│       ├── README.md                # why the cohort was archived
│       └── <harness>/<model_slug>/<exp_id>/
│           └── {config.json, env.json, prompt.md, errors.jsonl,
│              analysis/, results/, stdout/, stderr/, sandbox/}
│
└── <harness>/                       # live per-run dirs (NOT committed)
    └── <model_slug>/
        └── <exp_id>/                # e.g. 2026-04-24_0340_83eab3/
            ├── config.json          # single source of truth for the run
            ├── env.json             # env snapshot at start
            ├── prompt.md            # prompt actually used (hashed in config)
            ├── errors.jsonl         # one JSONL row per failed trajectory
            ├── analysis/            # aggregates written by analyze scripts
            ├── results/             # per-trajectory verdict JSONs
            ├── stdout/, stderr/     # per-trajectory judge subprocess stdio
            └── sandbox/             # identical structure to .meta/template/sandbox/
```

Active harnesses currently under `experiments/`: `claude/`, `codex/`, `gemini/`, `openhands/`, `opencode/`, `pi/`.

## What is committed vs. not

### Committed (tracked by git)

- `experiments/.gitignore`, `experiments/README.md`
- `.meta/exp_ids.jsonl` (~80 KB, one JSONL row per run, including archived)
- `.meta/template/` skeleton: `AGENTS.md`, symlinks, empty `errors.jsonl`, empty result dirs with `.gitkeep`, the `data/media/images` symlink, and the placeholder `prompt.md`
- `.meta/template/sandbox/agenthorizon_json/` and `.meta/template/sandbox/agenthorizon_md/` (1,700 files each, ~150 MB raw / ~20 MB packed). Committed so that a fresh clone is self-contained without needing `data/standard/agenthorizon.jsonl` to be rendered first.

Enforced by `experiments/.gitignore`:

```
*
!.gitignore
!README.md
!.meta/
!.meta/**
```

That is: ignore everything, then re-allow the two top-level docs and the entire `.meta/` tree. No other whitelists.

### Not committed

1. **Live per-run dirs** (`experiments/<harness>/<model_slug>/<exp_id>/`). Each is ~0.5 to 2 GB between `results/`, `stdout/`, `stderr/`, and its sandbox copy. Reproducible from `config.json` + prompt hash; the full mirror lives on EAI.
2. **Everything under `experiments/.archive/`**. Retired cohorts of runs (v3, v4, ...). The directory starts with a dot, so it is gitignored by the top-level `*` rule with no exception. `.archive/` exists only on local disk and on EAI.

## `.meta/template/` in detail

`.meta/template/` is the skeleton that `provision_exp_dir()` in `scripts/evaluate_trajectories.py` rsync's into every new `<exp_dir>/`. Think of it as the prototype for one run.

| Path | Purpose | Committed? |
|---|---|---|
| `errors.jsonl` | Fresh empty file; the harness appends one JSONL row per failed trajectory. | yes (empty) |
| `analysis/` | Destination for analysis JSONs written by `scripts/analyze_results.py`. | yes (empty + `.gitkeep`) |
| `results/` | Destination for per-trajectory verdict JSONs (`<uuid>.json`), one per judged trajectory. | yes (empty + `.gitkeep`) |
| `stdout/`, `stderr/` | Per-trajectory captured stdio (`<uuid>.txt`) from the judge subprocess. | yes (empty + `.gitkeep`) |
| `sandbox/` | CWD the judge subprocess runs in. See next section. | yes (everything; see below) |

`errors.jsonl`, `stdout/`, `stderr/`, `results/`, `analysis/` exist only so that the rsync produces the right shape. They do not hold data in the template itself.

## `.meta/template/sandbox/` in detail

This is the working directory the judge sees when it runs. Its layout matches what `AGENTS.md` tells the judge to expect.

| Path | Purpose | Committed? |
|---|---|---|
| `AGENTS.md` | Judge-framework prompt (mistake taxonomy, verdict rules). Read by agentic harnesses. | yes |
| `CLAUDE.md`, `GEMINI.md` | Symlinks to `AGENTS.md`. Some harnesses look for tool-specific names. | yes (symlinks) |
| `prompt.md` | 424-byte placeholder. Each run overwrites this with the real prompt (and also writes a copy to `<exp_dir>/prompt.md`). The prompt hash lands in `config.json`. | yes (placeholder) |
| `.gitkeep` | Forces the dir to exist in git. | yes |
| `data/media/images` | **Symlink** -> `../../../../../data/media/images`, i.e. `data/media/images/` at the repo root. Screenshots are referenced from the markdown as `./data/media/images/<id>/step_N.png`. Not copied, not large. | yes (symlink) |
| `agenthorizon_json/<uuid>.json` | 1,700 structured trajectories, one per file. The `task` + `steps[]` form the judge's structured view. | yes |
| `agenthorizon_md/<uuid>.md` | 1,700 markdown trajectories, one per file, with inline screenshot links. The judge reads these. | yes |

The per-trajectory files are locked read-only (`444` on files, `555` on dirs) in provisioned run dirs so the judge can't mutate them mid-eval. The template copy is read/write.

If you ever need to rebuild the per-trajectory files (from a schema change, or a fresh `data/standard/agenthorizon.jsonl`), run:

```bash
uv run python scripts/render_trajectories.py \
    data/standard/agenthorizon.jsonl \
    experiments/.meta/template/sandbox/agenthorizon \
    --format both
# writes both agenthorizon_md/ and agenthorizon_json/
```

Do NOT pass `--no-images`: the judge reads screenshots as part of evaluation.

## How to obtain the non-committed data

### On `nlp-cpu-1` (where this repo usually lives)

The non-committed data is already present on disk at `/home/nlp/users/xlu41/dev/cuarm/experiments/`, including every live per-run dir and the entire `.archive/`. If you cloned the repo fresh elsewhere on this host, just re-run evaluations; do not copy from another user's home.

### From EAI: per-run `<exp_dir>/` and the `.archive/`

Per-run `<exp_dir>/` contents and archived cohorts are stored in a dedicated EAI data object, separate from the `agenthorizon_data` / `agenthorizon_media` / `agenthorizon_results` objects.

> **`scripts/eai_pull.sh` does NOT fetch anything under `experiments/`.** It only pulls `data/`, `data/media/`, and `results/`. The experiments mirror uses `snow.xinghanlu.agenthorizon_experiments` and is populated by `scripts/export_experiments.sh`.

To list what's in the registry:

```bash
cat experiments/.meta/exp_ids.jsonl | jq -r '.exp_id + "  " + .path'
```

To pull one run (live or archived):

```bash
eai data pull snow.xinghanlu.agenthorizon_experiments experiments/<harness>/<slug>/<exp_id>
# for archived:
eai data pull snow.xinghanlu.agenthorizon_experiments experiments/.archive/v3/<harness>/<slug>/<exp_id>
```

To push new or archived runs back up (requires write access to `snow.xinghanlu`):

```bash
./scripts/export_experiments.sh --exp-id 2026-04-24_0340_83eab3
./scripts/export_experiments.sh --since 2026-04-20
./scripts/export_experiments.sh                            # everything
./scripts/export_experiments.sh --registry-only            # just .meta/
```

## Registry: `.meta/exp_ids.jsonl`

Append-only log, one JSONL row per provisioning call. Fields:

- `exp_id`, `harness`, `model`, `model_slug`, `platform`
- `prompt_hash`, `prompt_file`
- `started`, `completed`
- `path` (repo-relative; points under `.archive/vN/` once archived)
- `notes`, `effort`, `uses_console_key`, `workers`
- `counts`: `{total, ok, skip, error}` (filled on completion)
- `archived`, `archived_at`, `archive_reason` (added when a run is moved into `.archive/`)

This file is the authoritative index. Resume logic in `evaluate_trajectories.py` keys off `(harness, model_slug, exp_id)` and reuses the existing `config.json` instead of rewriting it. Unlike the per-run dirs themselves, the registry is committed, so every archived run remains discoverable from git alone.
