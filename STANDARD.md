# AgentHorizon: Trajectory Standard Format (v1.1)

This document defines the unified JSON format for storing AgentHorizon benchmark trajectories. The goal is to normalize data from different sources (human annotations, AgentSynth, future sources) into a single format for training and evaluation.

## Overview

A **trajectory** is a sequence of steps representing a computer-use task execution. Each step pairs a screenshot (observation) with an action taken on the screen. Trajectories are labeled as `positive` (correct execution) or `negative` (incorrect — the actions do not fulfill the task instruction).

**Important:** Labels are stored in a **separate file** to support blind evaluation. The trajectory file contains NO labels, NO identifiable IDs, and NO metadata that could reveal ground truth.

## File Organization

Trajectories are stored as JSONL files (one JSON object per line):

```
data/standard/human.jsonl           # Trajectories (UUIDs, shuffled, NO labels)
data/standard/human_labels.jsonl    # Ground-truth labels (separate file)
data/standard/agentsynth.jsonl      # AgentSynth trajectories
```

Each line is a self-contained trajectory JSON object. The trajectory and labels files have matching line counts and can be joined on `trajectory_id`.

---

## Top-Level Schema (Trajectory File)

```json
{
  "version": "1.0",
  "trajectory_id": "uuid-string",
  "source": {
    "name": "string",
    "original_file": "string"
  },
  "task": {
    "instruction": "string",
    "persona": "string (optional)",
    "category": "string (optional)",
    "subcategory": "string (optional)",
    "task_type": "string (optional)",
    "applications": ["string"]
  },
  "environment": {
    "os": "string",
    "screen_resolution": [1920, 1080]
  },
  "steps": [ ... ],
  "milestones": [ ... ]
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | yes | Schema version. Always `"1.0"`. |
| `trajectory_id` | string | yes | UUID v4. Opaque — reveals nothing about source or label. |
| `source.name` | string | yes | Origin dataset: `"human"`, `"agentsynth"`. |
| `source.original_file` | string | no | Path to the original source file, relative to repo root. |
| `task.instruction` | string | yes | The task instruction describing what the agent should accomplish. |
| `task.persona` | string | no | Agent persona / role description. |
| `task.category` | string | no | High-level task category (e.g., `"Tool Usage"`). |
| `task.subcategory` | string | no | Subcategory within the category. |
| `task.task_type` | string | no | Task type (e.g., `"DEVELOPMENT"`, `"PRODUCTIVITY"`). |
| `task.applications` | list[string] | no | Applications used in the trajectory. |
| `environment.os` | string | no | Operating system: `"linux"`, `"windows"`, `"macos"`. |
| `environment.screen_resolution` | list[int] | no | Screen resolution as `[width, height]`. |
| `steps` | list[Step] | yes | Ordered list of action-observation steps. |
| `milestones` | list[Milestone] | no | Key checkpoints in the trajectory. |

**Fields intentionally omitted from trajectory file (to prevent label leakage):**
- `label` — stored in labels file only
- `source.original_id` — could reveal positive/negative pairing
- `task.instruction_original` — reveals cross-assignment for negatives
- `metadata` (paired_task_id, mistake_type, etc.) — reveals pairing structure

---

## Labels File Schema

Each line in the labels JSONL maps a trajectory UUID to its ground truth:

```json
{
  "trajectory_id": "uuid-string",
  "label": "positive | negative",
  "original_id": "string",
  "trajectory_scenario": "string (optional)",
  "trajectory_type": "string (optional)",
  "paired_id": "string (optional)",
  "negative_source": "string (optional)",
  "mistake_type": "string (optional)"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trajectory_id` | string | yes | UUID matching the trajectory file. |
| `label` | string | yes | `"positive"` or `"negative"`. |
| `original_id` | string | yes | Deliverable ID from the source data. |
| `trajectory_scenario` | string | no | e.g., `"Happy Path"`. |
| `trajectory_type` | string | no | e.g., `"underspecified"`, `"fullyspecified"`. |
| `paired_id` | string | no | Deliverable ID of the paired task (for negatives). |
| `negative_source` | string | no | How the negative was constructed: `"parent_instruction_child_trajectory"` or `"child_instruction_parent_trajectory"`. |
| `mistake_type` | string | no | e.g., `"Critical Mistake"`, `"Bad Side Effect"`, `"Misunderstanding of the Instruction"`. |

**CRITICAL:** This file must NEVER be exposed to, read by, or referenced during judge evaluation. It is solely for post-hoc analysis.

---

## Step Schema

Each step represents one atomic action paired with the screenshot taken **before** the action was executed (i.e., the observation the agent sees before deciding what to do).

```json
{
  "step_id": 0,
  "screenshot": "string (optional)",
  "action": {
    "type": "string",
    "parameters": { }
  },
  "thought": "string (optional)",
  "action_description": "string (optional)",
  "timestamp_us": 0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_id` | int | yes | Zero-indexed step number. |
| `screenshot` | string | no | Path or URL to the screenshot image. Can be a relative file path, an absolute URL, or `null` if no screenshot is available. |
| `action` | Action | yes | The action taken at this step. |
| `thought` | string | no | Agent reasoning / chain-of-thought before this action. |
| `action_description` | string | no | Human-readable description of the action. |
| `timestamp_us` | int | no | Timestamp in microseconds from trajectory start. |

---

## Action Types

All actions follow the format `{"type": "<type>", "parameters": {...}}`. The table below lists every supported action type and its parameters.

### `click`

Mouse click at a screen coordinate.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `x` | int | yes | X coordinate in pixels. |
| `y` | int | yes | Y coordinate in pixels. |
| `button` | string | yes | `"left"` or `"right"`. |
| `num_clicks` | int | no | Number of clicks. Default `1`. Use `2` for double-click. |

### `type`

Type text via keyboard input.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | yes | The text to type. |

### `press`

Press one or more keys sequentially.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `keys` | list[string] | yes | Key names to press in sequence (e.g., `["Enter"]`, `["Tab"]`). |

### `hotkey`

Press multiple keys simultaneously (keyboard shortcut).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `keys` | list[string] | yes | Keys to press together (e.g., `["Ctrl", "c"]`). |

### `scroll`

Scroll at a screen position.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `x` | int | yes | X coordinate of scroll position. |
| `y` | int | yes | Y coordinate of scroll position. |
| `direction` | string | yes | `"up"` or `"down"`. |
| `amount` | int | yes | Scroll distance in pixels. |

### `drag`

Drag from one coordinate to another.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `start_x` | int | yes | Starting X coordinate. |
| `start_y` | int | yes | Starting Y coordinate. |
| `end_x` | int | yes | Ending X coordinate. |
| `end_y` | int | yes | Ending Y coordinate. |

### `move`

Move the mouse cursor to a position (no click).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `x` | int | yes | X coordinate. |
| `y` | int | yes | Y coordinate. |

### `key_down`

Press and hold a key (must be paired with `key_up`).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `key` | string | yes | Key name (e.g., `"Shift"`). |

### `key_up`

Release a held key.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `key` | string | yes | Key name (e.g., `"Shift"`). |

### `wait`

Wait/pause for a duration.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `duration_s` | float | yes | Duration in seconds. |

---

## Milestone Schema

```json
{
  "step_id": 8,
  "description": "string"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_id` | int | yes | The step at which this milestone is reached. |
| `description` | string | yes | Description of the subgoal achieved. |

---

## Source-Specific Mapping

### Human Trajectories (`delivery_*.json`)

| Source field | Standard field |
|---|---|
| `deliverable_id` | `source.original_id` |
| `messages[0].content` (user message) | `task.instruction` |
| `notes.os` | `environment.os` (normalized to lowercase; invalid values dropped) |
| `notes.task_type` | `task.task_type` |
| `notes.application_names` | `task.applications` |
| `notes.task_category_list[0].category` | `task.category` |
| `notes.task_category_list[0].subcategory` | `task.subcategory` |
| `notes.milestones` | `milestones` |
| `notes.trajectory_type` | `metadata.trajectory_type` (normalized) |
| `notes.trajectory_scenario` | `metadata.trajectory_scenario` |
| `notes.paired_task_id` | `metadata.paired_task_id` |
| `notes.mistake_type` | `metadata.mistake_type` |

**Action type mapping:**

| Human action type | Standard type | Notes |
|---|---|---|
| `click` | `click` | `text` → `button` (`"left-click"` → `"left"`), `numClicks` → `num_clicks` |
| `doubleClick` | `click` | Same as `click`, defaults `num_clicks=2` |
| `typing` | `type` | `text` → `text` |
| `press` | `press` | `text` → `keys` (wrapped in list) |
| `hotkey` | `hotkey` | `text` split on `+` → `keys` |
| `scroll` | `scroll` | `scrollX/scrollY` → `x/y`, `scrollDirection` → `direction`, `totalScrollDistance` → `amount` |
| `dragFromTo` | `drag` | `x/y` → `start_x/start_y`, `xEnd/yEnd` → `end_x/end_y` |
| `drag` | `drag` | `x/y` used as end coordinates (offset-style drag) |
| `keyDown` | `key_down` | `text` → `key` |
| `keyUp` | `key_up` | `text` → `key` |

**Parent-child negative pairing:**

The human delivery data contains 426 paired deliverables. A child deliverable (identified by having a `notes.paired_task_id`) has a slightly different instruction from its parent. For each pair, the converter emits:

1. A **positive** trajectory for each deliverable: its own instruction paired with its own execution trace.
2. **Negative (parent instruction + child trajectory)**: The parent's instruction paired with the child's execution trace. The mistake type is `notes.mistake_type_inst2_demo_1`.
3. **Negative (child instruction + parent trajectory)**: The child's instruction paired with the parent's execution trace. The mistake type is `notes.mistake_type_inst1_demo_2`.

This produces 852 positive + 852 negative = **1704 total trajectories**. All are assigned UUID IDs and shuffled. Negative metadata (mistake_type, paired_id, negative_source) is stored only in the labels file, never in the trajectory file.

### AgentSynth Positive Trajectories (`extracted/*.json`)

Each file contains a multi-step trajectory with parallel arrays. Steps are flattened: for step index `i` and sub-step index `j`, each `(thoughts[i][j], actions[i][j], commands[i][j], screenshots[i][j])` tuple becomes one standard step.

| Source field | Standard field |
|---|---|
| `task_id` (from filename or field) | `source.original_id` |
| `summary_task` | `task.instruction` |
| `persona` | `task.persona` |
| `commands[i][j]` (parsed) | `steps[k].action` |
| `thoughts[i][j]` | `steps[k].thought` |
| `actions[i][j]` | `steps[k].action_description` |
| `screenshots[i][j]` | `steps[k].screenshot` (saved as PNG file) |
| `done` | `metadata.done_flags` |
| `task_history` | `metadata.task_history` |
| `task_history_original` | `metadata.task_history_original` |
| `info_history` | `metadata.info_history` |
| `task_levels` | `metadata.task_levels` |

**PyAutoGUI command parsing:**

| PyAutoGUI command | Standard type | Parameter extraction |
|---|---|---|
| `pyautogui.click(x=N, y=N, button='...')` | `click` | Extract `x`, `y`, `button` |
| `pyautogui.doubleClick(x=N, y=N)` | `click` | Extract `x`, `y`; set `num_clicks=2` |
| `pyautogui.write('...')` | `type` | Extract text |
| `pyautogui.press(['key'])` or `pyautogui.press('key')` | `press` | Extract key(s) |
| `pyautogui.hotkey('k1', 'k2')` | `hotkey` | Extract keys |
| `pyautogui.scroll(N)` | `scroll` | Positive = up, negative = down; `amount = abs(N)` |
| `pyautogui.moveTo(x=N, y=N)` | `move` | Extract `x`, `y` |
| `pyautogui.dragTo(x=N, y=N)` | `drag` | Requires previous `moveTo` for `start_x/start_y` |
| `time.sleep(N)` | `wait` | `duration_s = N` |

### AgentSynth Negative Trajectories (`negative/*.json`)

Negative trajectories are **modified task instructions** that, when paired with the positive trajectory's execution trace, create a negative example. The processing works as follows:

1. Load the negative file to get `task_id` and modified `summary_task`.
2. Load the corresponding positive trajectory (same `task_id`) to get the execution steps.
3. Create a standard trajectory using the negative `summary_task` as `task.instruction`, the positive `summary_task` as `task.instruction_original`, and all steps from the positive trajectory, with `label` set to `"negative"`.

---

## Conventions

- **Coordinates** are in pixels, origin at top-left of the screen.
- **Key names** follow [PyAutoGUI key names](https://pyautogui.readthedocs.io/en/latest/keyboard.html#keyboard-keys) (e.g., `"Enter"`, `"Shift"`, `"Ctrl"`, `"Alt"`, `"Tab"`, `"Backspace"`).
- **Screenshots** can be stored as:
  - Relative file paths (e.g., `"screenshots/step_000.png"`)
  - HTTP(S) URLs (e.g., `"https://storage.googleapis.com/...png"`)
  - `null` if not available
- **OS names** are normalized to lowercase: `"linux"`, `"windows"`, `"macos"`.
- **Trajectory IDs** are UUID v4 strings (e.g., `"a1b2c3d4-e5f6-7890-abcd-ef1234567890"`). They are opaque and reveal nothing about the trajectory's source or label.
- **File naming:** Data files use fixed names without date suffixes (e.g., `human.jsonl`).

---

## Example (Trajectory File)

```json
{
  "version": "1.0",
  "trajectory_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "source": {
    "name": "human",
    "original_file": "data/new/final/final_delivery_batch.json"
  },
  "task": {
    "instruction": "Create a folder named 'Workspace' inside the Documents directory...",
    "category": "Tool Usage",
    "subcategory": "GUI Agent",
    "task_type": "DEVELOPMENT",
    "applications": ["NetBeans", "JupyterLab Notebook"]
  },
  "environment": {
    "os": "windows",
    "screen_resolution": null
  },
  "steps": [
    {
      "step_id": 0,
      "screenshot": "https://storage.googleapis.com/video-annotation-tool-new/screenshot_4fd794eb.png",
      "action": {
        "type": "hotkey",
        "parameters": {
          "keys": ["Windows", "e"]
        }
      },
      "thought": null,
      "action_description": null,
      "timestamp_us": 0
    },
    {
      "step_id": 1,
      "screenshot": "https://storage.googleapis.com/video-annotation-tool-new/screenshot_abc123.png",
      "action": {
        "type": "click",
        "parameters": {
          "x": 103,
          "y": 345,
          "button": "left",
          "num_clicks": 1
        }
      },
      "thought": null,
      "action_description": null,
      "timestamp_us": 244000
    }
  ],
  "milestones": [
    {
      "step_id": 8,
      "description": "Set up a local workspace by creating a folder 'Workspace' inside the Documents directory."
    }
  ]
}
```

## Example (Labels File)

```json
{
  "trajectory_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "label": "positive",
  "original_id": "000231601-231601-0000-000000231601",
  "trajectory_scenario": "Happy Path",
  "trajectory_type": "Underspecified"
}
```

```json
{
  "trajectory_id": "f9e8d7c6-b5a4-3210-fedc-ba9876543210",
  "label": "negative",
  "original_id": "000231601-231601-0000-000000231601",
  "paired_id": "000231665-231665-0000-000000231665",
  "negative_source": "parent_instruction_child_trajectory",
  "mistake_type": "Bad Side Effect"
}
```

---

## Version History

### v1.1 (2026-03-16) — Current

Changes from v1.0:
- **Labels separated:** `label` field removed from trajectory file; stored in a separate `*_labels_*.jsonl` file
- **UUID trajectory IDs:** `trajectory_id` is now a UUID v4 (was `{source}_{original_id}`)
- **No identifiable metadata in trajectories:** `source.original_id`, `task.instruction_original`, and `metadata` (paired_task_id, mistake_type, negative_source) removed from trajectory file to prevent label leakage
- **Shuffled output:** Positive and negative trajectories are intermixed
- **Separate labels file:** `human.jsonl` + `human_labels.jsonl`
- **Mistake type fields updated:** Human delivery now uses `mistake_type_inst1_demo_2` and `mistake_type_inst2_demo_1` (was single `mistake_type`)

### v1.0 (2026-03-06) — Archived

Original format. Labels embedded in trajectory file. Identifiable trajectory IDs (`human_231601`, `agentsynth_neg_54`). Single `human.jsonl` / `agentsynth.jsonl` without date suffix.

<details>
<summary>v1.0 Top-Level Schema (click to expand)</summary>

```json
{
  "version": "1.0",
  "trajectory_id": "{source}_{original_id}",
  "source": {
    "name": "string",
    "original_id": "string",
    "original_file": "string"
  },
  "task": {
    "instruction": "string",
    "instruction_original": "string (optional)",
    "persona": "string (optional)",
    "category": "string (optional)",
    "subcategory": "string (optional)",
    "task_type": "string (optional)",
    "applications": ["string"]
  },
  "environment": {
    "os": "string",
    "screen_resolution": [1920, 1080]
  },
  "label": "positive | negative",
  "steps": [ ... ],
  "milestones": [ ... ],
  "metadata": { }
}
```

File organization:
```
data/standard/human.jsonl         # Labels embedded, identifiable IDs
data/standard/agentsynth.jsonl    # Labels embedded, identifiable IDs
```

</details>
