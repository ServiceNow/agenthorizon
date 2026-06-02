#!/usr/bin/env python3
"""Approach B: Two-Stage Filtering -- select important steps, full-res screenshots.

Stage 1: A fast/cheap model reads the text-only trajectory and selects the
top-K most important steps (the ones most likely to reveal whether the
task was completed).

Stage 2 (done at evaluation time): The judge model receives the instruction
plus only the selected steps with full-resolution screenshots.

Input:
    - Trajectory JSONL (data/standard/agenthorizon.jsonl)

Output:
    - One JSON file per trajectory containing the filtered messages payload:
      {
        "trajectory_id": str,
        "messages": [...],  # ready for judge API call (with full-res images)
        "selected_steps": [int, ...],  # indices of selected steps
        "total_steps": int,
        "token_estimate": int,
        "num_screenshots": int,
        "stage1_model": str,
        "stage1_response": str  # raw selection response for debugging
      }

Usage:
    # Stage 1: select important steps (uses cheap model)
    uv run python llm_judges/preprocess_filter.py \
        --input data/standard/agenthorizon.jsonl \
        --output-dir data/preprocessed/filter \
        --stage1-provider openrouter \
        --stage1-model qwen/qwen3.5-9b \
        --max-tokens 60000 \
        --top-k 15

    # Test with limit
    uv run python llm_judges/preprocess_filter.py \
        --input data/standard/agenthorizon.jsonl \
        --output-dir data/preprocessed/filter \
        --stage1-provider openrouter \
        --stage1-model qwen/qwen3.5-9b \
        --limit 3

    # Skip Stage 1 (use heuristic: first, last, and evenly spaced steps)
    uv run python llm_judges/preprocess_filter.py \
        --input data/standard/agenthorizon.jsonl \
        --output-dir data/preprocessed/filter \
        --heuristic \
        --top-k 15
"""

import argparse
import asyncio
import functools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_judges.utils import (
    build_instruction_text,
    estimate_image_tokens,
    estimate_text_tokens,
    format_step_text,
    get_screenshot_path,
    load_api_key,
    load_image_as_base64,
    load_local_image_map,
    load_trajectories,
)

print = functools.partial(print, flush=True)

# Evaluation prompt -- AgentHorizon judge framework, 4-field output schema
EVAL_PROMPT = (
    "You are evaluating a trajectory of an agent performing a computer task. "
    "The trajectory includes:\n"
    "- A Goal describing what the agent was supposed to accomplish\n"
    "- A sequence of Steps showing the actions taken (clicks, typing, key presses, etc.)\n\n"
    "## Judging framework\n\n"
    "A task has three layers:\n"
    "- **Core Objective** (What + Who): Primary Action + Primary Entity. Changing it -> Critical Mistake.\n"
    "- **Parameters** (When, Which, Where): Scope and data points. Changing them -> Bad Side Effect or Misunderstanding.\n"
    "- **Constraints** (Format, Style): How it looks. Changing them -> Misunderstanding.\n\n"
    "## Mistake types\n"
    "- **Critical Mistake** -- core objective not achieved; user must restart.\n"
    "- **Bad Side Effect** -- goal achieved but with unwanted baggage to delete (>5 min cleanup or external cost).\n"
    "- **Misunderstanding of the Instruction** -- agent took a reasonable action but misinterpreted a detail (<1 min fix).\n\n"
    "## Verdict rule\n"
    "- success = true when all three layers are satisfied.\n"
    "- success = false when any mistake type applies. A Misunderstanding is still a failure.\n\n"
    "## Output\n\n"
    "Be decisive. Respond with a JSON object only:\n"
    "{\n"
    '  "success": true | false,\n'
    '  "reasoning": "1-3 sentences citing which layer (core objective, parameters, or constraints) was or was not satisfied.",\n'
    '  "confidence": "low" | "medium" | "high",\n'
    '  "mistake_type": null when success=true, else one of: "Critical Mistake", "Bad Side Effect", "Misunderstanding of the Instruction"\n'
    "}\n\n"
    "Notes: confidence is evidence strength, not verdict. Low confidence != failure. mistake_type is null on success."
)

# Stage 1 prompt: ask the model to select the most important steps
STAGE1_PROMPT = """\
You are analyzing a trajectory of a computer-use agent to identify the most important steps for evaluating whether the task was completed.

Below is the task instruction and a list of all steps (actions only, no screenshots). Your job is to select the {top_k} most important step indices. Focus on:
1. Steps that directly accomplish parts of the task goal
2. The final few steps (which show the end state)
3. Steps where the agent interacts with key UI elements mentioned in the goal
4. Steps that might reveal errors or wrong actions

Respond with ONLY a JSON array of step indices (integers), e.g.:
[0, 5, 12, 45, 67, 99, 130, 131, 132, 133, 134, 135]

Select exactly {top_k} steps. Order them by step index (ascending)."""


def select_steps_heuristic(n_steps: int, top_k: int) -> list[int]:
    """Heuristic step selection: first, last, and evenly spaced.

    Guarantees step 0 (initial state) and step n-1 (final state) are
    included, with the remaining budget distributed evenly.
    """
    if n_steps <= top_k:
        return list(range(n_steps))

    # Always include first and last
    selected = {0, n_steps - 1}
    # Also include last 3 steps (critical for completion assessment)
    for i in range(max(0, n_steps - 3), n_steps):
        selected.add(i)

    # Fill remaining budget with evenly spaced steps
    remaining = top_k - len(selected)
    if remaining > 0:
        # Evenly spaced from the middle
        step_size = (n_steps - 1) / (remaining + 1)
        for j in range(1, remaining + 1):
            idx = int(j * step_size)
            selected.add(idx)

    # If we still have room, add more near key transitions
    result = sorted(selected)
    return result[:top_k]


async def select_steps_llm(
    trajectory: dict,
    top_k: int,
    provider: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> tuple[list[int], str]:
    """Use an LLM to select the most important steps (Stage 1).

    Returns (selected_indices, raw_response).
    """
    instruction = build_instruction_text(trajectory)
    steps = trajectory.get("steps", [])

    # Build text-only trajectory for Stage 1
    step_lines = []
    for i, step in enumerate(steps):
        step_lines.append(format_step_text(step, i))

    trajectory_text = (
        f"## Goal\n\n{instruction}\n\n"
        f"## Steps\n\nTotal steps: **{len(steps)}**\n\n"
        + "\n\n".join(step_lines)
    )

    prompt = STAGE1_PROMPT.format(top_k=top_k)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": trajectory_text},
    ]

    async with semaphore:
        if provider == "openrouter":
            response_text = await _call_openrouter(messages, model, api_key)
        elif provider == "gemini":
            response_text = await _call_gemini(messages, model, api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Parse the response
    response_text = response_text.strip()
    # Strip markdown fences
    response_text = re.sub(r"^```(?:json)?\s*\n?", "", response_text)
    response_text = re.sub(r"\n?```\s*$", "", response_text)
    response_text = response_text.strip()

    try:
        indices = json.loads(response_text)
        if isinstance(indices, list):
            # Validate indices
            valid = [i for i in indices if isinstance(i, int) and 0 <= i < len(steps)]
            return sorted(valid[:top_k]), response_text
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract numbers from the response
    numbers = re.findall(r'\b(\d+)\b', response_text)
    valid = [int(n) for n in numbers if 0 <= int(n) < len(steps)]
    if valid:
        return sorted(list(set(valid))[:top_k]), response_text

    # Last resort: heuristic
    return select_steps_heuristic(len(steps), top_k), response_text


async def _call_openrouter(messages: list[dict], model: str, api_key: str) -> str:
    """Call OpenRouter API. Returns response text."""
    import aiohttp

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            data = await resp.json()
            if resp.status != 200:
                raise RuntimeError(f"OpenRouter API error {resp.status}: {data}")
            return data["choices"][0]["message"]["content"] or ""


async def _call_gemini(messages: list[dict], model: str, api_key: str) -> str:
    """Call Gemini API via google-genai SDK. Returns response text."""
    from google import genai

    client = genai.Client(api_key=api_key)

    # Convert messages to Gemini format
    system_text = ""
    user_text = ""
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]

    combined = system_text + "\n\n" + user_text

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=model,
            contents=combined,
            config=genai.types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=512,
            ),
        ),
    )
    return response.text or ""


def build_filtered_payload(
    trajectory: dict,
    selected_steps: list[int],
    base_dir: Path,
    max_tokens: int,
    local_image_id: str | None = None,
) -> dict:
    """Build the Stage 2 payload with full-resolution screenshots for selected steps.

    Returns the messages payload ready for the judge API call.
    """
    instruction = build_instruction_text(trajectory)
    steps = trajectory.get("steps", [])
    selected_set = set(selected_steps)

    content_parts = []
    header = (
        f"## Goal\n\n{instruction}\n\n"
        f"## Steps\n\nTotal steps: **{len(steps)}** "
        f"(showing {len(selected_steps)} key steps with screenshots)\n"
    )
    content_parts.append({"type": "text", "text": header})
    total_tokens = estimate_text_tokens(EVAL_PROMPT) + estimate_text_tokens(header)
    n_screenshots = 0

    for i, step in enumerate(steps):
        if i in selected_set:
            step_text = format_step_text(step, i)
            content_parts.append({"type": "text", "text": step_text})
            total_tokens += estimate_text_tokens(step_text)

            # Add full-resolution screenshot
            screenshot_path = get_screenshot_path(step, base_dir, local_image_id=local_image_id)
            if screenshot_path is not None:
                try:
                    # Full resolution but JPEG compressed for size
                    b64, w, h = load_image_as_base64(screenshot_path, quality=80)
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    })
                    total_tokens += estimate_image_tokens(w, h)
                    n_screenshots += 1
                except Exception as e:
                    content_parts.append({
                        "type": "text",
                        "text": f"[Screenshot unavailable: {e}]",
                    })
        else:
            # Include text-only summary for non-selected steps
            step_text = format_step_text(step, i)
            content_parts.append({"type": "text", "text": step_text})
            total_tokens += estimate_text_tokens(step_text)

    # Check if we're over budget; if so, drop some non-selected step text
    if total_tokens > max_tokens:
        # Rebuild with only selected steps (skip text for non-selected)
        content_parts_slim = []
        content_parts_slim.append({"type": "text", "text": header})
        total_tokens = estimate_text_tokens(EVAL_PROMPT) + estimate_text_tokens(header)
        n_screenshots = 0

        for i, step in enumerate(steps):
            if i not in selected_set:
                continue
            step_text = format_step_text(step, i)
            content_parts_slim.append({"type": "text", "text": step_text})
            total_tokens += estimate_text_tokens(step_text)

            screenshot_path = get_screenshot_path(step, base_dir, local_image_id=local_image_id)
            if screenshot_path is not None:
                try:
                    b64, w, h = load_image_as_base64(screenshot_path, quality=80)
                    content_parts_slim.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                        },
                    })
                    total_tokens += estimate_image_tokens(w, h)
                    n_screenshots += 1
                except Exception:
                    pass
        content_parts = content_parts_slim

    messages = [
        {"role": "system", "content": EVAL_PROMPT},
        {"role": "user", "content": content_parts},
    ]

    return {
        "messages": messages,
        "token_estimate": total_tokens,
        "num_screenshots": n_screenshots,
    }


async def process_one(
    trajectory: dict,
    index: int,
    total: int,
    top_k: int,
    use_heuristic: bool,
    provider: str,
    model: str,
    api_key: str,
    base_dir: Path,
    output_dir: Path | None,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    local_image_id: str | None = None,
) -> dict:
    """Process a single trajectory through Stage 1 + build Stage 2 payload."""
    tid = trajectory["trajectory_id"]
    n_steps = len(trajectory.get("steps", []))

    # Check for existing output (resumability)
    if output_dir is not None:
        out_path = output_dir / f"{tid}.json"
        if out_path.exists():
            print(f"[{index}/{total}] [SKIP] {tid[:12]}...")
            return json.loads(out_path.read_text())

    # Stage 1: select steps
    if use_heuristic:
        selected = select_steps_heuristic(n_steps, top_k)
        stage1_response = "heuristic"
    else:
        selected, stage1_response = await select_steps_llm(
            trajectory, top_k, provider, model, api_key, semaphore
        )

    # Build Stage 2 payload with full-res screenshots
    payload = build_filtered_payload(
        trajectory, selected, base_dir, max_tokens, local_image_id=local_image_id,
    )

    result = {
        "trajectory_id": tid,
        "messages": payload["messages"],
        "selected_steps": selected,
        "total_steps": n_steps,
        "token_estimate": payload["token_estimate"],
        "num_screenshots": payload["num_screenshots"],
        "stage1_model": model if not use_heuristic else "heuristic",
        "stage1_response": stage1_response,
    }

    print(
        f"[{index}/{total}] {tid[:12]}... "
        f"steps={n_steps} selected={len(selected)} "
        f"screenshots={payload['num_screenshots']} "
        f"tokens~{payload['token_estimate']}"
    )

    if output_dir is not None:
        out_path = output_dir / f"{tid}.json"
        out_path.write_text(json.dumps(result, indent=None) + "\n")

    return result


async def run(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(args.input, limit=args.limit)
    print(f"Loaded {len(trajectories)} trajectories")
    image_map = load_local_image_map(args.labels)
    print(f"Loaded local_image_id mapping for {len(image_map)} trajectories from {args.labels}")
    print(f"Top-K: {args.top_k}, Max tokens: {args.max_tokens}")
    if args.heuristic:
        print(f"Step selection: heuristic (no API calls)")
    else:
        print(f"Stage 1 model: {args.stage1_provider}/{args.stage1_model}")
    print()

    api_key = ""
    if not args.heuristic:
        key_name = {
            "openrouter": "OPENROUTER_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }[args.stage1_provider]
        api_key = load_api_key(key_name)

    semaphore = asyncio.Semaphore(args.workers)
    total = len(trajectories)

    tasks = [
        process_one(
            trajectory=traj,
            index=i + 1,
            total=total,
            top_k=args.top_k,
            use_heuristic=args.heuristic,
            provider=args.stage1_provider,
            model=args.stage1_model,
            api_key=api_key,
            base_dir=base_dir,
            output_dir=output_dir,
            max_tokens=args.max_tokens,
            semaphore=semaphore,
            local_image_id=image_map.get(traj["trajectory_id"]),
        )
        for i, traj in enumerate(trajectories)
    ]

    results = await asyncio.gather(*tasks)

    # Summary
    token_counts = [r["token_estimate"] for r in results]
    screenshot_counts = [r["num_screenshots"] for r in results]
    print()
    print("=== Summary ===")
    if token_counts:
        import statistics
        print(f"Token estimates: mean={statistics.mean(token_counts):.0f}, "
              f"median={statistics.median(token_counts):.0f}, "
              f"max={max(token_counts):.0f}")
        print(f"Screenshots:    mean={statistics.mean(screenshot_counts):.0f}, "
              f"median={statistics.median(screenshot_counts):.0f}")
        over = sum(1 for t in token_counts if t > args.max_tokens)
        print(f"Over budget:    {over}/{len(token_counts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Approach B: Two-stage filtering -- select important steps."
    )
    parser.add_argument("--input", required=True, help="Path to trajectory JSONL")
    parser.add_argument(
        "--labels", default="data/standard/agenthorizon_labels.jsonl",
        help="Labels JSONL for trajectory_id -> original_id mapping (used only for "
             "screenshot path resolution; verdict fields are never read).",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for preprocessed files")
    parser.add_argument("--base-dir", default=".", help="Base dir for screenshot paths")
    parser.add_argument("--max-tokens", type=int, default=60000, help="Max token budget for Stage 2")
    parser.add_argument("--top-k", type=int, default=15, help="Number of steps to select")
    parser.add_argument("--heuristic", action="store_true", help="Use heuristic selection (no API calls)")
    parser.add_argument("--stage1-provider", default="openrouter", choices=["openrouter", "gemini"])
    parser.add_argument("--stage1-model", default="qwen/qwen3.5-9b", help="Model for Stage 1 filtering")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent Stage 1 API calls")
    parser.add_argument("--limit", type=int, default=None, help="Process first N trajectories only")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
