#!/usr/bin/env python3
"""Phase 2 evaluation runner: Direct LLM API calls without agent scaffolding.

Takes preprocessed trajectory payloads (from preprocess_compress.py,
preprocess_filter.py, or preprocess_summarize.py) and calls LLM APIs
directly to get judge verdicts. Outputs are compatible with the Phase 1
analysis pipeline (scripts/analyze_eval_results.py).

Supported providers:
    - openrouter: OpenRouter API (for Qwen, Gemma, etc.)
    - gemini: Google Gemini API (for Gemini Pro, Flash, etc.)

Output format (per trajectory):
    {
        "success": true/false,
        "reasoning": "...",
        "deliverable_id": "uuid",
        "_meta": {
            "model": "...",
            "provider": "...",
            "usage": {...},
            "approach": "compress|filter|summarize",
            "preprocessing": {...}
        }
    }

Usage:
    # Approach A (compress) + Qwen 3.5 397B
    uv run python llm_judges/evaluate.py \
        --preprocessed-dir data/preprocessed/compress \
        --output-dir results/eval_agenthorizon_llmjudge_compress_qwen397b \
        --provider openrouter \
        --model qwen/qwen3.5-397b-a17b \
        --workers 4

    # Approach A (compress) + Gemini Pro
    uv run python llm_judges/evaluate.py \
        --preprocessed-dir data/preprocessed/compress \
        --output-dir results/eval_agenthorizon_llmjudge_compress_gemini_pro \
        --provider gemini \
        --model gemini-3.1-pro \
        --workers 4

    # Test with limit
    uv run python llm_judges/evaluate.py \
        --preprocessed-dir data/preprocessed/compress \
        --output-dir results/eval_agenthorizon_llmjudge_test \
        --provider openrouter \
        --model qwen/qwen3.5-9b \
        --limit 5

    # Dry run
    uv run python llm_judges/evaluate.py \
        --preprocessed-dir data/preprocessed/compress \
        --output-dir results/eval_agenthorizon_llmjudge_test \
        --provider openrouter \
        --model qwen/qwen3.5-9b \
        --dry-run --limit 10
"""

import argparse
import asyncio
import functools
import json
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_judges.utils import load_api_key, parse_judge_response, save_result

print = functools.partial(print, flush=True)


async def call_openai_compatible(
    messages: list[dict],
    model: str,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    max_tokens: int = 2048,
    temperature: float = 0,
    reasoning_effort: str | None = None,
    provider_label: str = "openrouter",
    enable_thinking: bool | None = None,
) -> tuple[str, dict]:
    """Call any OpenAI-compatible /chat/completions endpoint.

    Defaults to OpenRouter. For a self-hosted vLLM endpoint, pass
    base_url="http://nlp-gpu-2.cs.mcgill.ca:18000/v1" and any non-empty
    api_key (vLLM ignores it unless --api-key is set on the server).

    Returns (response_text, metadata). Metadata includes `thinking` (and
    `thinking_details`) when the model produces internal reasoning, so we
    can preserve it for later analysis.
    """
    import aiohttp

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Reasoning toggle, forwarded by OpenRouter and ignored by plain OpenAI-
    # compatible servers (vLLM accepts it on certain models).
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}

    # Qwen 3+ chat template supports `enable_thinking`. vLLM exposes this via
    # `chat_template_kwargs`. Disabling cuts the (potentially long) thinking
    # trace so the verdict fits in our max_tokens budget and is comparable to
    # non-reasoning models like Gemini Flash Lite.
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    url = base_url.rstrip("/") + "/chat/completions"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            data = await resp.json()

            if resp.status == 429:
                # Rate limited
                retry_after = data.get("error", {}).get("metadata", {}).get("retry_after")
                raise RateLimitError(
                    f"Rate limited (429)", retry_after=retry_after or 60
                )

            if resp.status != 200:
                raise RuntimeError(
                    f"{provider_label} API error {resp.status}: "
                    f"{json.dumps(data)[:500]}"
                )

            msg = data["choices"][0]["message"]
            response_text = msg.get("content") or ""
            # Preserve any internal reasoning the model emitted. Many Qwen/GLM/
            # Gemini thinking models return `message.reasoning` alongside
            # `content`; keeping this is essential training data.
            thinking = msg.get("reasoning")
            thinking_details = msg.get("reasoning_details")
            meta = {
                "model": data.get("model"),
                "usage": data.get("usage"),
                "provider": provider_label,
                "base_url": base_url,
                "thinking": thinking,
                "thinking_details": thinking_details,
                "reasoning_effort_requested": reasoning_effort,
            }
            return response_text, meta


# Backwards-compat alias (other code in the file calls this name).
call_openrouter = call_openai_compatible


async def call_gemini(
    messages: list[dict],
    model: str,
    api_key: str,
    max_tokens: int = 1024,
    temperature: float = 0,
    thinking_budget: int | None = None,
) -> tuple[str, dict]:
    """Call Gemini API via google-genai SDK. Returns (response_text, metadata).

    When thinking_budget is set (>0), Gemini returns thought summaries that we
    capture into metadata. Set to -1 for dynamic/unlimited budget, or a positive
    int for a token cap. 0 / None disables thinking.
    """
    import base64
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Convert OpenAI-format messages to Gemini format
    system_text = ""
    parts = []

    for msg in messages:
        if msg["role"] == "system":
            if isinstance(msg["content"], str):
                system_text = msg["content"]
            continue

        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item["text"]))
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if url.startswith("data:image/jpeg;base64,"):
                        img_bytes = base64.b64decode(url.split(",", 1)[1])
                        parts.append(types.Part.from_bytes(
                            data=img_bytes, mime_type="image/jpeg"
                        ))
                    elif url.startswith("data:image/png;base64,"):
                        img_bytes = base64.b64decode(url.split(",", 1)[1])
                        parts.append(types.Part.from_bytes(
                            data=img_bytes, mime_type="image/png"
                        ))

    # Prepend system text to user content
    if system_text:
        parts.insert(0, types.Part.from_text(text=system_text + "\n\n"))

    config_kwargs = dict(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    # Enable thought summaries when a thinking budget is configured.
    if thinking_budget is not None and thinking_budget != 0:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=thinking_budget,
        )
    config = types.GenerateContentConfig(**config_kwargs)

    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model,
                contents=[types.Content(parts=parts)],
                config=config,
            ),
        )
    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate" in error_str or "quota" in error_str:
            raise RateLimitError(f"Gemini rate limit: {e}", retry_after=60)
        raise

    # Separate thought parts from answer parts. Gemini marks thought parts with
    # `thought=True` when ThinkingConfig.include_thoughts is enabled.
    answer_chunks: list[str] = []
    thought_chunks: list[str] = []
    try:
        for cand in response.candidates or []:
            for part in (cand.content.parts or []):
                text = getattr(part, "text", None)
                if not text:
                    continue
                if getattr(part, "thought", False):
                    thought_chunks.append(text)
                else:
                    answer_chunks.append(text)
    except Exception:
        # Best-effort; if structure differs, fall back to response.text
        pass

    response_text = "".join(answer_chunks) or (response.text or "")
    thinking = "".join(thought_chunks) or None

    meta = {
        "model": model,
        "provider": "gemini",
        "thinking": thinking,
        "thinking_budget_requested": thinking_budget,
    }
    # Extract usage if available
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        meta["usage"] = {
            "prompt_tokens": getattr(um, "prompt_token_count", None),
            "completion_tokens": getattr(um, "candidates_token_count", None),
            "thoughts_tokens": getattr(um, "thoughts_token_count", None),
            "total_tokens": getattr(um, "total_token_count", None),
        }

    return response_text, meta


# JSON Schema enforcing the 4-field judge output. Passed to `codex exec
# --output-schema` so GPT-5.x via Codex CLI is constrained to emit the same
# verdict shape as the OpenAI / Gemini / vLLM providers.
_CODEX_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "success": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "mistake_type": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "string",
                    "enum": [
                        "Critical Mistake",
                        "Bad Side Effect",
                        "Misunderstanding of the Instruction",
                    ],
                },
            ]
        },
    },
    "required": ["success", "reasoning", "confidence", "mistake_type"],
    "additionalProperties": False,
}


async def call_codex(
    messages: list[dict],
    model: str,
    reasoning_effort: str | None = None,
    platform: str = "chatgpt_subscription",
    timeout_s: int = 1200,
) -> tuple[str, dict]:
    """Call GPT-5.x via the Codex CLI in non-interactive mode.

    The codex CLI shells out to OpenAI's chat completions under the hood, but
    auth comes from `~/.codex/auth.json` (ChatGPT Pro/Plus subscription) and
    images are passed as file paths via repeated --image flags rather than
    inline base64. So we:
        1. Decode every base64 image in `messages` to a temp PNG.
        2. Concatenate system + user text (in order; images are interleaved).
        3. Run `codex exec ... --image f1 --image f2 ... --output-schema ...`.
        4. Read the last-message file and parse it as JSON.

    `--output-schema` constrains codex to emit JSON matching `_CODEX_OUTPUT_SCHEMA`,
    so the judge response shape is consistent with the other providers without
    needing to ask the model to obey JSON formatting in the prompt.

    Returns (response_text, metadata).
    """
    import base64
    import os
    import tempfile

    # Build the prompt text (system + user text, in order). Images are
    # collected separately and passed via --image; we keep their textual
    # context (the step header) inline in the same order.
    text_chunks: list[str] = []
    image_paths: list[str] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="llmjudge_codex_"))

    try:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_chunks.append(content)
                continue
            if not isinstance(content, list):
                continue
            for item in content:
                if item.get("type") == "text":
                    text_chunks.append(item["text"])
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        continue
                    header, b64 = url.split(",", 1)
                    ext = "jpg" if "jpeg" in header else "png"
                    img_path = tmpdir / f"img_{len(image_paths):03d}.{ext}"
                    img_path.write_bytes(base64.b64decode(b64))
                    image_paths.append(str(img_path))

        prompt_text = "\n\n".join(text_chunks)

        # Write the schema file (one per call; `tmpdir` is already isolated).
        schema_path = tmpdir / "schema.json"
        schema_path.write_text(json.dumps(_CODEX_OUTPUT_SCHEMA))

        # Output-last-message file.
        out_path = tmpdir / "output.txt"

        cmd: list[str] = [
            "codex", "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox", "read-only",
            "--output-schema", str(schema_path),
            "-o", str(out_path),
            "-m", model,
        ]
        # Provider override: chatgpt_subscription routes Codex's built-in
        # OpenAI provider via `~/.codex/auth.json` (the user's ChatGPT plan),
        # NOT OpenRouter. Same convention as scripts/evaluate_trajectories.py.
        if platform == "chatgpt_subscription":
            cmd.extend(["-c", 'model_provider = "openai"'])
        if reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort = "{reasoning_effort}"'])
        for ip in image_paths:
            cmd.extend(["--image", ip])
        # Prompt via stdin (`-` is the documented signal). Avoids arg-parser
        # confusion when there are many --image flags before the positional,
        # and side-steps any ARG_MAX issue on long trajectory text.
        cmd.append("-")

        # Spawn the codex process. Codex emits some progress chatter to stdout
        # but the final JSON answer lands in `out_path` thanks to -o.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=prompt_text.encode()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"codex exec timed out after {timeout_s}s")

        if proc.returncode != 0:
            err_msg = (stderr_bytes or b"").decode(errors="replace")[:500]
            raise RuntimeError(f"codex exec exit {proc.returncode}: {err_msg}")

        if not out_path.exists():
            raise RuntimeError("codex exec produced no output-last-message file")
        response_text = out_path.read_text().strip()

        meta = {
            "model": model,
            "provider": "codex",
            "platform": platform,
            "reasoning_effort_requested": reasoning_effort,
            "n_images": len(image_paths),
        }
        return response_text, meta
    finally:
        # Best-effort cleanup of temp dir + files.
        try:
            for p in tmpdir.iterdir():
                p.unlink(missing_ok=True)
            tmpdir.rmdir()
        except OSError:
            pass


class RateLimitError(Exception):
    """Raised when an API returns a rate limit error."""
    def __init__(self, message: str, retry_after: float = 60):
        super().__init__(message)
        self.retry_after = retry_after


async def evaluate_one(
    preprocessed_path: Path,
    output_dir: Path,
    provider: str,
    model: str,
    api_key: str,
    approach: str,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
    dry_run: bool,
    max_retries: int = 3,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    base_url: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Evaluate a single preprocessed trajectory. Returns 'ok', 'skip', or 'error'."""
    # Load preprocessed payload
    data = json.loads(preprocessed_path.read_text())
    trajectory_id = data["trajectory_id"]
    messages = data["messages"]

    # Resumability
    json_path = output_dir / f"{trajectory_id}.json"
    if json_path.exists():
        print(f"[{index}/{total}] [SKIP] {trajectory_id[:12]}...")
        return "skip"

    if dry_run:
        tokens = data.get("token_estimate", 0)
        print(f"[{index}/{total}] [DRY-RUN] {trajectory_id[:12]}... tokens~{tokens}")
        return "skip"

    result = None
    response_text = ""
    meta = {}

    for attempt in range(max_retries):
        rate_limit_retries = 0
        api_success = False

        while rate_limit_retries < 5:
            wait = 60  # default wait for rate limits
            api_error = None

            async with semaphore:
                label = f"[{index}/{total}]"
                if attempt > 0:
                    label += f" [RETRY {attempt}]"
                print(f"{label} [RUN]  {trajectory_id[:12]}...")

                try:
                    if provider == "openrouter":
                        response_text, meta = await call_openai_compatible(
                            messages, model, api_key,
                            base_url=base_url or "https://openrouter.ai/api/v1",
                            reasoning_effort=reasoning_effort,
                            provider_label="openrouter",
                        )
                    elif provider == "vllm":
                        if not base_url:
                            raise ValueError("--base-url is required when --provider vllm")
                        response_text, meta = await call_openai_compatible(
                            messages, model, api_key or "EMPTY",
                            base_url=base_url,
                            reasoning_effort=reasoning_effort,
                            provider_label="vllm",
                            enable_thinking=enable_thinking,
                            max_tokens=(16384 if enable_thinking else 2048),
                        )
                    elif provider == "gemini":
                        response_text, meta = await call_gemini(
                            messages, model, api_key,
                            thinking_budget=thinking_budget,
                            max_tokens=(16384 if thinking_budget not in (None, 0) else 1024),
                        )
                    elif provider == "codex":
                        # GPT-5.x via Codex CLI (chatgpt_subscription auth).
                        # api_key is unused (codex reads ~/.codex/auth.json).
                        response_text, meta = await call_codex(
                            messages, model,
                            reasoning_effort=reasoning_effort,
                        )
                    else:
                        raise ValueError(f"Unknown provider: {provider}")
                    api_success = True
                    break  # success, exit rate-limit loop
                except RateLimitError as e:
                    rate_limit_retries += 1
                    wait = e.retry_after
                    print(
                        f"[{index}/{total}] [RATE_LIMIT] {trajectory_id[:12]}... "
                        f"waiting {wait}s (retry {rate_limit_retries})"
                    )
                except Exception as e:
                    api_error = e
                    print(
                        f"[{index}/{total}] [API_ERROR] {trajectory_id[:12]}... "
                        f"{type(e).__name__}: {str(e)[:200]}"
                    )
                    break  # exit rate-limit loop, let outer retry handle it

            if api_error is not None:
                break  # exit rate-limit while loop on non-rate-limit error

            # Sleep outside semaphore for rate limits
            await asyncio.sleep(wait)
        else:
            # while condition became false: too many rate limit retries
            print(f"[{index}/{total}] [ERROR] {trajectory_id[:12]}... rate limited too many times")
            return "error"

        if not api_success:
            if attempt < max_retries - 1:
                print(f"[{index}/{total}] [RETRY] {trajectory_id[:12]}... attempt {attempt + 1}/{max_retries}")
                continue
            else:
                print(f"[{index}/{total}] [ERROR] {trajectory_id[:12]}... all retries exhausted")
                return "error"

        # Parse response
        parsed = parse_judge_response(response_text)
        if "success" in parsed:
            result = parsed
            break
        elif attempt < max_retries - 1:
            print(
                f"[{index}/{total}] [PARSE_FAIL] {trajectory_id[:12]}... "
                f"retrying ({attempt + 1}/{max_retries})"
            )
        else:
            result = parsed  # will have raw_response

    if result is None:
        return "error"

    # Save result
    meta["approach"] = approach
    meta["preprocessing"] = {
        "token_estimate": data.get("token_estimate"),
        "num_screenshots": data.get("num_screenshots", data.get("num_screenshots_summarized", 0)),
        "image_resolution": data.get("image_resolution"),
        "selected_steps": data.get("selected_steps"),
    }
    save_result(result, trajectory_id, output_dir, meta=meta)

    status = "ok" if "success" in result else "error"
    print(f"[{index}/{total}] [{status.upper()}]   {trajectory_id[:12]}...")
    return status


async def run(args: argparse.Namespace) -> None:
    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find preprocessed files
    files = sorted(preprocessed_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if not files:
        print(f"No preprocessed files found in {preprocessed_dir}")
        return

    if args.limit is not None:
        files = files[:args.limit]

    # Detect approach from directory name
    approach = "unknown"
    dir_name = preprocessed_dir.name.lower()
    if "compress" in dir_name:
        approach = "compress"
    elif "filter" in dir_name:
        approach = "filter"
    elif "summar" in dir_name:
        approach = "summarize"

    # Load API key (skipped in dry-run mode, where no API calls are made)
    # vLLM provider doesn't need a real key -- the local server doesn't
    # authenticate by default; we just send a dummy bearer to keep the OpenAI
    # client happy.
    api_key = ""
    if not args.dry_run:
        if args.provider in ("openrouter", "gemini"):
            key_name = {
                "openrouter": "OPENROUTER_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }[args.provider]
            api_key = load_api_key(key_name)
        elif args.provider == "vllm":
            if not args.base_url:
                raise ValueError("--base-url is required when --provider vllm")
            # Self-hosted vLLM cluster: load bearer from ~/.vllm/.env if present;
            # fall back to a placeholder if the server doesn't authenticate.
            try:
                api_key = load_api_key("VLLM_BEARER_TOKEN")
            except ValueError:
                api_key = "EMPTY"
        elif args.provider == "codex":
            # Codex reads `~/.codex/auth.json` (ChatGPT subscription); no API
            # key needed in our process. Sanity-check that the auth file
            # exists so we fail fast instead of inside the subprocess.
            auth_p = Path.home() / ".codex" / "auth.json"
            if not auth_p.exists():
                raise ValueError(
                    f"--provider codex requires {auth_p} (run `codex login`)."
                )
            api_key = ""  # unused

    total = len(files)
    print(f"Evaluating {total} preprocessed trajectories from {preprocessed_dir}")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Approach: {approach}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {output_dir}")
    if args.dry_run:
        print(f"  *** DRY RUN ***")
    print()

    semaphore = asyncio.Semaphore(args.workers)
    start = time.time()

    tasks = [
        evaluate_one(
            preprocessed_path=f,
            output_dir=output_dir,
            provider=args.provider,
            model=args.model,
            api_key=api_key,
            approach=approach,
            semaphore=semaphore,
            index=i + 1,
            total=total,
            dry_run=args.dry_run,
            max_retries=args.max_retries,
            reasoning_effort=args.reasoning_effort,
            thinking_budget=args.thinking_budget,
            base_url=args.base_url,
            enable_thinking=args.enable_thinking,
        )
        for i, f in enumerate(files)
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    counts = {"total": total, "ok": 0, "skip": 0, "error": 0}
    for r in results:
        counts[r] += 1

    print()
    print(f"Done in {elapsed:.1f}s -- OK: {counts['ok']}, Skip: {counts['skip']}, Error: {counts['error']}")

    if not args.dry_run:
        summary = {
            "preprocessed_dir": str(preprocessed_dir),
            "provider": args.provider,
            "model": args.model,
            "approach": approach,
            "workers": args.workers,
            "elapsed_seconds": round(elapsed, 1),
            **counts,
        }
        summary_path = output_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Summary written to {summary_path}")

    # Print analysis command for convenience
    print()
    print("To analyze results:")
    print(
        f"  uv run python scripts/analyze_eval_results.py \\\n"
        f"    --results-dir {output_dir} \\\n"
        f"    --labels data/standard/agenthorizon_labels.jsonl \\\n"
        f"    --trajectories data/standard/agenthorizon.jsonl \\\n"
        f"    --output results/analysis/eval_agenthorizon_llmjudge_{approach}_{args.model.split('/')[-1].replace('.', '_')}.json"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 evaluation runner: direct LLM API calls."
    )
    parser.add_argument(
        "--preprocessed-dir", required=True,
        help="Directory with preprocessed trajectory JSON files"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write judge result JSON files"
    )
    parser.add_argument(
        "--provider", required=True, choices=["openrouter", "gemini", "vllm", "codex"],
        help="API provider. 'vllm' targets a self-hosted OpenAI-compatible "
             "endpoint; pass its URL with --base-url. 'codex' shells out to "
             "the Codex CLI in non-interactive mode for GPT-5.x via "
             "chatgpt_subscription auth (~/.codex/auth.json)."
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name. For openrouter/gemini, the routed model id. For "
             "vllm, the value of --served-model-name on the server."
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible base URL ending in /v1. Required when "
             "--provider vllm (e.g., http://nlp-gpu-2.cs.mcgill.ca:18000/v1). "
             "Optional override for openrouter."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Max concurrent API calls (default: 4)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries on parse failure (default: 3)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate first N files"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print token estimates without making API calls"
    )
    parser.add_argument(
        "--reasoning-effort", default=None,
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning/thinking effort. For --provider openrouter this sets "
             "the OpenRouter `reasoning.effort` field; for --provider codex "
             "this sets `model_reasoning_effort` via `-c`. Captured into "
             "result _meta.thinking when the provider returns it."
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=None,
        help="Gemini thinking token budget. Positive int caps thought tokens, "
             "-1 enables dynamic/unlimited, 0 or unset disables thinking. When "
             "set, thought summaries are captured into result _meta.thinking. "
             "Only applies to --provider gemini."
    )
    parser.add_argument(
        "--enable-thinking", action="store_true", default=False,
        help="Enable Qwen 3+ chat-template thinking mode on --provider vllm. "
             "Sets chat_template_kwargs.enable_thinking=true, raises max_tokens "
             "to 16384, and captures the thinking trace into _meta.thinking. "
             "Default off (faster, comparable to non-reasoning judges)."
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
