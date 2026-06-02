#!/usr/bin/env python3
"""Convert a Claude Code session JSONL file into readable Markdown.

Parses a Claude Code session transcript (.jsonl) and produces a human-readable
Markdown document with user messages, assistant responses, tool calls/results,
and collapsible reasoning blocks.

Usage:
    uv run python scripts/jsonl_to_md.py <input.jsonl> [output.md]

If no output path is given, prints to stdout.
"""

import json
import sys
import re
from pathlib import Path

MAX_TOOL_OUTPUT_LEN = 2000  # truncate tool outputs longer than this


def strip_xml_tags(text: str) -> str:
    """Remove common system XML wrapper tags but keep inner content."""
    # Remove specific system tags but keep content
    for tag in [
        "local-command-caveat",
        "local-command-stdout",
        "command-name",
        "command-message",
        "command-args",
        "system-reminder",
    ]:
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
        text = re.sub(rf"<{tag}[^>]*/>", "", text)
    return text.strip()


def format_tool_use(block: dict) -> str:
    """Format a tool_use content block."""
    name = block.get("name", "unknown_tool")
    inp = block.get("input", {})
    lines = [f"**Tool Call: `{name}`**\n"]
    # Never truncate the prompt parameter for Task tool calls
    no_truncate_keys = set()
    if name == "Task":
        no_truncate_keys.add("prompt")
    for k, v in inp.items():
        v_str = str(v)
        if k not in no_truncate_keys and len(v_str) > 500:
            v_str = v_str[:500] + "... [truncated]"
        if "\n" in v_str and len(v_str) > 200:
            lines.append(f"- **{k}**:\n\n```\n{v_str}\n```")
        else:
            lines.append(f"- **{k}**: `{v_str}`")
    return "\n".join(lines)


def format_tool_result(block: dict) -> str:
    """Format a tool_result content block."""
    tool_use_id = block.get("tool_use_id", "")
    content = block.get("content", "")
    is_error = block.get("is_error", False)

    if isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(f"[{item.get('type', 'unknown')} content]")
        content = "\n".join(parts)

    prefix = "**Tool Error**" if is_error else "**Tool Result**"

    if len(content) > MAX_TOOL_OUTPUT_LEN:
        content = content[:MAX_TOOL_OUTPUT_LEN] + f"\n\n... [truncated, {len(content)} chars total]"

    return f"{prefix}\n\n```\n{content}\n```"


def convert_jsonl_to_md(jsonl_path: str, output_path: str | None = None) -> str:
    """Convert JSONL session file to markdown."""
    lines_out = []
    lines_out.append("# Claude Code Session Transcript\n")

    with open(jsonl_path) as f:
        raw_lines = f.readlines()

    session_id = None
    msg_num = 0

    for raw_line in raw_lines:
        obj = json.loads(raw_line.strip())
        obj_type = obj.get("type")

        if not session_id:
            session_id = obj.get("sessionId")

        # Only process user/assistant message entries
        if obj_type in ("file-history-snapshot", "progress", "system"):
            continue

        message = obj.get("message", {})
        role = message.get("role", "")
        content = message.get("content", "")

        if obj_type == "user":
            # Tool results come as list content blocks within user messages
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        lines_out.append(format_tool_result(block))
                        lines_out.append("")
                continue

            # Regular user message — strip system XML tags
            cleaned = strip_xml_tags(content) if isinstance(content, str) else str(content)
            if not cleaned:
                continue

            msg_num += 1
            lines_out.append(f"---\n\n## User (message {msg_num})\n")
            lines_out.append(cleaned)
            lines_out.append("")

        elif obj_type == "assistant":
            # Assistant content is a list of typed blocks (thinking, text, tool_use)
            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type")

                    if block_type == "thinking":
                        thinking_text = block.get("thinking", "")
                        if thinking_text:
                            lines_out.append("<details>\n<summary>Reasoning</summary>\n")
                            lines_out.append(thinking_text)
                            lines_out.append("\n</details>\n")

                    elif block_type == "text":
                        text = block.get("text", "").strip()
                        if text:
                            lines_out.append(f"### Assistant\n")
                            lines_out.append(text)
                            lines_out.append("")

                    elif block_type == "tool_use":
                        lines_out.append(format_tool_use(block))
                        lines_out.append("")

            elif isinstance(content, str) and content.strip():
                lines_out.append(f"### Assistant\n")
                lines_out.append(content.strip())
                lines_out.append("")

    # Add session metadata header
    header = f"**Session ID:** `{session_id}`\n" if session_id else ""
    lines_out.insert(1, header)

    md_content = "\n".join(lines_out)

    if output_path:
        Path(output_path).write_text(md_content)
        print(f"Written to {output_path}")
    else:
        print(md_content)

    return md_content


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: jsonl_to_md.py <input.jsonl> [output.md]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    convert_jsonl_to_md(inp, out)
