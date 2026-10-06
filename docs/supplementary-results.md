# Supplementary results

This page preserves the exhaustive diagnostics that are useful for reproduction
but not necessary for reading the paper. The two model-interface tables below
use the **submitted Qwen-defined partition** (605 challenging items and 768
simple items). The revised paper uses a held-out development set and a pooled
three-splitter evaluation partition; these legacy tables remain available as a
sensitivity analysis rather than as the current leaderboard.

## Full model-interface grid: submitted challenging split

`MT` is exact mistake-type recall. Token, tool, and image values are means per
trajectory. The splitter row is affected mechanically by its role in defining
this legacy partition.

| Model | Interface | Bal. acc. | Pos. | Neg. | MT | Input tokens | Output tokens | Tools | Images |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.5 | Codex | 83.1 | 75.6 | 90.6 | 51.1 | 513,556 | 4,184 | 18.6 | 11.9 |
| Gemini 3.1 Pro | Gemini CLI | 79.4 | 67.6 | 91.2 | 52.1 | 1,393,271 | 9,811 | 24.2 | 30.2 |
| Claude Opus 4.7 | Claude Code | 77.6 | 78.4 | 76.7 | 44.1 | 554,387 | 5,655 | 9.2 | 7.8 |
| Qwen 3.6 27B | OpenCode | 72.0 | 77.7 | 66.4 | 36.7 | 243,440 | 4,405 | 8.1 | 6.0 |
| GPT-5.4 mini | Codex | 70.0 | 67.9 | 72.0 | 36.4 | 323,808 | 8,905 | 20.0 | 8.4 |
| Qwen 3.6 35B-A3B | OpenCode | 66.1 | 77.7 | 54.4 | 27.8 | 190,550 | 4,329 | 8.1 | 5.6 |
| Qwen 3.6 27B | OpenHands | 57.8 | 66.2 | 49.4 | 24.3 | 1,258,306 | 8,999 | 14.9 | 9.4 |
| Claude Haiku 4.5 | Claude Code | 56.0 | 48.4 | 63.5 | 28.1 | 452,266 | 5,423 | 10.6 | 8.3 |
| Gemini 3.1 Flash Lite | OpenCode | 48.8 | 40.1 | 57.5 | 20.8 | 115,860 | 771 | 3.4 | 1.1 |
| Gemini 3.1 Flash Lite | Gemini CLI | 48.3 | 55.4 | 41.2 | 11.5 | 1,041,455 | 708 | 8.1 | 3.6 |
| Gemini 3.1 Flash Lite | OpenHands | 47.3 | 28.6 | 66.0 | 13.7 | 175,274 | 2,540 | 5.7 | 0.9 |
| Gemma 4 26B-A4B | OpenCode | 47.1 | 69.7 | 24.5 | 5.8 | 243,892 | 909 | 3.9 | 1.2 |
| Gemini 3.1 Flash Lite | Codex | 45.3 | 35.9 | 54.7 | 14.4 | 160,174 | 654 | 5.7 | 1.1 |
| Gemma 4 31B | OpenCode | 44.5 | 60.6 | 28.3 | 8.6 | 101,528 | 372 | 2.3 | 1.0 |
| Qwen 3.5 122B-A10B (splitter) | OpenCode | 38.4 | 40.1 | 36.8 | 16.6 | 159,688 | 994 | 3.9 | — |

## Full model-interface grid: submitted simple split

| Model | Interface | Bal. acc. | Pos. | Neg. | MT | Input tokens | Output tokens | Tools | Images |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 3.5 122B-A10B (splitter) | OpenCode | 98.3 | 97.9 | 98.7 | 56.4 | 159,688 | 994 | 3.9 | — |
| Claude Opus 4.7 | Claude Code | 93.2 | 89.4 | 97.0 | 63.8 | 554,387 | 5,655 | 9.2 | 7.8 |
| Qwen 3.6 27B | OpenCode | 92.5 | 89.0 | 96.1 | 59.3 | 243,440 | 4,405 | 8.1 | 6.0 |
| GPT-5.5 | Codex | 91.1 | 83.5 | 98.7 | 66.9 | 513,556 | 4,184 | 18.6 | 11.9 |
| GPT-5.4 mini | Codex | 90.5 | 85.6 | 95.5 | 58.6 | 323,808 | 8,905 | 20.0 | 8.4 |
| Gemma 4 31B | OpenCode | 88.0 | 89.8 | 86.1 | 50.3 | 101,528 | 372 | 2.3 | 1.0 |
| Qwen 3.6 35B-A3B | OpenCode | 88.0 | 84.7 | 91.2 | 52.5 | 190,550 | 4,329 | 8.1 | 5.6 |
| Claude Haiku 4.5 | Claude Code | 86.5 | 77.1 | 95.9 | 51.4 | 452,266 | 5,423 | 10.6 | 8.3 |
| Gemini 3.1 Pro | Gemini CLI | 84.6 | 70.8 | 98.5 | 57.8 | 1,393,271 | 9,811 | 24.2 | 30.2 |
| Gemini 3.1 Flash Lite | OpenCode | 83.6 | 74.6 | 92.7 | 45.6 | 115,860 | 771 | 3.4 | 1.1 |
| Qwen 3.6 27B | OpenHands | 83.3 | 79.7 | 87.0 | 53.3 | 1,258,306 | 8,999 | 14.9 | 9.4 |
| Gemini 3.1 Flash Lite | Gemini CLI | 79.0 | 76.7 | 81.4 | 44.6 | 1,041,455 | 708 | 8.1 | 3.6 |
| Gemini 3.1 Flash Lite | Codex | 75.4 | 61.4 | 89.3 | 43.5 | 160,174 | 654 | 5.7 | 1.1 |
| Gemma 4 26B-A4B | OpenCode | 71.2 | 84.7 | 57.7 | 33.0 | 243,892 | 909 | 3.9 | 1.2 |
| Gemini 3.1 Flash Lite | OpenHands | 68.7 | 46.6 | 90.8 | 40.5 | 175,274 | 2,540 | 5.7 | 0.9 |

## Full-benchmark mistake-type recall

A prediction is correct only when the judge emits `success=false` and identifies
the reference failure type. The fixed denominators are 273 Critical Mistakes,
232 Bad Side Effects, and 339 Misunderstandings; missing or malformed outputs
count as incorrect.

| Model | Interface | Critical | Bad Side Effect | Misunderstanding |
|---|---|---:|---:|---:|
| Gemini 3.1 Pro | Gemini CLI | 57.5 (157/273) | 7.3 (17/232) | 87.3 (296/339) |
| GPT-5.5 | Codex | 77.7 (212/273) | 24.1 (56/232) | 72.9 (247/339) |
| Claude Opus 4.7 | Claude Code | 75.8 (207/273) | 12.1 (28/232) | 71.4 (242/339) |
| GPT-5.4 mini | Codex | 60.8 (166/273) | 2.6 (6/232) | 74.6 (253/339) |
| Qwen 3.6 27B | OpenCode | 60.4 (165/273) | 5.6 (13/232) | 74.3 (252/339) |
| Claude Haiku 4.5 | Claude Code | 58.6 (160/273) | 8.6 (20/232) | 53.4 (181/339) |
| Qwen 3.5 122B-A10B | OpenCode | 66.7 (182/273) | 0.9 (2/232) | 49.3 (167/339) |
| Gemini 3.1 Flash Lite | OpenHands | 60.8 (166/273) | 0.0 (0/232) | 27.1 (92/339) |
| Gemini 3.1 Flash Lite | Codex | 64.1 (175/273) | 0.0 (0/232) | 29.8 (101/339) |
| Gemini 3.1 Flash Lite | OpenCode | 50.2 (137/273) | 1.3 (3/232) | 49.3 (167/339) |
| Qwen 3.6 35B-A3B | OpenCode | 63.0 (172/273) | 6.0 (14/232) | 53.1 (180/339) |
| Qwen 3.6 27B | OpenHands | 56.8 (155/273) | 3.4 (8/232) | 57.8 (196/339) |
| Gemini 3.1 Flash Lite | Gemini CLI | 62.3 (170/273) | 1.7 (4/232) | 29.2 (99/339) |
| Gemma 4 31B | OpenCode | 74.0 (202/273) | 0.0 (0/232) | 27.1 (92/339) |
| Gemma 4 26B-A4B | OpenCode | 47.6 (130/273) | 0.0 (0/232) | 18.6 (63/339) |
