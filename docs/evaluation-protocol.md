# Judge evaluation protocol

This document specifies the common evaluation contract used by AgentHorizon.
Harness-specific implementations may retrieve and organize evidence
differently, but every reported configuration is scored against the same blind
trajectory task and fixed label file.

## Judge input and output

A judge receives:

- an opaque trajectory ID;
- the instruction to evaluate;
- the ordered action record; and
- screenshots referenced by the trajectory.

It does not receive the gold label, paired instruction ID, original delivery
ID, negative-construction direction, or reference failure type.

At minimum, a judge emits:

```json
{"success": false, "reasoning": "..."}
```

Configurations that predict a failure type may additionally emit the taxonomy
label. Missing, malformed, or non-Boolean `success` values count as incorrect;
they never reduce the evaluation denominator.

## Scoring

The primary metric is balanced accuracy:

```text
0.5 × (positive accuracy + negative accuracy)
```

Balanced accuracy prevents a configuration from benefiting from the class
prior of a particular difficulty subset. Positive and negative accuracy are
reported alongside it because two judges with the same balanced accuracy can
have very different acceptance behavior.

Failure-type recall is stricter than binary rejection. A prediction counts as
correct only if the judge emits `success=false` and matches the reference
failure type. The six retained negatives without a reference type are excluded
only from type-specific denominators, not from binary evaluation.

## Direct multimodal evaluation

Direct LLM-as-judge configurations receive one chat request per trajectory.
Screenshots are downscaled to a 512×332 target resolution. The default payload
preserves one image per decision step.

For image sequences that exceed a serving limit, four consecutive screenshots
are combined into a 2×2 row-major grid at 1024×664 pixels. The prompt states
the reading order, and the composite preserves the total visible pixel area.
This transformation changes packaging rather than the evidence shown to the
judge. Implementations should record whether merging was used for each task.

## Harnessed evaluation

Agent harnesses may differ in observation selection, context organization,
image handling, retries, and tool use. Results therefore identify both the
model and the harness. A reproducible run record should include:

- model/provider identifier and version;
- harness name and version;
- prompt revision;
- sampling parameters;
- screenshot preprocessing mode;
- per-task verdict and parse status; and
- the exact label manifest used for scoring.

Token counts, images viewed, tool calls, latency, and billed cost are telemetry,
not scoring denominators. Their coverage should be stated separately when a
harness does not expose a metric for every task.

## Recommended workflow

1. Download the released trajectory inputs and labels.
2. Render trajectories if pre-rendered Markdown is not used.
3. Run the judge without exposing the label file.
4. Save one result record per trajectory ID.
5. Score against the fixed manifest, counting invalid outputs as incorrect.
6. Preserve the environment, prompt, run configuration, raw verdicts, and
   aggregate report together.

See the repository [README](../README.md) for commands and
[STANDARD.md](../STANDARD.md) for the on-disk schema.
