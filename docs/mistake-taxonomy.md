# Failure-type annotation rubric

AgentHorizon assigns a failure type to each swapped negative independently.
The label describes why the displayed trajectory fails the instruction it is
being judged against; the two directions of one instruction pair need not have
the same label.

## Categories

| Type | Operational distinction | Typical recovery |
|---|---|---|
| **Critical Mistake** | The core objective is not achieved. | Restart or redo the task. |
| **Bad Side Effect** | The main objective is largely achieved, but execution creates a harmful, costly, risky, or externally visible consequence. | Undo unwanted work or coordinate non-trivial recovery. |
| **Misunderstanding of the Instruction** | The result is coherent and useful but misses a parameter or constraint without creating a substantial side effect. | Make a local edit or repeat a small step. |

### Critical Mistake

A Critical Mistake changes or defeats the core objective: the primary action,
entity, or subject. The requested outcome is absent, and continuing from the
current result does not satisfy the original task.

Example: the instruction asks for a flight on November 1, but the completed
booking is for November 11. The requested booking was not made, and the
purchase cannot be corrected as a simple edit.

### Bad Side Effect

A Bad Side Effect preserves most of the requested outcome but introduces an
unrequested consequence that is harmful, costly, risky, or burdensome. The
problem is not merely that a setting differs; the user must reverse unwanted
work or manage an effect beyond the immediate artifact.

Example: the requested internal finance summary is posted to the correct
channel and also forwarded to an external distribution list. The main action
occurred, but the unintended disclosure requires escalation and recovery.

### Misunderstanding of the Instruction

A Misunderstanding is a plausible execution based on a wrong interpretation
of a parameter or constraint. The output remains coherent, and the user can
usually correct it locally without material cost, risk, or external impact.

Example: the instruction asks for a high-resolution PNG, but the trajectory
exports a high-resolution JPEG. The artifact is usable and can be corrected by
changing the export format.

## Decision procedure

Reviewers apply the following questions in order.

1. **Was the core objective achieved?** If not, label **Critical Mistake**.
2. **Must unwanted work be deleted or reversed rather than locally edited?**
   If yes, label **Bad Side Effect**.
3. **Does the error affect another person or system, lose funds, create active
   risk, or require non-trivial recovery?** If yes, label **Bad Side Effect**.
4. Otherwise, label **Misunderstanding of the Instruction**.

The rubric treats core objective, parameters, and constraints as distinct
layers. Changes to the primary action or entity are critical. Parameter and
constraint errors are distinguished by their consequences: externality,
reversal burden, time, cost, and risk.

## Boundary checks

The following checks resolve most Bad Side Effect versus Misunderstanding
cases:

- **Delete versus edit:** reversing unwanted output indicates a side effect;
  changing a setting or word indicates a misunderstanding.
- **Blast radius:** impact beyond the user and current artifact indicates a
  side effect.
- **Recovery cost:** lost funds, additional people, active risk, or more than a
  few minutes of recovery indicates a side effect.

If none of these conditions applies and the core objective remains intact, the
failure is a Misunderstanding.
