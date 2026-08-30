# Benchmark construction and review

This document records how the human demonstrations become the released
AgentHorizon judging tasks. It separates the construction pool from the
reviewed release so that every denominator is explicit.

## Stage-by-stage counts

| Stage | Positive | Negative | Total |
|---|---:|---:|---:|
| Human demonstrations in 425 instruction pairs | 850 | — | 850 |
| Candidate judging tasks after instruction swapping | 850 | 850 | 1,700 |
| Confirmed positives after evidence review | 523 | — | 523 |
| Released benchmark | 523 | 850 | 1,373 |

Each instruction pair contains two closely related tasks, `A` and `B`, and a
successful human demonstration for each. The two candidate positives are
`instruction A + trajectory A` and `instruction B + trajectory B`. The two
adversarial negatives swap the instructions while preserving the recorded
trajectories: `instruction A + trajectory B` and `instruction B + trajectory
A`.

A positive task makes the strong claim that a trajectory satisfies its own
instruction. A negative task makes a different claim: the displayed
trajectory does not satisfy the instruction with which it is paired. The
negative therefore remains well-defined even if the same trajectory is not
retained as a confirmed positive under its original instruction.

## Review of candidate positives

Candidate positives pass through an initial quality review and a separate
evidence review over the instruction, action sequence, and screenshots.

1. The two assessments agree directly on 388 of the 850 candidate positives.
2. The remaining 462 cases are arbitrated against the complete evidence.
3. Arbitration retains 135 cases for which no material concern remains.
4. The remaining 327 material or unresolved cases are excluded rather than
   relabeled.

The released positive pool is therefore `388 + 135 = 523` items.

## Review of swapped negatives

The two swap directions can violate different requirements and can therefore
have different failure types. Reviewers re-examined all 850 swapped tasks one
direction at a time under the final rubric. This pass required 212.5
contributor-hours and resulted in 844 typed negatives plus 6 retained legacy
cases without a failure-type label. Those six items remain negatives; the
missing type is represented explicitly rather than silently changing a
denominator.

The review establishes that each swapped instruction is not satisfied by its
paired trajectory. It is not an independent third-party adjudication, and it
does not require the trajectory's original positive candidate to survive the
stricter positive-evidence review.

## Calibration and quality assurance

Collection proceeded iteratively over approximately seven weeks. Review
identified 404 items requiring some correction. Depending on the issue,
corrections could affect the instruction, subtask, event record, metadata, or
recording. Twenty-seven items lacked a replacement recording within the
collection window and were dropped. A final integrity pass normalized
failure-type spelling and repaired pair identifiers before the 425 complete
pairs were converted into the 1,700-task construction pool.

The following reliability quantities answer different questions and should not
be conflated:

- **Human inter-annotator agreement was not collected.** A three-reviewer
  diagnostic on 23 of 425 pairs was used to identify calibration issues, not
  to estimate population agreement.
- **Cross-pass label stability** compares the later failure-type review with
  the final QA labels. Cohen's kappa is 0.802 over all 850 negatives (including
  the untyped state) and 0.810 over the 844 negatives with final typed labels.
- **Residual model-based QA** examined 156 sampled tasks. A first model-based
  pass flagged 82; a second model-based review confirmed 18. The resulting
  18/156 estimate is 11.5% (Wilson 95% interval: 7.4%–17.5%). This is a
  model-based residual-QA estimate, not a human-validated error rate.

## Contributor effort

The annotation partner reported 6,077.5 contributor-hours over the 850 unique
demonstrations. This excludes onboarding, vetting, project setup,
post-processing, and project management.

| Stage | Average per demonstration | Total hours |
|---|---:|---:|
| Annotation | 4.00 h | 3,400.0 |
| First review | 1.70 h | 1,445.0 |
| Failure-type review | 0.25 h | 212.5 |
| Second-stage review | 1.20 h | 1,020.0 |
| **Delivered total** | **7.15 h** | **6,077.5** |

The original planning estimate was 1,960 hours (2.31 hours per demonstration).
The difference reflects the review passes added during calibration.

## Released representation

Labels are stored separately from trajectory inputs. A judge receives an
opaque trajectory ID, the instruction, screenshots, and actions, but not the
gold label, paired ID, original source ID, or failure type. See
[STANDARD.md](../STANDARD.md) for the trajectory and label schemas and
[mistake-taxonomy.md](mistake-taxonomy.md) for the failure-type rubric.
