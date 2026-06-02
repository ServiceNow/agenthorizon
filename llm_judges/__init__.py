"""Phase 2: LLM-as-Judge evaluation without agent harness scaffolding.

This package provides preprocessing pipelines and an evaluation runner for
direct LLM API calls to judge computer-use agent trajectories. Three
preprocessing approaches are supported:

    A) Naive compression -- all screenshots at reduced resolution
    B) Two-stage filtering -- select important steps, full-res screenshots
    C) Two-stage summarization -- vision model describes screenshots, text-only judge

See README.md for usage and architecture details.
"""
