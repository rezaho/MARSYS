"""Deterministic simulator for the MARSYS orchestration redesign.

See `implementations/077-2026-04-16-unified-branch-orchestration-plan.md` and
`/home/rezaho/.claude/plans/ok-i-will-now-graceful-shore.md` for context.

The simulator decouples the orchestration algorithm from agents, LLMs, and IO.
It exercises the algorithm against scripted traces that control timing and
concurrency explicitly, with ground-truth assertions.
"""
