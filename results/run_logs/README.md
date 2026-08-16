# Run logs

The two logs here are the **authoritative runs that produced every current number** in
`results/`, both from the 2026-08-17 correctness audit:

- `2026-08-17_rebuild_after_bugfixes.log` — rescored + re-guardrailed all five arms after the
  audit sampler, guardrail splitter, and fabricated-count fixes (Bugs A–C).
- `2026-08-17_langgraph_bedrock_maxtokens_rerun.log` — regenerated the LangGraph-Bedrock arm
  after the `max_tokens` fix (Bug E), then rebuilt every downstream number. **This is the run
  whose output is currently on disk.**

The previous per-pipeline `results/*.log` files were deleted: they were orphans from superseded
runs and three of them contained guardrail numbers (0.675 / 0.657 / 0.556) that contradict the
current results. They remain in git history if ever needed.
